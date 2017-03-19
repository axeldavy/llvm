//===-- SIMachineScheduler.cpp - SI Scheduler Interface -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief SI Machine Scheduler interface
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "SIMachineScheduler.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "misched"

// This scheduler implements a different scheduling algorithm than
// GenericScheduler.
//
// There are several specific architecture behaviours that can't be modelled
// for GenericScheduler:
// . When accessing the result of an SGPR load instruction, you have to wait
// for all the SGPR load instructions before your current instruction to
// have finished.
// . When accessing the result of an VGPR load instruction, you have to wait
// for all the VGPR load instructions previous to the VGPR load instruction
// you are interested in to finish.
// . The less the register pressure, the best load latencies are hidden
//
// Moreover some specifities (like the fact a lot of instructions in the shader
// have few dependencies) makes the generic scheduler have some unpredictable
// behaviours. For example when register pressure becomes high, it can either
// manage to prevent register pressure from going too high, or it can
// increase register pressure even more than if it hadn't taken register
// pressure into account.
//
// Also some other bad behaviours are generated, like loading at the beginning
// of the shader a constant in VGPR you won't need until the end of the shader.
//
// The scheduling problem for SI can distinguish three main parts:
// . Hiding high latencies (texture sampling, etc)
// . Hiding low latencies (SGPR constant loading, etc)
// . Keeping register usage low for better latency hiding and general
//   performance
//
// Some other things can also affect performance, but are hard to predict
// (cache usage, the fact the HW can issue several instructions from different
// wavefronts if different types, etc)
//
// This scheduler tries to solve the scheduling problem by dividing it into
// simpler sub-problems. It divides the instructions into blocks, schedules
// locally inside the blocks where it takes care of low latencies, and then
// chooses the order of the blocks by taking care of high latencies.
// Dividing the instructions into blocks helps control keeping register
// usage low.
//
// First the instructions are put into blocks.
//   We want the blocks help control register usage and hide high latencies
//   later. To help control register usage, we typically want all local
//   computations, when for example you create a result that can be comsummed
//   right away, to be contained in a block. Block inputs and outputs would
//   typically be important results that are needed in several locations of
//   the shader. Since we do want blocks to help hide high latencies, we want
//   the instructions inside the block to have a minimal set of dependencies
//   on high latencies. It will make it easy to pick blocks to hide specific
//   high latencies.
//   The block creation algorithm is divided into several steps, and several
//   variants can be tried during the scheduling process.
//
// Second the order of the instructions inside the blocks is chosen.
//   At that step we do take into account only register usage and hiding
//   low latency instructions
//
// Third the block order is chosen, there we try to hide high latencies
// and keep register usage low.
//
// After the third step, a pass is done to improve the hiding of low
// latencies.
//
// Actually when talking about 'low latency' or 'high latency' it includes
// both the latency to get the cache (or global mem) data go to the register,
// and the bandwidth limitations.
// Increasing the number of active wavefronts helps hide the former, but it
// doesn't solve the latter, thus why even if wavefront count is high, we have
// to try have as many instructions hiding high latencies as possible.
// The OpenCL doc says for example latency of 400 cycles for a global mem access,
// which is hidden by 10 instructions if the wavefront count is 10.

// Some figures taken from AMD docs:
// Both texture and constant L1 caches are 4-way associative with 64 bytes
// lines.
// Constant cache is shared with 4 CUs.
// For texture sampling, the address generation unit receives 4 texture
// addresses per cycle, thus we could expect texture sampling latency to be
// equivalent to 4 instructions in the very best case (a VGPR is 64 work items,
// instructions in a wavefront group are executed every 4 cycles),
// or 16 instructions if the other wavefronts associated to the 3 other VALUs
// of the CU do texture sampling too. (Don't take these figures too seriously,
// as I'm not 100% sure of the computation)
// Data exports should get similar latency.
// For constant loading, the cache is shader with 4 CUs.
// The doc says "a throughput of 16B/cycle for each of the 4 Compute Unit"
// I guess if the other CU don't read the cache, it can go up to 64B/cycle.
// It means a simple s_buffer_load should take one instruction to hide, as
// well as a s_buffer_loadx2 and potentially a s_buffer_loadx8 if on the same
// cache line.
//
// As of today the driver doesn't preload the constants in cache, thus the
// first loads get extra latency. The doc says global memory access can be
// 300-600 cycles. We do not specially take that into account when scheduling
// As we expect the driver to be able to preload the constants soon.

// common code //

SISchedulerRPTracker::SISchedulerRPTracker(
    const SmallVectorImpl<RegisterMaskPair> &LiveIns,
    const SmallVectorImpl<RegisterMaskPair> &LiveOuts,
    const std::vector<SmallVector<unsigned, 8>> &ItemSuccs,
    const std::vector<SmallVector<unsigned, 8>> &ItemPreds,
    const std::vector<SmallVector<RegisterMaskPair, 8>> &InRegsForItem_,
    const std::vector<SmallVector<RegisterMaskPair, 8>> &OutRegsForItem_,
    const MachineRegisterInfo *MRI,
    const TargetRegisterInfo *TRI,
    unsigned VGPRSetID,
    unsigned SGPRSetID
  ):
  MRI(MRI), TRI(TRI), VGPRSetID(VGPRSetID), SGPRSetID(SGPRSetID),
  ItemSuccs(ItemSuccs), ItemPreds(ItemPreds), CurrentVGPRUsage(0),
  CurrentSGPRUsage(0)
{
  unsigned DAGSize = ItemSuccs.size();

  fillTopoData();

  // To track register usage, we define for each register definition
  // a number of usages before it gets released.
  // This doesn't work with LaneMasks.
  // To handle LaneMasks, we 'cut' registers affected by LaneMasks
  // into all their different Lanes possible
  // and behave as if that (reg, LaneMask) was a register.
  std::map<unsigned, SmallVector<LaneBitmask, 8>> RegWithLaneMask;
  auto addRegToRegWithLaneMask = [&](const RegisterMaskPair &P) {
    if (!P.LaneMask.all() &&
        P.LaneMask != MRI->getMaxLaneMaskForVReg(P.RegUnit))
      RegWithLaneMask[P.RegUnit].push_back(P.LaneMask);
  };

  std::for_each(LiveIns.begin(), LiveIns.end(), addRegToRegWithLaneMask);
  std::for_each(LiveOuts.begin(), LiveOuts.end(), addRegToRegWithLaneMask);
  for (const auto &InRegs : InRegsForItem_)
    std::for_each(InRegs.begin(), InRegs.end(), addRegToRegWithLaneMask);
  for (const auto &OutRegs : OutRegsForItem_)
    std::for_each(OutRegs.begin(), OutRegs.end(), addRegToRegWithLaneMask);

  // Since we ignored when the lane mask was getMaxLaneMaskForVReg,
  // we need to add it back. It doesn't hurt if there was no element
  // with this mask for this register.
  for (auto &RegLaneMasks : RegWithLaneMask) {
    RegLaneMasks.second.push_back(MRI->getMaxLaneMaskForVReg(RegLaneMasks.first));
  }

  // Fills LaneMaskBasisForReg
  for (const auto &RegLaneMasks : RegWithLaneMask) {
    SmallVector<LaneBitmask, 8> &LaneBasis =
      LaneMaskBasisForReg[RegLaneMasks.first];
    for (const LaneBitmask &LaneMask : RegLaneMasks.second) {
      LaneBitmask Remaining = LaneMask;
      for (auto I = LaneBasis.begin(); I != LaneBasis.end(); ++I) {
        LaneBitmask Elem = *I;
        if ((Remaining & Elem).none())
          continue;
        if ((Remaining & Elem) == Elem) {
          Remaining &= ~Elem;
          continue;
        }
        // Remaining intersects with Elem, but Elem is not
        // included in remaining. We divide Elem into two elements.
        // The one included in Remaining, and the rest.
        LaneBitmask NewElem = Elem & ~Remaining;
        *I = Elem & Remaining;
        LaneBasis.push_back(NewElem);
      }
      if (Remaining.any())
        LaneBasis.push_back(Remaining);
    }
  }

  auto convertRegs = [=](const SmallVectorImpl<RegisterMaskPair> &Regs) {
    return getPairsForRegs(Regs);
  };

  InRegsForItem.resize(DAGSize);
  OutRegsForItem.resize(DAGSize);
  std::transform(InRegsForItem_.begin(), InRegsForItem_.end(),
                 InRegsForItem.begin(), convertRegs);
  std::transform(OutRegsForItem_.begin(), OutRegsForItem_.end(),
                 OutRegsForItem.begin(), convertRegs);

  // Add InRegs to LiveRegs after we add missing LiveIns
  SmallVector<RegisterMaskPair, 8> InRegs = getPairsForRegs(LiveIns);

  // Fill the usage of every output
  // Warning: while by construction we always have a link between two items
  // when one needs a result from the other, the number of users of an output
  // is not the sum of child items having as input the same virtual register.
  // Here is an example. A produces x and y. B eats x and produces x'.
  // C eats x' and y. The register coalescer may have attributed the same
  // virtual register to x and x'.
  // To count accurately, we do a topological sort. In case the register is
  // found for several parents, we increment the usage of the one with the
  // highest topological index.
  OutRegsNumUsages.resize(DAGSize);
  for (unsigned i = 0; i < DAGSize; i++) {
    for (const auto &RegPair : InRegsForItem[i]) {
      bool Found = false;
      int topoInd = -1;
      for (unsigned PredID : ItemPreds[i]) {
        const auto &PredOutRegs =
          this->OutRegsForItem[PredID];
        for (const auto &RegPair2 : PredOutRegs) {
          if (RegPair == RegPair2) {
            Found = true;
            if (topoInd < (int)TopoItemToIndex[PredID]) {
              topoInd = TopoItemToIndex[PredID];
            }
            break;
          }
        }
      }

      if (!Found) {
        // Fill RemainingRegsConsumers for regs that were already
        // defined before scheduling.
        ++RemainingRegsConsumers[RegPair];
        // Workaround for incomplete liveIns: Add missing liveIns.
        // addLiveRegs is noop if already Live.
        InRegs.push_back(RegPair);
      }
      else {
        unsigned PredID = TopoIndexToItem[topoInd];
        ++OutRegsNumUsages[PredID][RegPair];
      }
    }
  }

  addLiveRegs(InRegs);

  ItemNumPredsLeft.resize(DAGSize);

  for (unsigned i = 0; i < DAGSize; i++) {
    unsigned NumPreds = ItemPreds[i].size();
    ItemNumPredsLeft[i] = NumPreds;
    if (NumPreds == 0)
      ReadyItems.push_back(i);
  }

  // Increase OutRegsNumUsages for items
  // producing registers consumed in another
  // scheduling region.
  for (const RegisterMaskPair &RegPair : getPairsForRegs(LiveOuts)) {
    for (unsigned i = 0; i < DAGSize; i++) {
      // Do reverse traversal
      bool Found = false;
      int ID = TopoIndexToItem[DAGSize-1-i];
      const auto &OutRegs =
        this->OutRegsForItem[ID];

      for (const auto &RegPair2 : OutRegs) {
        if (RegPair == RegPair2) {
          Found = true;
          break;
        }
      }

      if (!Found)
        continue;

      ++OutRegsNumUsages[ID][RegPair];
      break;
    }
  }
}

void
SISchedulerRPTracker::getCurrentRegUsage(unsigned &VGPR, unsigned &SGPR)
{
  VGPR = CurrentVGPRUsage;
  SGPR = CurrentSGPRUsage;
}

void
SISchedulerRPTracker::checkRegUsageImpact(unsigned ID,
                                          int &DiffVGPR,
                                          int &DiffSGPR) {
  SmallDenseMap<unsigned, LaneBitmask> Map;
  SmallDenseSet<unsigned> Set;

  DiffVGPR = 0;
  DiffSGPR = 0;

  for (const auto &RegPair : InRegsForItem[ID]) {
    // For now only track virtual registers.
    unsigned Reg = RegPair.RegUnit;
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    if (RemainingRegsConsumers[RegPair] > 1)
      continue;
    Map[Reg] |= RegPair.LaneMask;
  }

  for (const auto &RegPair : Map) {
    if (LiveRegs[RegPair.first] == RegPair.second) {
      PSetIterator PSetI = MRI->getPressureSets(RegPair.first);
      for (; PSetI.isValid(); ++PSetI) {
        if (*PSetI == VGPRSetID)
          DiffVGPR -= PSetI.getWeight();
        if (*PSetI == SGPRSetID)
          DiffSGPR -= PSetI.getWeight();
      }
    }
  }

  for (const auto &RegPair : OutRegsForItem[ID]) {
    // For now only track virtual registers.
    unsigned Reg = RegPair.RegUnit;
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    Set.insert(Reg);
  }

  for (unsigned Reg : Set) {
    // Check register is not already alive (at least some lanes)
    if (LiveRegs.find(Reg) == LiveRegs.end()) {
      PSetIterator PSetI = MRI->getPressureSets(Reg);
      for (; PSetI.isValid(); ++PSetI) {
        if (*PSetI == VGPRSetID)
          DiffVGPR += PSetI.getWeight();
        if (*PSetI == SGPRSetID)
          DiffSGPR += PSetI.getWeight();
      }
    }
  }
}

#ifndef NDEBUG

void
SISchedulerRPTracker::printDebugLives()
{
  for (const auto &RegPair : LiveRegs) {
    dbgs() << PrintVRegOrUnit(RegPair.first, TRI);
    if (!RegPair.second.all())
      dbgs() << ':' << PrintLaneMask(RegPair.second);
   dbgs() << ' ';
  }
}

#endif

SmallVector<RegisterMaskPair, 8>
SISchedulerRPTracker::getPairsForReg(unsigned Reg, LaneBitmask Mask)
{
  SmallVector<RegisterMaskPair, 8> Result;

  getPairsForReg(Result, Reg, Mask);

  return Result;
}

void
SISchedulerRPTracker::getPairsForReg(SmallVector<RegisterMaskPair, 8> &ToAppend,
                                     unsigned Reg, LaneBitmask Mask)
{
  auto Basis = LaneMaskBasisForReg.find(Reg);
  if (Basis == LaneMaskBasisForReg.end()) {
    assert(Mask.all() || Mask == MRI->getMaxLaneMaskForVReg(Reg));
    // We want unicity of the RegisterMaskPair for a same register/mask
    // Thus replace getMaxLaneMaskForVReg by all, since they have the same
    // meaning.
    // Note: Physical registers have Mask.all(), but are disallowed
    // to call getMaxLaneMaskForVReg.
    if (!Mask.all() && Mask == MRI->getMaxLaneMaskForVReg(Reg))
      Mask = LaneBitmask::getAll();
    ToAppend.push_back(RegisterMaskPair(Reg, Mask));
  } else {
    for (const auto &Elem : Basis->second) {
      if ((Mask & Elem).any()) {
        assert((Mask & Elem) == Elem);
        ToAppend.push_back(RegisterMaskPair(Reg, Elem));
        Mask &= ~Elem;
      }
    }
    // Mask.all will have a non-none value.
    // We want Mask.all equivalent to the max lane mask.
    assert((Mask & MRI->getMaxLaneMaskForVReg(Reg)).none());
  }
}

SmallVector<RegisterMaskPair, 8>
SISchedulerRPTracker::getPairsForRegs(const SmallVectorImpl<RegisterMaskPair> &Regs)
{
  SmallVector<RegisterMaskPair, 8> Result;

  std::for_each(Regs.begin(), Regs.end(),
                [&](const RegisterMaskPair &RegPair){
                  getPairsForReg(Result, RegPair.RegUnit,
                                 RegPair.LaneMask);
                });

  return Result;
}

void
SISchedulerRPTracker::fillTopoData()
{
  unsigned DAGSize = ItemSuccs.size();
  std::vector<int> WorkList;

  DEBUG(dbgs() << "Topological Sort\n");

  WorkList.reserve(DAGSize);
  TopoIndexToItem.resize(DAGSize);
  TopoItemToIndex.resize(DAGSize);

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    unsigned Degree = ItemSuccs[i].size();
    TopoItemToIndex[i] = Degree;
    if (Degree == 0) {
      WorkList.push_back(i);
    }
  }

  int Id = DAGSize;
  while (!WorkList.empty()) {
    int i = WorkList.back();
    WorkList.pop_back();
    TopoItemToIndex[i] = --Id;
    TopoIndexToItem[Id] = i;
    for (unsigned PredID : ItemPreds[i]) {
      if (!--TopoItemToIndex[PredID])
        WorkList.push_back(PredID);
    }
  }

#ifndef NDEBUG
  // Check correctness of the ordering.
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    for (unsigned PredID : ItemPreds[i]) {
      assert(TopoItemToIndex[i] > TopoItemToIndex[PredID] &&
      "Wrong Top Down topological sorting");
    }
  }
#endif
}

void
SISchedulerRPTracker::addLiveRegs(const SmallVectorImpl<RegisterMaskPair> &Regs) {
  for (const RegisterMaskPair &RegPair : Regs) {
    unsigned Reg = RegPair.RegUnit;
    // For now only track virtual registers.
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    auto Pos = LiveRegs.find(Reg);

    if (Pos == LiveRegs.end()) {
      assert(RegPair.LaneMask.any());
      LiveRegs.insert(std::make_pair(Reg, RegPair.LaneMask));
      PSetIterator PSetI = MRI->getPressureSets(Reg);
      for (; PSetI.isValid(); ++PSetI) {
        if (*PSetI == VGPRSetID)
          CurrentVGPRUsage += PSetI.getWeight();
        if (*PSetI == SGPRSetID)
          CurrentSGPRUsage += PSetI.getWeight();
      }
    }
    else {
      Pos->second |= RegPair.LaneMask;
    }
  }
}

void
SISchedulerRPTracker::decreaseLiveRegs(const SmallVectorImpl<RegisterMaskPair> &Regs) {
  for (const RegisterMaskPair &RegPair : Regs) {
    // For now only track virtual registers.
    unsigned Reg = RegPair.RegUnit;
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    std::map<unsigned, LaneBitmask>::iterator Pos = LiveRegs.find(Reg);
    assert(Pos != LiveRegs.end() && // Reg must be live.
           RemainingRegsConsumers.find(RegPair) !=
             RemainingRegsConsumers.end() &&
           RemainingRegsConsumers[RegPair] >= 1);
    --RemainingRegsConsumers[RegPair];
    if (RemainingRegsConsumers[RegPair] == 0) {
      if (Pos->second == RegPair.LaneMask) {
        LiveRegs.erase(Pos);
        PSetIterator PSetI = MRI->getPressureSets(Reg);
        for (; PSetI.isValid(); ++PSetI) {
          if (*PSetI == VGPRSetID)
            CurrentVGPRUsage -= PSetI.getWeight();
          if (*PSetI == SGPRSetID)
            CurrentSGPRUsage -= PSetI.getWeight();
        }
      }
      else
        Pos->second &= ~RegPair.LaneMask;
    }
  }
}

void
SISchedulerRPTracker::releaseItemSuccs(unsigned ID) {
  for (unsigned SuccID : ItemSuccs[ID]) {
    if (--ItemNumPredsLeft[SuccID] == 0)
      ReadyItems.push_back(SuccID);
  }
}

void
SISchedulerRPTracker::itemScheduled(unsigned ID) {
  auto ReadyItemPos = std::find(ReadyItems.begin(), ReadyItems.end(), ID);
  assert(ReadyItemPos != ReadyItems.end());
  ReadyItems.erase(ReadyItemPos);
  decreaseLiveRegs(InRegsForItem[ID]);
  addLiveRegs(OutRegsForItem[ID]);
  releaseItemSuccs(ID);
  for (std::map<RegisterMaskPair, unsigned>::iterator RegI =
       OutRegsNumUsages[ID].begin(),
       E = OutRegsNumUsages[ID].end(); RegI != E; ++RegI) {
    std::pair<RegisterMaskPair, unsigned> RegP = *RegI;
    if (!TargetRegisterInfo::isVirtualRegister(RegP.first.RegUnit))
      continue;
    // We produce this register, thus it must not be previously alive.
    assert(RemainingRegsConsumers.find(RegP.first) ==
             RemainingRegsConsumers.end() ||
           RemainingRegsConsumers[RegP.first] == 0);
    RemainingRegsConsumers[RegP.first] += RegP.second;
  }
}

#ifndef NDEBUG

static const char *getReasonStr(SIScheduleCandReason Reason) {
  switch (Reason) {
  case NoCand:         return "NOCAND";
  case RegUsage:       return "REGUSAGE";
  case Latency:        return "LATENCY";
  case Successor:      return "SUCCESSOR";
  case Depth:          return "DEPTH";
  case NodeOrder:      return "ORDER";
  }
  llvm_unreachable("Unknown reason!");
}

#endif

static bool tryLess(int TryVal, int CandVal,
                    SISchedulerCandidate &TryCand,
                    SISchedulerCandidate &Cand,
                    SIScheduleCandReason Reason) {
  if (TryVal < CandVal) {
    TryCand.Reason = Reason;
    return true;
  }
  if (TryVal > CandVal) {
    if (Cand.Reason > Reason)
      Cand.Reason = Reason;
    return true;
  }
  Cand.setRepeat(Reason);
  return false;
}

static bool tryGreater(int TryVal, int CandVal,
                       SISchedulerCandidate &TryCand,
                       SISchedulerCandidate &Cand,
                       SIScheduleCandReason Reason) {
  if (TryVal > CandVal) {
    TryCand.Reason = Reason;
    return true;
  }
  if (TryVal < CandVal) {
    if (Cand.Reason > Reason)
      Cand.Reason = Reason;
    return true;
  }
  Cand.setRepeat(Reason);
  return false;
}

// SIScheduleBlock //

void SIScheduleBlock::addUnit(SUnit *SU) {
  NodeNum2Index[SU->NodeNum] = SUnits.size();
  SUnits.push_back(SU);
}

#ifndef NDEBUG
void SIScheduleBlock::traceCandidate(const SISchedCandidate &Cand) {

  dbgs() << "  SU(" << Cand.SU->NodeNum << ") " << getReasonStr(Cand.Reason);
  dbgs() << '\n';
}
#endif

void SIScheduleBlock::tryCandidateTopDown(SISchedCandidate &Cand,
                                          SISchedCandidate &TryCand) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return;
  }

  if (Cand.SGPRUsage > 60 &&
      tryLess(TryCand.SGPRUsage, Cand.SGPRUsage, TryCand, Cand, RegUsage))
    return;

  // Schedule low latency instructions as top as possible.
  // Order of priority is:
  // . Low latency instructions which do not depend on other low latency
  //   instructions we haven't waited for
  // . Other instructions which do not depend on low latency instructions
  //   we haven't waited for
  // . Low latencies
  // . All other instructions
  // Goal is to get: low latency instructions - independent instructions
  //     - (eventually some more low latency instructions)
  //     - instructions that depend on the first low latency instructions.
  // If in the block there is a lot of constant loads, the SGPR usage
  // could go quite high, thus above the arbitrary limit of 60 will encourage
  // use the already loaded constants (in order to release some SGPRs) before
  // loading more.
  if (tryLess(TryCand.HasLowLatencyNonWaitedParent,
              Cand.HasLowLatencyNonWaitedParent,
              TryCand, Cand, SIScheduleCandReason::Depth))
    return;

  if (tryGreater(TryCand.IsLowLatency, Cand.IsLowLatency,
                 TryCand, Cand, SIScheduleCandReason::Depth))
    return;

  if (TryCand.IsLowLatency &&
      tryLess(TryCand.LowLatencyOffset, Cand.LowLatencyOffset,
              TryCand, Cand, SIScheduleCandReason::Depth))
    return;

  if (tryLess(TryCand.VGPRUsage, Cand.VGPRUsage, TryCand, Cand, RegUsage))
    return;

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum < Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
  }
}

SUnit* SIScheduleBlock::pickNode() {
  SmallVector<unsigned, 16> TopReadySUs = RPTracker->getReadyItems();
  SISchedCandidate TopCand;
  unsigned VGPRCurrentUsage, SGPRCurrentUsage;
  RPTracker->getCurrentRegUsage(VGPRCurrentUsage, SGPRCurrentUsage);

  for (unsigned ID : TopReadySUs) {
    SUnit *SU = SUnits[ID];
    SISchedCandidate TryCand;
    int VGPRDiff, SGPRDiff;
    TryCand.SU = SU;
    RPTracker->checkRegUsageImpact(ID, VGPRDiff, SGPRDiff);
    TryCand.SGPRUsage = SGPRCurrentUsage + SGPRDiff;
    TryCand.VGPRUsage = VGPRCurrentUsage + VGPRDiff;
    TryCand.IsLowLatency = DAG->IsLowLatencySU[SU->NodeNum];
    TryCand.LowLatencyOffset = DAG->LowLatencyOffset[SU->NodeNum];
    TryCand.HasLowLatencyNonWaitedParent =
      HasLowLatencyNonWaitedParent[NodeNum2Index[SU->NodeNum]];
    tryCandidateTopDown(TopCand, TryCand);
    if (TryCand.Reason != NoCand)
      TopCand.setBest(TryCand);
  }

  return TopCand.SU;
}

static void addRegLanes(SmallVectorImpl<RegisterMaskPair> &RegUnits,
                        RegisterMaskPair Pair) {
  unsigned RegUnit = Pair.RegUnit;
  assert(Pair.LaneMask.any());
  auto I = llvm::find_if(RegUnits, [RegUnit](const RegisterMaskPair Other) {
    return Other.RegUnit == RegUnit;
  });
  if (I == RegUnits.end()) {
    RegUnits.push_back(Pair);
  } else {
    I->LaneMask |= Pair.LaneMask;
  }
}

static void getDefsForSU(SmallVector<RegisterMaskPair, 8> &Defs,
                         const SUnit &SU, const TargetRegisterInfo *TRI,
                         bool ShouldTrackLaneMasks)
{
  for (const MachineOperand &MO : SU.getInstr()->operands()) {
    if (MO.isReg() && MO.isDef() && !MO.isDead()) {
      unsigned Reg = MO.getReg();
      // For now only track virtual registers
      if (!TargetRegisterInfo::isVirtualRegister(Reg))
        continue;
      if (!ShouldTrackLaneMasks)
        addRegLanes(Defs, RegisterMaskPair(Reg, LaneBitmask::getAll()));
      else {
        unsigned SubRegIdx = MO.getSubReg();
        if (MO.isUndef())
          SubRegIdx = 0;
        addRegLanes(Defs, RegisterMaskPair(Reg,
          SubRegIdx != 0 ?
            TRI->getSubRegIndexLaneMask(SubRegIdx) :
            LaneBitmask::getAll()));
      }
    }
  }
}

// Here one big difference with RegOpers.collect is that
// we don't count in the Uses the Defs with readsReg() when
// ShouldTrackLaneMasks is false.
// We don't choose not to count them because the undef flags
// are not always set properly (we would need to schedule first
// and update LiveIntervals to get them correct).
// For better tracking, we prefer to miss some uses, rather than
// having incorrect uses.
static void getUsesForSU(SmallVector<RegisterMaskPair, 8> &Uses,
                         const SUnit &SU, const TargetRegisterInfo *TRI,
                         bool ShouldTrackLaneMasks)
{
  for (const MachineOperand &MO : SU.getInstr()->operands()) {
    if (MO.isReg() && MO.isUse() && !MO.isUndef() && !MO.isInternalRead()) {
      unsigned Reg = MO.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(Reg))
        continue;
      if (!ShouldTrackLaneMasks)
        addRegLanes(Uses, RegisterMaskPair(Reg, LaneBitmask::getAll()));
      else {
        unsigned SubRegIdx = MO.getSubReg();
        if (MO.isUndef())
          SubRegIdx = 0;
        addRegLanes(Uses, RegisterMaskPair(Reg,
          SubRegIdx != 0 ?
            TRI->getSubRegIndexLaneMask(SubRegIdx) :
            LaneBitmask::getAll()));
      }
    }
  }
}

void SIScheduleBlock::schedule() {
  std::vector<SmallVector<unsigned, 8>> SUSuccs;
  std::vector<SmallVector<unsigned, 8>> SUPreds;
  std::vector<SmallVector<RegisterMaskPair, 8>> InRegsForSU;
  std::vector<SmallVector<RegisterMaskPair, 8>> OutRegsForSU;

  SUSuccs.resize(SUnits.size());
  SUPreds.resize(SUnits.size());
  InRegsForSU.resize(SUnits.size());
  OutRegsForSU.resize(SUnits.size());

  for (unsigned i = 0; i < SUnits.size(); i++) {
    SUnit *SU = SUnits[i];

    for (SDep& SuccDep : SU->Succs) {
      SUnit *Succ = SuccDep.getSUnit();
      if (Succ->isBoundaryNode() || !BC->isSUInBlock(Succ, ID))
        continue;
      if (SuccDep.isWeak())
        continue;
      SUSuccs[i].push_back(NodeNum2Index[Succ->NodeNum]);
    }
    for (SDep& PredDep : SU->Preds) {
      SUnit *Pred = PredDep.getSUnit();
      if (Pred->isBoundaryNode() || !BC->isSUInBlock(Pred, ID))
        continue;
      if (PredDep.isWeak())
        continue;
      SUPreds[i].push_back(NodeNum2Index[Pred->NodeNum]);
    }
    getUsesForSU(InRegsForSU[i], *SU, DAG->getTRI(),
                 DAG->shouldTrackLaneMasks());
    getDefsForSU(OutRegsForSU[i], *SU, DAG->getTRI(),
                 DAG->shouldTrackLaneMasks());
  }

  RPTracker.reset(new SISchedulerRPTracker(
    LiveInRegs,
    LiveOutRegs,
    SUSuccs,
    SUPreds,
    InRegsForSU,
    OutRegsForSU,
    DAG->getMRI(),
    DAG->getTRI(),
    DAG->getVGPRSetID(),
    DAG->getSGPRSetID()
  ));

  while (!RPTracker->getReadyItems().empty()) {
    SUnit *SU = pickNode();
    ScheduledSUnits.push_back(SU);
    nodeScheduled(SU);
  }

  // TODO: compute InternalAdditionnalPressure.
  InternalAdditionnalPressure.resize(DAG->getTRI()->getNumRegPressureSets());

  // Check everything is right.
#ifndef NDEBUG
  assert(SUnits.size() == ScheduledSUnits.size() &&
         RPTracker->getReadyItems().empty());
#endif
}

void SIScheduleBlock::nodeScheduled(SUnit *SU) {
  RPTracker->itemScheduled(NodeNum2Index[SU->NodeNum]);
  // Scheduling this node will trigger a wait,
  // thus propagate to other instructions that they do not need to wait either.
  if (HasLowLatencyNonWaitedParent[NodeNum2Index[SU->NodeNum]])
    HasLowLatencyNonWaitedParent.assign(SUnits.size(), 0);

  if (DAG->IsLowLatencySU[SU->NodeNum]) {
     for (SDep& Succ : SU->Succs) {
      std::map<unsigned, unsigned>::iterator I =
        NodeNum2Index.find(Succ.getSUnit()->NodeNum);
      if (I != NodeNum2Index.end())
        HasLowLatencyNonWaitedParent[I->second] = 1;
    }
  }
}

void SIScheduleBlock::finalize() {
  for (SUnit* SU : SUnits) {
    if (DAG->IsHighLatencySU[SU->NodeNum])
      HighLatencyBlock = true;
  }
  HasLowLatencyNonWaitedParent.resize(SUnits.size(), 0);
  schedule();
}

// we maintain ascending order of IDs
void SIScheduleBlock::addPred(SIScheduleBlock *Pred) {
  unsigned PredID = Pred->getID();

  // Check if not already predecessor.
  for (SIScheduleBlock* P : Preds) {
    if (PredID == P->getID())
      return;
  }
  Preds.push_back(Pred);

  assert(none_of(Succs,
                 [=](std::pair<SIScheduleBlock*,
                     SIScheduleBlockLinkKind> S) {
                   return PredID == S.first->getID();
                    }) &&
         "Loop in the Block Graph!");
}

void SIScheduleBlock::addSucc(SIScheduleBlock *Succ,
                              SIScheduleBlockLinkKind Kind) {
  unsigned SuccID = Succ->getID();

  // Check if not already predecessor.
  for (std::pair<SIScheduleBlock*, SIScheduleBlockLinkKind> &S : Succs) {
    if (SuccID == S.first->getID()) {
      if (S.second == SIScheduleBlockLinkKind::NoData &&
          Kind == SIScheduleBlockLinkKind::Data)
        S.second = Kind;
      return;
    }
  }
  if (Succ->isHighLatencyBlock())
    ++NumHighLatencySuccessors;
  Succs.push_back(std::make_pair(Succ, Kind));

  assert(none_of(Preds,
                 [=](SIScheduleBlock *P) { return SuccID == P->getID(); }) &&
         "Loop in the Block Graph!");
}

void SIScheduleBlock::addLiveIns(SmallVector<RegisterMaskPair, 8> Ins)
{
  const MachineRegisterInfo *MRI = DAG->getMRI();
  unsigned VGPRSetID = DAG->getVGPRSetID();
  unsigned SGPRSetID = DAG->getSGPRSetID();

  auto addLiveIn = [&](RegisterMaskPair &RegPair) {
    unsigned Reg = RegPair.RegUnit;
    assert(RegPair.LaneMask.any());
    auto Pos = std::find_if(LiveInRegs.begin(), LiveInRegs.end(),
                            [=](RegisterMaskPair &RegPair2) {
                              return RegPair2.RegUnit == Reg;
                            });
    if (Pos == LiveInRegs.end()) {
      LiveInRegs.push_back(RegPair);
      PSetIterator PSetI = MRI->getPressureSets(Reg);
      for (; PSetI.isValid(); ++PSetI) {
        if (*PSetI == VGPRSetID)
          LiveInVGPRPressure += PSetI.getWeight();
        if (*PSetI == SGPRSetID)
          LiveInSGPRPressure += PSetI.getWeight();
      }
    }
    else
      Pos->LaneMask |= RegPair.LaneMask;
  };

  std::for_each(Ins.begin(), Ins.end(), addLiveIn);
}

void SIScheduleBlock::addLiveOuts(SmallVector<RegisterMaskPair, 8> Outs)
{
  const MachineRegisterInfo *MRI = DAG->getMRI();
  unsigned VGPRSetID = DAG->getVGPRSetID();
  unsigned SGPRSetID = DAG->getSGPRSetID();

  auto addLiveOut = [&](RegisterMaskPair &RegPair) {
    unsigned Reg = RegPair.RegUnit;
    auto Pos = std::find_if(LiveOutRegs.begin(), LiveOutRegs.end(),
                            [=](RegisterMaskPair &RegPair2) {
                              return RegPair2.RegUnit == Reg;
                            });
    if (Pos == LiveOutRegs.end()) {
      LiveOutRegs.push_back(RegPair);
      PSetIterator PSetI = MRI->getPressureSets(Reg);
      for (; PSetI.isValid(); ++PSetI) {
        if (*PSetI == VGPRSetID)
          LiveOutVGPRPressure += PSetI.getWeight();
        if (*PSetI == SGPRSetID)
          LiveOutSGPRPressure += PSetI.getWeight();
      }
    }
    else
      Pos->LaneMask |= RegPair.LaneMask;
  };

  std::for_each(Outs.begin(), Outs.end(), addLiveOut);
}

#ifndef NDEBUG
void SIScheduleBlock::printDebug(bool full) {
  dbgs() << "Block (" << ID << ")\n";
  if (!full)
    return;

  dbgs() << "\nContains High Latency Instruction: "
         << HighLatencyBlock << '\n';
  dbgs() << "\nDepends On:\n";
  for (SIScheduleBlock* P : Preds) {
    P->printDebug(false);
  }

  dbgs() << "\nSuccessors:\n";
  for (std::pair<SIScheduleBlock*, SIScheduleBlockLinkKind> S : Succs) {
    if (S.second == SIScheduleBlockLinkKind::Data)
      dbgs() << "(Data Dep) ";
    S.first->printDebug(false);
  }

  dbgs() << "LiveInPressure " << LiveInVGPRPressure << ' '
         << LiveInSGPRPressure << '\n';
  dbgs() << "LiveOutPressure " << LiveOutVGPRPressure << ' '
         << LiveOutSGPRPressure << "\n\n";
  dbgs() << "LiveIns:\n";
  for (const auto &Ins : LiveInRegs) {
    dbgs() << PrintVRegOrUnit(Ins.RegUnit, DAG->getTRI());
    if (!Ins.LaneMask.all())
      dbgs() << ':' << PrintLaneMask(Ins.LaneMask);
    dbgs() << ' ';
  }

  dbgs() << "\nLiveOuts:\n";
  for (const auto &Outs : LiveOutRegs) {
    dbgs() << PrintVRegOrUnit(Outs.RegUnit, DAG->getTRI());
    if (!Outs.LaneMask.all())
      dbgs() << ':' << PrintLaneMask(Outs.LaneMask);
    dbgs() << ' ';
  }

  dbgs() << "\nInstructions:\n";
  for (SUnit* SU : SUnits) {
    SU->dump(DAG);
  }

  dbgs() << "///////////////////////\n";
}
#endif

// SIScheduleBlockCreator //

SIScheduleBlockCreator::SIScheduleBlockCreator(SIScheduleDAGMI *DAG) :
DAG(DAG) {
}

SIScheduleBlockCreator::~SIScheduleBlockCreator() = default;

SIScheduleBlocks
SIScheduleBlockCreator::getBlocks(SISchedulerBlockCreatorVariant BlockVariant) {
  std::map<SISchedulerBlockCreatorVariant, SIScheduleBlocks>::iterator B =
    Blocks.find(BlockVariant);
  if (B == Blocks.end()) {
    SIScheduleBlocks Res;
    createBlocksForVariant(BlockVariant);
    topologicalSort();
    fillStats();
    Res.Blocks = CurrentBlocks;
    Res.TopDownIndex2Block = TopDownIndex2Block;
    Res.TopDownBlock2Index = TopDownBlock2Index;
    Blocks[BlockVariant] = Res;
    return Res;
  } else {
    return B->second;
  }
}

bool SIScheduleBlockCreator::isSUInBlock(SUnit *SU, unsigned ID) {
  if (SU->NodeNum >= DAG->SUnits.size())
    return false;
  return CurrentBlocks[Node2CurrentBlock[SU->NodeNum]]->getID() == ID;
}

void SIScheduleBlockCreator::colorHighLatenciesAlone() {
  unsigned DAGSize = DAG->SUnits.size();

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    if (DAG->IsHighLatencySU[SU->NodeNum]) {
      CurrentColoring[SU->NodeNum] = NextReservedID++;
    }
  }
}

static bool
hasDataDependencyPred(const SUnit &SU, const SUnit &FromSU) {
  for (const auto &PredDep : SU.Preds) {
    if (PredDep.getSUnit() == &FromSU &&
        PredDep.getKind() == llvm::SDep::Data)
      return true;
  }
  return false;
}

void SIScheduleBlockCreator::colorHighLatenciesGroups() {
  unsigned DAGSize = DAG->SUnits.size();
  unsigned NumHighLatencies = 0;
  unsigned GroupSize;
  int Color = NextReservedID;
  unsigned Count = 0;
  std::set<unsigned> FormingGroup;

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    if (DAG->IsHighLatencySU[SU->NodeNum])
      ++NumHighLatencies;
  }

  if (NumHighLatencies == 0)
    return;

  if (NumHighLatencies <= 6)
    GroupSize = 2;
  else if (NumHighLatencies <= 12)
    GroupSize = 3;
  else
    GroupSize = 4;

  for (unsigned SUNum : DAG->TopDownIndex2SU) {
    const SUnit &SU = DAG->SUnits[SUNum];
    if (DAG->IsHighLatencySU[SU.NodeNum]) {
      unsigned CompatibleGroup = true;
      int ProposedColor = Color;
      std::vector<int> AdditionalElements;

      // We don't want to put in the same block
      // two high latency instructions that depend
      // on each other.
      // One way would be to check canAddEdge
      // in both directions, but that currently is not
      // enough because there the high latency order is
      // enforced (via links).
      // Instead, look at the dependencies between the
      // high latency instructions and deduce if it is
      // a data dependency or not.
      for (unsigned j : FormingGroup) {
        bool HasSubGraph;
        std::vector<int> SubGraph;
        // By construction (topological order), if SU and
        // DAG->SUnits[j] are linked, DAG->SUnits[j] is neccessary
        // in the parent graph of SU.
#ifndef NDEBUG
        SubGraph = DAG->getTopo()->GetSubGraph(SU, DAG->SUnits[j],
                                               HasSubGraph);
        assert(!HasSubGraph);
#endif
        SubGraph = DAG->getTopo()->GetSubGraph(DAG->SUnits[j], SU,
                                               HasSubGraph);
        if (!HasSubGraph)
          continue; // No dependencies between each other
        else if (SubGraph.size() > 5) {
          // Too many elements would be required to be added to the block.
          CompatibleGroup = false;
          break;
        }
        else {
          // Check the type of dependency
          for (unsigned k : SubGraph) {
            // If in the path to join the two instructions,
            // there is another high latency instruction,
            // or instructions colored for another block
            // abort the merge.
            if (DAG->IsHighLatencySU[k] ||
                (CurrentColoring[k] != ProposedColor &&
                 CurrentColoring[k] != 0)) {
              CompatibleGroup = false;
              break;
            }
            // If one of the SU in the subgraph depends on the result of SU j,
            // there'll be a data dependency.
            if (hasDataDependencyPred(DAG->SUnits[k], DAG->SUnits[j])) {
              CompatibleGroup = false;
              break;
            }
          }
          if (!CompatibleGroup)
            break;
          // Same check for the SU
          if (hasDataDependencyPred(SU, DAG->SUnits[j])) {
            CompatibleGroup = false;
            break;
          }
          // Add all the required instructions to the block
          // These cannot live in another block (because they
          // depend (order dependency) on one of the
          // instruction in the block, and are required for the
          // high latency instruction we add.
          AdditionalElements.insert(AdditionalElements.end(),
                                    SubGraph.begin(), SubGraph.end());
        }
      }
      if (CompatibleGroup) {
        FormingGroup.insert(SU.NodeNum);
        for (unsigned j : AdditionalElements)
          CurrentColoring[j] = ProposedColor;
        CurrentColoring[SU.NodeNum] = ProposedColor;
        ++Count;
      }
      // Found one incompatible instruction,
      // or has filled a big enough group.
      // -> start a new one.
      if (!CompatibleGroup) {
        FormingGroup.clear();
        Color = ++NextReservedID;
        ProposedColor = Color;
        FormingGroup.insert(SU.NodeNum);
        CurrentColoring[SU.NodeNum] = ProposedColor;
        Count = 0;
      } else if (Count == GroupSize) {
        FormingGroup.clear();
        Color = ++NextReservedID;
        ProposedColor = Color;
        Count = 0;
      }
    }
  }
}

void SIScheduleBlockCreator::colorComputeReservedDependencies() {
  unsigned DAGSize = DAG->SUnits.size();
  std::map<std::set<unsigned>, unsigned> ColorCombinations;

  CurrentTopDownReservedDependencyColoring.clear();
  CurrentBottomUpReservedDependencyColoring.clear();

  CurrentTopDownReservedDependencyColoring.resize(DAGSize, 0);
  CurrentBottomUpReservedDependencyColoring.resize(DAGSize, 0);

  // Traverse TopDown, and give different colors to SUs depending
  // on which combination of High Latencies they depend on.

  for (unsigned SUNum : DAG->TopDownIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    std::set<unsigned> SUColors;

    // Already given.
    if (CurrentColoring[SU->NodeNum]) {
      CurrentTopDownReservedDependencyColoring[SU->NodeNum] =
        CurrentColoring[SU->NodeNum];
      continue;
    }

   for (SDep& PredDep : SU->Preds) {
      SUnit *Pred = PredDep.getSUnit();
      if (PredDep.isWeak() || Pred->NodeNum >= DAGSize)
        continue;
      if (CurrentTopDownReservedDependencyColoring[Pred->NodeNum] > 0)
        SUColors.insert(CurrentTopDownReservedDependencyColoring[Pred->NodeNum]);
    }
    // Color 0 by default.
    if (SUColors.empty())
      continue;
    // Same color than parents.
    if (SUColors.size() == 1 && *SUColors.begin() > DAGSize)
      CurrentTopDownReservedDependencyColoring[SU->NodeNum] =
        *SUColors.begin();
    else {
      std::map<std::set<unsigned>, unsigned>::iterator Pos =
        ColorCombinations.find(SUColors);
      if (Pos != ColorCombinations.end()) {
          CurrentTopDownReservedDependencyColoring[SU->NodeNum] = Pos->second;
      } else {
        CurrentTopDownReservedDependencyColoring[SU->NodeNum] =
          NextNonReservedID;
        ColorCombinations[SUColors] = NextNonReservedID++;
      }
    }
  }

  ColorCombinations.clear();

  // Same as before, but BottomUp.

  for (unsigned SUNum : DAG->BottomUpIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    std::set<unsigned> SUColors;

    // Already given.
    if (CurrentColoring[SU->NodeNum]) {
      CurrentBottomUpReservedDependencyColoring[SU->NodeNum] =
        CurrentColoring[SU->NodeNum];
      continue;
    }

    for (SDep& SuccDep : SU->Succs) {
      SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (CurrentBottomUpReservedDependencyColoring[Succ->NodeNum] > 0)
        SUColors.insert(CurrentBottomUpReservedDependencyColoring[Succ->NodeNum]);
    }
    // Keep color 0.
    if (SUColors.empty())
      continue;
    // Same color than parents.
    if (SUColors.size() == 1 && *SUColors.begin() > DAGSize)
      CurrentBottomUpReservedDependencyColoring[SU->NodeNum] =
        *SUColors.begin();
    else {
      std::map<std::set<unsigned>, unsigned>::iterator Pos =
        ColorCombinations.find(SUColors);
      if (Pos != ColorCombinations.end()) {
        CurrentBottomUpReservedDependencyColoring[SU->NodeNum] = Pos->second;
      } else {
        CurrentBottomUpReservedDependencyColoring[SU->NodeNum] =
          NextNonReservedID;
        ColorCombinations[SUColors] = NextNonReservedID++;
      }
    }
  }
}

void SIScheduleBlockCreator::colorAccordingToReservedDependencies() {
  unsigned DAGSize = DAG->SUnits.size();
  std::map<std::pair<unsigned, unsigned>, unsigned> ColorCombinations;

  // Every combination of colors given by the top down
  // and bottom up Reserved node dependency

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    std::pair<unsigned, unsigned> SUColors;

    // High latency instructions: already given.
    if (CurrentColoring[SU->NodeNum])
      continue;

    SUColors.first = CurrentTopDownReservedDependencyColoring[SU->NodeNum];
    SUColors.second = CurrentBottomUpReservedDependencyColoring[SU->NodeNum];

    std::map<std::pair<unsigned, unsigned>, unsigned>::iterator Pos =
      ColorCombinations.find(SUColors);
    if (Pos != ColorCombinations.end()) {
      CurrentColoring[SU->NodeNum] = Pos->second;
    } else {
      CurrentColoring[SU->NodeNum] = NextNonReservedID;
      ColorCombinations[SUColors] = NextNonReservedID++;
    }
  }
}

void SIScheduleBlockCreator::colorEndsAccordingToDependencies() {
  unsigned DAGSize = DAG->SUnits.size();
  std::vector<int> PendingColoring = CurrentColoring;

  assert(DAGSize >= 1 &&
         CurrentBottomUpReservedDependencyColoring.size() == DAGSize &&
         CurrentTopDownReservedDependencyColoring.size() == DAGSize);
  // If there is no reserved block at all, do nothing. We don't want
  // everything in one block.
  if (*std::max_element(CurrentBottomUpReservedDependencyColoring.begin(),
                        CurrentBottomUpReservedDependencyColoring.end()) == 0 &&
      *std::max_element(CurrentTopDownReservedDependencyColoring.begin(),
                        CurrentTopDownReservedDependencyColoring.end()) == 0)
    return;

  for (unsigned SUNum : DAG->BottomUpIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    std::set<unsigned> SUColors;
    std::set<unsigned> SUColorsPending;

    if (CurrentColoring[SU->NodeNum] <= (int)DAGSize)
      continue;

    if (CurrentBottomUpReservedDependencyColoring[SU->NodeNum] > 0 ||
        CurrentTopDownReservedDependencyColoring[SU->NodeNum] > 0)
      continue;

    for (SDep& SuccDep : SU->Succs) {
      SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (CurrentBottomUpReservedDependencyColoring[Succ->NodeNum] > 0 ||
          CurrentTopDownReservedDependencyColoring[Succ->NodeNum] > 0)
        SUColors.insert(CurrentColoring[Succ->NodeNum]);
      SUColorsPending.insert(PendingColoring[Succ->NodeNum]);
    }
    // If there is only one child/parent block, and that block
    // is not among the ones we are removing in this path, then
    // merge the instruction to that block
    if (SUColors.size() == 1 && SUColorsPending.size() == 1)
      PendingColoring[SU->NodeNum] = *SUColors.begin();
    else // TODO: Attribute new colors depending on color
         // combination of children.
      PendingColoring[SU->NodeNum] = NextNonReservedID++;
  }
  CurrentColoring = PendingColoring;
}


void SIScheduleBlockCreator::colorForceConsecutiveOrderInGroup() {
  unsigned DAGSize = DAG->SUnits.size();
  unsigned PreviousColor;
  std::set<unsigned> SeenColors;

  if (DAGSize <= 1)
    return;

  PreviousColor = CurrentColoring[0];

  for (unsigned i = 1, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    unsigned CurrentColor = CurrentColoring[i];
    unsigned PreviousColorSave = PreviousColor;
    assert(i == SU->NodeNum);

    if (CurrentColor != PreviousColor)
      SeenColors.insert(PreviousColor);
    PreviousColor = CurrentColor;

    if (CurrentColoring[SU->NodeNum] <= (int)DAGSize)
      continue;

    if (SeenColors.find(CurrentColor) == SeenColors.end())
      continue;

    if (PreviousColorSave != CurrentColor)
      CurrentColoring[i] = NextNonReservedID++;
    else
      CurrentColoring[i] = CurrentColoring[i-1];
  }
}

void SIScheduleBlockCreator::colorMergeConstantLoadsNextGroup() {
  unsigned DAGSize = DAG->SUnits.size();

  for (unsigned SUNum : DAG->BottomUpIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    std::set<unsigned> SUColors;

    if (CurrentColoring[SU->NodeNum] <= (int)DAGSize)
      continue;

    // No predecessor: Vgpr constant loading.
    // Low latency instructions usually have a predecessor (the address)
    if (SU->Preds.size() > 0 && !DAG->IsLowLatencySU[SU->NodeNum])
      continue;

    for (SDep& SuccDep : SU->Succs) {
      SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.insert(CurrentColoring[Succ->NodeNum]);
    }
    if (SUColors.size() == 1)
      CurrentColoring[SU->NodeNum] = *SUColors.begin();
  }
}

void SIScheduleBlockCreator::colorMergeIfPossibleNextGroup() {
  unsigned DAGSize = DAG->SUnits.size();

  for (unsigned SUNum : DAG->BottomUpIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    std::set<unsigned> SUColors;

    if (CurrentColoring[SU->NodeNum] <= (int)DAGSize)
      continue;

    for (SDep& SuccDep : SU->Succs) {
       SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.insert(CurrentColoring[Succ->NodeNum]);
    }
    if (SUColors.size() == 1)
      CurrentColoring[SU->NodeNum] = *SUColors.begin();
  }
}

void SIScheduleBlockCreator::colorMergeIfPossibleNextGroupOnlyForReserved() {
  unsigned DAGSize = DAG->SUnits.size();

  for (unsigned SUNum : DAG->BottomUpIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    std::set<unsigned> SUColors;

    if (CurrentColoring[SU->NodeNum] <= (int)DAGSize)
      continue;

    for (SDep& SuccDep : SU->Succs) {
       SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.insert(CurrentColoring[Succ->NodeNum]);
    }
    if (SUColors.size() == 1 && *SUColors.begin() <= DAGSize)
      CurrentColoring[SU->NodeNum] = *SUColors.begin();
  }
}

void SIScheduleBlockCreator::colorMergeIfPossibleSmallGroupsToNextGroup() {
  unsigned DAGSize = DAG->SUnits.size();
  std::map<unsigned, unsigned> ColorCount;

  for (unsigned SUNum : DAG->BottomUpIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    unsigned color = CurrentColoring[SU->NodeNum];
     ++ColorCount[color];
  }

  for (unsigned SUNum : DAG->BottomUpIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    unsigned color = CurrentColoring[SU->NodeNum];
    std::set<unsigned> SUColors;

    if (CurrentColoring[SU->NodeNum] <= (int)DAGSize)
      continue;

    if (ColorCount[color] > 1)
      continue;

    for (SDep& SuccDep : SU->Succs) {
       SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.insert(CurrentColoring[Succ->NodeNum]);
    }
    if (SUColors.size() == 1 && *SUColors.begin() != color) {
      --ColorCount[color];
      CurrentColoring[SU->NodeNum] = *SUColors.begin();
      ++ColorCount[*SUColors.begin()];
    }
  }
}

void SIScheduleBlockCreator::cutHugeBlocks() {
  // TODO
}

void SIScheduleBlockCreator::regroupNoUserInstructions() {
  unsigned DAGSize = DAG->SUnits.size();
  int GroupID = NextNonReservedID++;

  for (unsigned SUNum : DAG->BottomUpIndex2SU) {
    SUnit *SU = &DAG->SUnits[SUNum];
    bool hasSuccessor = false;

    if (CurrentColoring[SU->NodeNum] <= (int)DAGSize)
      continue;

    for (SDep& SuccDep : SU->Succs) {
       SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      hasSuccessor = true;
    }
    if (!hasSuccessor)
      CurrentColoring[SU->NodeNum] = GroupID;
  }
}

void SIScheduleBlockCreator::createBlocksForVariant(SISchedulerBlockCreatorVariant BlockVariant) {
  unsigned DAGSize = DAG->SUnits.size();
  std::map<unsigned,unsigned> RealID;

  CurrentBlocks.clear();
  CurrentColoring.clear();
  CurrentColoring.resize(DAGSize, 0);
  Node2CurrentBlock.clear();

  NextReservedID = 1;
  NextNonReservedID = DAGSize + 1;

  DEBUG(dbgs() << "Coloring the graph\n");

  if (BlockVariant == SISchedulerBlockCreatorVariant::LatenciesGrouped)
    colorHighLatenciesGroups();
  else
    colorHighLatenciesAlone();
  colorComputeReservedDependencies();
  colorAccordingToReservedDependencies();
  colorEndsAccordingToDependencies();
  if (BlockVariant == SISchedulerBlockCreatorVariant::LatenciesAlonePlusConsecutive)
    colorForceConsecutiveOrderInGroup();
  regroupNoUserInstructions();
  colorMergeConstantLoadsNextGroup();
  colorMergeIfPossibleNextGroupOnlyForReserved();

  // Put SUs of same color into same block
  Node2CurrentBlock.resize(DAGSize, -1);
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    unsigned Color = CurrentColoring[SU->NodeNum];
    if (RealID.find(Color) == RealID.end()) {
      int ID = CurrentBlocks.size();
      BlockPtrs.push_back(llvm::make_unique<SIScheduleBlock>(DAG, this, ID));
      CurrentBlocks.push_back(BlockPtrs.rbegin()->get());
      RealID[Color] = ID;
    }
    CurrentBlocks[RealID[Color]]->addUnit(SU);
    Node2CurrentBlock[SU->NodeNum] = RealID[Color];
  }

  // Build dependencies between blocks.
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    int SUID = Node2CurrentBlock[i];
     for (SDep& SuccDep : SU->Succs) {
       SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (Node2CurrentBlock[Succ->NodeNum] != SUID)
        CurrentBlocks[SUID]->addSucc(CurrentBlocks[Node2CurrentBlock[Succ->NodeNum]],
                                     SuccDep.isCtrl() ? NoData : Data);
    }
    for (SDep& PredDep : SU->Preds) {
      SUnit *Pred = PredDep.getSUnit();
      if (PredDep.isWeak() || Pred->NodeNum >= DAGSize)
        continue;
      if (Node2CurrentBlock[Pred->NodeNum] != SUID)
        CurrentBlocks[SUID]->addPred(CurrentBlocks[Node2CurrentBlock[Pred->NodeNum]]);
    }
  }

  // Compute Block LiveIns
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    int SUID = Node2CurrentBlock[i];
    SmallVector<RegisterMaskPair, 8> Uses;

    getUsesForSU(Uses, *SU, DAG->getTRI(),
                 DAG->shouldTrackLaneMasks());

    // Remove From Uses everything that is defined by SUs of the same Block.
    for (SDep& PredDep : SU->Preds) {
      SUnit *Pred = PredDep.getSUnit();
      if (Pred->isBoundaryNode() || Node2CurrentBlock[Pred->NodeNum] != SUID)
        continue;
      if (!PredDep.isAssignedRegDep())
        continue;
      if (!TargetRegisterInfo::isVirtualRegister(PredDep.getReg()))
        continue;
      removeUseFromDef(Uses, PredDep.getReg(), Pred);
    }
    // The remaining Uses are Block LiveIns.
    CurrentBlocks[SUID]->addLiveIns(Uses);
  }

  // Compute Block LiveOut
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    int SUID = Node2CurrentBlock[i];
    SmallVector<RegisterMaskPair, 8> LiveOuts;

    for (SDep& SuccDep : SU->Succs) {
      SUnit *Succ = SuccDep.getSUnit();
      if (!Succ->isBoundaryNode() && Node2CurrentBlock[Succ->NodeNum] == SUID)
        continue;
      if (!SuccDep.isAssignedRegDep())
        continue;
      // We don't track physical registers
      if (!TargetRegisterInfo::isVirtualRegister(SuccDep.getReg()))
        continue;
      addDefFromUse(LiveOuts, SuccDep.getReg(), SU, Succ);
    }
    CurrentBlocks[SUID]->addLiveOuts(LiveOuts);
  }

  for (unsigned i = 0, e = CurrentBlocks.size(); i != e; ++i) {
    SIScheduleBlock *Block = CurrentBlocks[i];
    Block->finalize();
  }
  DEBUG(
    dbgs() << "Blocks created:\n\n";
    for (unsigned i = 0, e = CurrentBlocks.size(); i != e; ++i) {
      SIScheduleBlock *Block = CurrentBlocks[i];
      Block->printDebug(true);
    }
  );
}

// Adapted from Codegen/MachineScheduler.cpp

void SIScheduleBlockCreator::topologicalSort() {
  unsigned DAGSize = CurrentBlocks.size();
  std::vector<int> WorkList;

  DEBUG(dbgs() << "Topological Sort\n");

  WorkList.reserve(DAGSize);
  TopDownIndex2Block.resize(DAGSize);
  TopDownBlock2Index.resize(DAGSize);
  BottomUpIndex2Block.resize(DAGSize);

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIScheduleBlock *Block = CurrentBlocks[i];
    unsigned Degree = Block->getSuccs().size();
    TopDownBlock2Index[i] = Degree;
    if (Degree == 0) {
      WorkList.push_back(i);
    }
  }

  int Id = DAGSize;
  while (!WorkList.empty()) {
    int i = WorkList.back();
    SIScheduleBlock *Block = CurrentBlocks[i];
    WorkList.pop_back();
    TopDownBlock2Index[i] = --Id;
    TopDownIndex2Block[Id] = i;
    for (SIScheduleBlock* Pred : Block->getPreds()) {
      if (!--TopDownBlock2Index[Pred->getID()])
        WorkList.push_back(Pred->getID());
    }
  }

#ifndef NDEBUG
  // Check correctness of the ordering.
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIScheduleBlock *Block = CurrentBlocks[i];
    for (SIScheduleBlock* Pred : Block->getPreds()) {
      assert(TopDownBlock2Index[i] > TopDownBlock2Index[Pred->getID()] &&
      "Wrong Top Down topological sorting");
    }
  }
#endif

  BottomUpIndex2Block = std::vector<int>(TopDownIndex2Block.rbegin(),
                                         TopDownIndex2Block.rend());
}

void SIScheduleBlockCreator::fillStats() {
  unsigned DAGSize = CurrentBlocks.size();

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    int BlockIndice = TopDownIndex2Block[i];
    SIScheduleBlock *Block = CurrentBlocks[BlockIndice];
    if (Block->getPreds().empty())
      Block->Depth = 0;
    else {
      unsigned Depth = 0;
      for (SIScheduleBlock *Pred : Block->getPreds()) {
        if (Depth < Pred->Depth + 1)
          Depth = Pred->Depth + 1;
      }
      Block->Depth = Depth;
    }
  }

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    int BlockIndice = BottomUpIndex2Block[i];
    SIScheduleBlock *Block = CurrentBlocks[BlockIndice];
    if (Block->getSuccs().empty())
      Block->Height = 0;
    else {
      unsigned Height = 0;
      for (const auto &Succ : Block->getSuccs())
        Height = std::min(Height, Succ.first->Height + 1);
      Block->Height = Height;
    }
  }
}

LaneBitmask SIScheduleBlockCreator::getLaneBitmaskForDef(const SUnit *SU,
                                                         unsigned Reg)
{
  LaneBitmask DefMask = LaneBitmask::getNone();

  for (const MachineOperand &MO : SU->getInstr()->operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() == Reg && !MO.isDead()) {
      unsigned SubRegIdx = MO.getSubReg();
      DefMask |= SubRegIdx != 0 ?
        DAG->getTRI()->getSubRegIndexLaneMask(SubRegIdx) :
        LaneBitmask::getAll();
    }
  }

  return DefMask;
}

LaneBitmask SIScheduleBlockCreator::getLaneBitmaskForUse(const SUnit *SU,
                                                         unsigned Reg)
{
  LaneBitmask UseMask = LaneBitmask::getNone();

  for (const MachineOperand &MO : SU->getInstr()->operands()) {
    if (MO.isReg() && MO.isUse() && MO.getReg() == Reg &&
        !MO.isUndef() && !MO.isInternalRead()) {
      unsigned SubRegIdx = MO.getSubReg();
      UseMask |= SubRegIdx != 0 ?
        DAG->getTRI()->getSubRegIndexLaneMask(SubRegIdx) :
        LaneBitmask::getAll();
    }
  }

  return UseMask;
}

static void replaceAllByMask(LaneBitmask &L, LaneBitmask &Mask)
{
  if (L.all())
    L = Mask;
}

void
SIScheduleBlockCreator::removeUseFromDef(SmallVectorImpl<RegisterMaskPair> &Uses,
                                         unsigned Reg, const SUnit *SU)
{
  assert(TargetRegisterInfo::isVirtualRegister(Reg));
  LaneBitmask LanesMaskMax, DefMask, UseMask;

  auto UsePos = std::find_if(Uses.begin(), Uses.end(),
                             [=](RegisterMaskPair P) {
                               return P.RegUnit == Reg;
                             });

  // Already removed
  if (UsePos == Uses.end())
    return;

  if (!DAG->shouldTrackLaneMasks()) {
    Uses.erase(UsePos);
    return;
  }

  LanesMaskMax = DAG->getMRI()->getMaxLaneMaskForVReg(Reg);
  DefMask = getLaneBitmaskForDef(SU, Reg);
  UseMask = UsePos->LaneMask;

  replaceAllByMask(DefMask, LanesMaskMax);
  replaceAllByMask(UseMask, LanesMaskMax);

  if ((UseMask & ~DefMask).none())
    Uses.erase(UsePos);
  else
    UsePos->LaneMask = UseMask & ~DefMask;
}

void
SIScheduleBlockCreator::addDefFromUse(SmallVectorImpl<RegisterMaskPair> &Defs,
                                      unsigned Reg, const SUnit *SUDef,
                                      const SUnit *SUUse)
{
  LaneBitmask LanesMaskMax, DefMask, UseMask;
  int DefIndex = DAG->getTopo()->getSUTopoIndex(*SUDef);

  auto DefPos = std::find_if(Defs.begin(), Defs.end(),
                             [=](RegisterMaskPair P) {
                               return P.RegUnit == Reg;
                             });

  if (!DAG->shouldTrackLaneMasks()) {
    if (DefPos == Defs.end())
      Defs.push_back(RegisterMaskPair(Reg, LaneBitmask::getAll()));
    return;
  }

  LanesMaskMax = DAG->getMRI()->getMaxLaneMaskForVReg(Reg);
  DefMask = getLaneBitmaskForDef(SUDef, Reg);
  UseMask = getLaneBitmaskForUse(SUUse, Reg);

  replaceAllByMask(DefMask, LanesMaskMax);
  replaceAllByMask(UseMask, LanesMaskMax);

  for (const SDep& PredDep : SUUse->Preds) {
    const SUnit *Pred = PredDep.getSUnit();
    if (!PredDep.isAssignedRegDep())
      continue;
    if (PredDep.getReg() != Reg)
      continue;
    // Look for SUs that are after SUDef in the
    // topological sort.
    if (DAG->getTopo()->getSUTopoIndex(*Pred) <= DefIndex)
      continue;
    // These lanes content are defined by Pred,
    // and thus not from SUDef.
    UseMask &= ~getLaneBitmaskForDef(Pred, Reg);
  }

  DefMask &= UseMask;
  // There is data dependency, thus DefMask cannot be none.
  assert(!DefMask.none());

  if (DefPos == Defs.end())
    Defs.push_back(RegisterMaskPair(Reg, DefMask));
  else
    DefPos->LaneMask |= DefMask;
}

// SIScheduleBlockScheduler //

SIScheduleBlockScheduler::SIScheduleBlockScheduler(SIScheduleDAGMI *DAG,
                                                   SISchedulerBlockSchedulerVariant Variant,
                                                   SIScheduleBlocks  BlocksStruct) :
  DAG(DAG), Variant(Variant), Blocks(BlocksStruct.Blocks),
  LastPosWaitedHighLatency(0), NumBlockScheduled(0),
  maxVregUsage(0), maxSregUsage(0) {

  LastPosHighLatencyParentScheduled.resize(Blocks.size(), 0);

#ifndef NDEBUG
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIScheduleBlock *Block = Blocks[i];
    assert(Block->getID() == i);
  }
#endif

  std::vector<SmallVector<unsigned, 8>> BlockSuccs;
  std::vector<SmallVector<unsigned, 8>> BlockPreds;
  std::vector<SmallVector<RegisterMaskPair, 8>> InRegsForBlock;
  std::vector<SmallVector<RegisterMaskPair, 8>> OutRegsForBlock;

  BlockSuccs.resize(Blocks.size());
  BlockPreds.resize(Blocks.size());
  InRegsForBlock.resize(Blocks.size());
  OutRegsForBlock.resize(Blocks.size());

  for (unsigned i = 0; i < Blocks.size(); i++) {
    SIScheduleBlock *Block = Blocks[i];
    for (const auto &Succ : Block->getSuccs())
      BlockSuccs[i].push_back(Succ.first->getID());
    for (const auto &Pred : Block->getPreds())
      BlockPreds[i].push_back(Pred->getID());
    InRegsForBlock[i] = Block->getInRegs();
    OutRegsForBlock[i] = Block->getOutRegs();
  }

  RPTracker.reset(new SISchedulerRPTracker(
    DAG->getInRegs(),
    DAG->getOutRegs(),
    BlockSuccs,
    BlockPreds,
    InRegsForBlock,
    OutRegsForBlock,
    DAG->getMRI(),
    DAG->getTRI(),
    DAG->getVGPRSetID(),
    DAG->getSGPRSetID()
  ));

  while (SIScheduleBlock *Block = pickBlock()) {
    BlocksScheduled.push_back(Block);
    blockScheduled(Block);
  }

  DEBUG(
    dbgs() << "Block Order:";
    for (SIScheduleBlock* Block : BlocksScheduled) {
      dbgs() << ' ' << Block->getID();
    }
    dbgs() << '\n';
  );
}

bool SIScheduleBlockScheduler::tryCandidateLatency(SIBlockSchedCandidate &Cand,
                                                   SIBlockSchedCandidate &TryCand) {
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Try to hide high latencies.
  if (tryLess(TryCand.LastPosHighLatParentScheduled,
              Cand.LastPosHighLatParentScheduled, TryCand, Cand, Latency))
    return true;
  // Schedule high latencies early so you can hide them better.
  if (tryGreater(TryCand.IsHighLatency, Cand.IsHighLatency,
                 TryCand, Cand, Latency))
    return true;
  if (TryCand.IsHighLatency && tryGreater(TryCand.Height, Cand.Height,
                                          TryCand, Cand, Depth))
    return true;
  if (tryGreater(TryCand.NumHighLatencySuccessors,
                 Cand.NumHighLatencySuccessors,
                 TryCand, Cand, Successor))
    return true;
  return false;
}

bool SIScheduleBlockScheduler::tryCandidateRegUsage(SIBlockSchedCandidate &Cand,
                                                    SIBlockSchedCandidate &TryCand) {
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  if (tryLess(TryCand.VGPRUsageDiff > 0, Cand.VGPRUsageDiff > 0,
              TryCand, Cand, RegUsage))
    return true;
  if (tryGreater(TryCand.NumSuccessors > 0,
                 Cand.NumSuccessors > 0,
                 TryCand, Cand, Successor))
    return true;
  if (tryGreater(TryCand.Height, Cand.Height, TryCand, Cand, Depth))
    return true;
  if (tryLess(TryCand.VGPRUsageDiff, Cand.VGPRUsageDiff,
              TryCand, Cand, RegUsage))
    return true;
  return false;
}

SIScheduleBlock *SIScheduleBlockScheduler::pickBlock() {
  SIBlockSchedCandidate Cand;
  unsigned VregCurrentUsage, SregCurrentUsage;
  SIScheduleBlock *Block;
  SmallVector<unsigned, 16> ReadyBlocks = RPTracker->getReadyItems();
  if (ReadyBlocks.empty())
    return nullptr;

  RPTracker->getCurrentRegUsage(VregCurrentUsage, SregCurrentUsage);
  if (VregCurrentUsage > maxVregUsage)
    maxVregUsage = VregCurrentUsage;
  if (SregCurrentUsage > maxSregUsage)
    maxSregUsage = SregCurrentUsage;
  DEBUG(
    dbgs() << "Picking New Blocks\n";
    dbgs() << "Available: ";
    for (unsigned ID : ReadyBlocks)
      dbgs() << ID << ' ';
    dbgs() << "\nCurrent Live:\n";
    RPTracker->printDebugLives();
    dbgs() << '\n';
    dbgs() << "Current VGPRs: " << VregCurrentUsage << '\n';
    dbgs() << "Current SGPRs: " << SregCurrentUsage << '\n';
  );

  Cand.Block = nullptr;
  for (unsigned ID : ReadyBlocks) {
    SIBlockSchedCandidate TryCand;
    int SGPRUsageDiff;

    TryCand.Block = Blocks[ID];
    TryCand.IsHighLatency = TryCand.Block->isHighLatencyBlock();
    RPTracker->checkRegUsageImpact(ID, TryCand.VGPRUsageDiff, SGPRUsageDiff);
    TryCand.NumSuccessors = TryCand.Block->getSuccs().size();
    TryCand.NumHighLatencySuccessors =
      TryCand.Block->getNumHighLatencySuccessors();
    TryCand.LastPosHighLatParentScheduled =
      (unsigned int) std::max<int> (0,
         LastPosHighLatencyParentScheduled[ID] -
           LastPosWaitedHighLatency);
    TryCand.Height = TryCand.Block->Height;
    // Try not to increase VGPR usage too much, else we may spill.
    if (VregCurrentUsage > 120 ||
        Variant != SISchedulerBlockSchedulerVariant::BlockLatencyRegUsage) {
      if (!tryCandidateRegUsage(Cand, TryCand) &&
          Variant != SISchedulerBlockSchedulerVariant::BlockRegUsage)
        tryCandidateLatency(Cand, TryCand);
    } else {
      if (!tryCandidateLatency(Cand, TryCand))
        tryCandidateRegUsage(Cand, TryCand);
    }
    if (TryCand.Reason != NoCand) {
      Cand.setBest(TryCand);
      DEBUG(dbgs() << "Best Current Choice: " << Cand.Block->getID() << ' '
                   << getReasonStr(Cand.Reason) << '\n');
    }
  }

  DEBUG(
    dbgs() << "Picking: " << Cand.Block->getID() << '\n';
    dbgs() << "Is a block with high latency instruction: "
      << (Cand.IsHighLatency ? "yes\n" : "no\n");
    dbgs() << "Position of last high latency dependency: "
           << Cand.LastPosHighLatParentScheduled << '\n';
    dbgs() << "VGPRUsageDiff: " << Cand.VGPRUsageDiff << '\n';
    dbgs() << '\n';
  );

  Block = Cand.Block;
  return Block;
}

void SIScheduleBlockScheduler::blockScheduled(SIScheduleBlock *Block) {
  unsigned ID = Block->getID();
  RPTracker->itemScheduled(ID);

  if (Block->isHighLatencyBlock()) {
    for (const auto &Succ : Block->getSuccs()) {
      if (Succ.second == SIScheduleBlockLinkKind::Data)
        LastPosHighLatencyParentScheduled[Succ.first->getID()] =
          NumBlockScheduled;
    }
  }

  if (LastPosHighLatencyParentScheduled[ID] >
        (unsigned)LastPosWaitedHighLatency)
    LastPosWaitedHighLatency =
      LastPosHighLatencyParentScheduled[ID];
  ++NumBlockScheduled;
}

// SIScheduler //

struct SIScheduleBlockResult
SIScheduler::scheduleVariant(SISchedulerBlockCreatorVariant BlockVariant,
                             SISchedulerBlockSchedulerVariant ScheduleVariant) {
  SIScheduleBlocks Blocks = BlockCreator.getBlocks(BlockVariant);
  SIScheduleBlockScheduler Scheduler(DAG, ScheduleVariant, Blocks);
  std::vector<SIScheduleBlock*> ScheduledBlocks;
  struct SIScheduleBlockResult Res;

  ScheduledBlocks = Scheduler.getBlocks();

  for (unsigned b = 0; b < ScheduledBlocks.size(); ++b) {
    SIScheduleBlock *Block = ScheduledBlocks[b];
    std::vector<SUnit*> SUs = Block->getScheduledUnits();

    for (SUnit* SU : SUs)
      Res.SUs.push_back(SU->NodeNum);
  }

  Res.MaxSGPRUsage = Scheduler.getSGPRUsage();
  Res.MaxVGPRUsage = Scheduler.getVGPRUsage();
  return Res;
}

// SIScheduleDAGMI //

SIScheduleDAGMI::SIScheduleDAGMI(MachineSchedContext *C) :
  ScheduleDAGMILive(C, llvm::make_unique<GenericScheduler>(C)) {
  SITII = static_cast<const SIInstrInfo*>(TII);
  SITRI = static_cast<const SIRegisterInfo*>(TRI);

  VGPRSetID = SITRI->getVGPRPressureSet();
  SGPRSetID = SITRI->getSGPRPressureSet();
}

SIScheduleDAGMI::~SIScheduleDAGMI() = default;

// Code adapted from scheduleDAG.cpp
// Does a topological sort over the SUs.
// Both TopDown and BottomUp
void SIScheduleDAGMI::topologicalSort() {
  Topo.InitDAGTopologicalSorting();

  TopDownIndex2SU = std::vector<int>(Topo.begin(), Topo.end());
  BottomUpIndex2SU = std::vector<int>(Topo.rbegin(), Topo.rend());
}

// Move low latencies further from their user without
// increasing SGPR usage (in general)
// This is to be replaced by a better pass that would
// take into account SGPR usage (based on VGPR Usage
// and the corresponding wavefront count), that would
// try to merge groups of loads if it make sense, etc
void SIScheduleDAGMI::moveLowLatencies() {
   unsigned DAGSize = SUnits.size();
   int LastLowLatencyUser = -1;
   int LastLowLatencyPos = -1;

   for (unsigned i = 0, e = ScheduledSUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[ScheduledSUnits[i]];
    bool IsLowLatencyUser = false;
    unsigned MinPos = 0;

    for (SDep& PredDep : SU->Preds) {
      SUnit *Pred = PredDep.getSUnit();
      if (SITII->isLowLatencyInstruction(*Pred->getInstr())) {
        IsLowLatencyUser = true;
      }
      if (Pred->NodeNum >= DAGSize)
        continue;
      unsigned PredPos = ScheduledSUnitsInv[Pred->NodeNum];
      if (PredPos >= MinPos)
        MinPos = PredPos + 1;
    }

    if (SITII->isLowLatencyInstruction(*SU->getInstr())) {
      unsigned BestPos = LastLowLatencyUser + 1;
      if ((int)BestPos <= LastLowLatencyPos)
        BestPos = LastLowLatencyPos + 1;
      if (BestPos < MinPos)
        BestPos = MinPos;
      if (BestPos < i) {
        for (unsigned u = i; u > BestPos; --u) {
          ++ScheduledSUnitsInv[ScheduledSUnits[u-1]];
          ScheduledSUnits[u] = ScheduledSUnits[u-1];
        }
        ScheduledSUnits[BestPos] = SU->NodeNum;
        ScheduledSUnitsInv[SU->NodeNum] = BestPos;
      }
      LastLowLatencyPos = BestPos;
      if (IsLowLatencyUser)
        LastLowLatencyUser = BestPos;
    } else if (IsLowLatencyUser) {
      LastLowLatencyUser = i;
    // Moves COPY instructions on which depends
    // the low latency instructions too.
    } else if (SU->getInstr()->getOpcode() == AMDGPU::COPY) {
      bool CopyForLowLat = false;
      for (SDep& SuccDep : SU->Succs) {
        SUnit *Succ = SuccDep.getSUnit();
        if (SITII->isLowLatencyInstruction(*Succ->getInstr())) {
          CopyForLowLat = true;
        }
      }
      if (!CopyForLowLat)
        continue;
      if (MinPos < i) {
        for (unsigned u = i; u > MinPos; --u) {
          ++ScheduledSUnitsInv[ScheduledSUnits[u-1]];
          ScheduledSUnits[u] = ScheduledSUnits[u-1];
        }
        ScheduledSUnits[MinPos] = SU->NodeNum;
        ScheduledSUnitsInv[SU->NodeNum] = MinPos;
      }
    }
  }
}

void SIScheduleDAGMI::schedule()
{
  SmallVector<SUnit*, 8> TopRoots, BotRoots;
  SIScheduleBlockResult Best, Temp;
  DEBUG(dbgs() << "Preparing Scheduling\n");

  buildDAGWithRegPressure();
  DEBUG(
    for(SUnit& SU : SUnits)
       SU.dumpAll(this)
  );

  topologicalSort();
  findRootsAndBiasEdges(TopRoots, BotRoots);
  // We reuse several ScheduleDAGMI and ScheduleDAGMILive
  // functions, but to make them happy we must initialize
  // the default Scheduler implementation (even if we do not
  // run it)
  SchedImpl->initialize(this);
  initQueues(TopRoots, BotRoots);

  // Fill some stats to help scheduling.

  SUnitsLinksBackup = SUnits;
  IsLowLatencySU.clear();
  LowLatencyOffset.clear();
  IsHighLatencySU.clear();

  IsLowLatencySU.resize(SUnits.size(), 0);
  LowLatencyOffset.resize(SUnits.size(), 0);
  IsHighLatencySU.resize(SUnits.size(), 0);

  for (unsigned i = 0, e = (unsigned)SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned BaseLatReg;
    int64_t OffLatReg;
    if (SITII->isLowLatencyInstruction(*SU->getInstr())) {
      IsLowLatencySU[i] = 1;
      if (SITII->getMemOpBaseRegImmOfs(*SU->getInstr(), BaseLatReg, OffLatReg,
                                       TRI))
        LowLatencyOffset[i] = OffLatReg;
    } else if (SITII->isHighLatencyInstruction(*SU->getInstr()))
      IsHighLatencySU[i] = 1;
  }

  SIScheduler Scheduler(this);
  Best = Scheduler.scheduleVariant(SISchedulerBlockCreatorVariant::LatenciesAlone,
                                   SISchedulerBlockSchedulerVariant::BlockLatencyRegUsage);

  // if VGPR usage is extremely high, try other good performing variants
  // which could lead to lower VGPR usage
  if (Best.MaxVGPRUsage > 180) {
    static const std::pair<SISchedulerBlockCreatorVariant,
                           SISchedulerBlockSchedulerVariant>
        Variants[] = {
      { LatenciesAlone, BlockRegUsageLatency },
//      { LatenciesAlone, BlockRegUsage },
      { LatenciesGrouped, BlockLatencyRegUsage },
//      { LatenciesGrouped, BlockRegUsageLatency },
//      { LatenciesGrouped, BlockRegUsage },
      { LatenciesAlonePlusConsecutive, BlockLatencyRegUsage },
//      { LatenciesAlonePlusConsecutive, BlockRegUsageLatency },
//      { LatenciesAlonePlusConsecutive, BlockRegUsage }
    };
    for (std::pair<SISchedulerBlockCreatorVariant, SISchedulerBlockSchedulerVariant> v : Variants) {
      Temp = Scheduler.scheduleVariant(v.first, v.second);
      if (Temp.MaxVGPRUsage < Best.MaxVGPRUsage)
        Best = Temp;
    }
  }
  // if VGPR usage is still extremely high, we may spill. Try other variants
  // which are less performing, but that could lead to lower VGPR usage.
  if (Best.MaxVGPRUsage > 200) {
    static const std::pair<SISchedulerBlockCreatorVariant,
                           SISchedulerBlockSchedulerVariant>
        Variants[] = {
//      { LatenciesAlone, BlockRegUsageLatency },
      { LatenciesAlone, BlockRegUsage },
//      { LatenciesGrouped, BlockLatencyRegUsage },
      { LatenciesGrouped, BlockRegUsageLatency },
      { LatenciesGrouped, BlockRegUsage },
//      { LatenciesAlonePlusConsecutive, BlockLatencyRegUsage },
      { LatenciesAlonePlusConsecutive, BlockRegUsageLatency },
      { LatenciesAlonePlusConsecutive, BlockRegUsage }
    };
    for (std::pair<SISchedulerBlockCreatorVariant, SISchedulerBlockSchedulerVariant> v : Variants) {
      Temp = Scheduler.scheduleVariant(v.first, v.second);
      if (Temp.MaxVGPRUsage < Best.MaxVGPRUsage)
        Best = Temp;
    }
  }

  ScheduledSUnits = Best.SUs;
  ScheduledSUnitsInv.resize(SUnits.size());

  for (unsigned i = 0, e = (unsigned)SUnits.size(); i != e; ++i) {
    ScheduledSUnitsInv[ScheduledSUnits[i]] = i;
  }

  moveLowLatencies();

  // Tell the outside world about the result of the scheduling.

  assert(TopRPTracker.getPos() == RegionBegin && "bad initial Top tracker");
  TopRPTracker.setPos(CurrentTop);

  for (std::vector<unsigned>::iterator I = ScheduledSUnits.begin(),
       E = ScheduledSUnits.end(); I != E; ++I) {
    SUnit *SU = &SUnits[*I];
    SU->NumPredsLeft = 0; // To please scheduleMI

    scheduleMI(SU, true);
    SU->isScheduled = true;

    DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                 << *SU->getInstr());
  }

  assert(CurrentTop == CurrentBottom && "Nonempty unscheduled zone.");

  placeDebugValues();

  DEBUG({
      unsigned BBNum = begin()->getParent()->getNumber();
      dbgs() << "*** Final schedule for BB#" << BBNum << " ***\n";
      dumpSchedule();
      dbgs() << '\n';
    });
}
