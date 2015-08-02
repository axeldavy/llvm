//===-- SIMachineScheduler.cpp - SI Scheduler Interface -*- C++ -*-----===//
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

#include "SIMachineScheduler.h"
#include "AMDGPUSubtarget.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterPressure.h"

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
// manage to prevent register pressure from going to high, or it can
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
//   latencies.
//   The current block creation algorithm puts instructions that depend on
//   the same high latency instructions together, and adds constant loadings
//   to the blocks that use them if they are the only user. This algorithm
//   can be improved.
//
// Second the order of the instructions inside the blocks is choosen.
//   At that step we do take into account only register usage and hiding
//   low latency instructions
//
// Third the block order is choosen, there we try to hide high latencies
// and keep register usage low.
//
// After the third step, a pass is done to improve the hiding of low
// latencies.
//
// Actually when talking about 'low latency' or 'high latency' it includes
// both the latency to get the cache (or global mem) data go to the register,
// and the bandwith limitations.
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

#ifndef NDEBUG

const char *getReasonStr(SIScheduleCandReason Reason)  {
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

  // Schedule low latency instructions as top as possible
  // Order of priority is:
  // . Low latency instructions which do not depend on other low latency
  //   instructions we haven't waited for
  // . Other instructions which do not depend on low latency instructions
  //   we haven't waited for
  // . Low latencies
  // . All other instructions
  // Goal is to get: low latency instructions - independant instructions
  //     - (eventually some more low latency instructions)
  //     - instructions that depend on the first low latency instructions.
  // If in the block there is a lot of constant loads, the SGPR usage
  // could go quite high, thus above the arbitrary limit of 60 will encourage
  // use the already loaded constants (in order to release some SGPRs) before
  // loading more
  if (tryLess(TryCand.HasLowLatencyNonWaitedParent,
              Cand.HasLowLatencyNonWaitedParent, TryCand, Cand, Depth))
    return;

  if (Cand.SGPRUsage > 60 &&
      tryLess(TryCand.SGPRUsage, Cand.SGPRUsage, TryCand, Cand, RegUsage))
    return;

  if (tryGreater(TryCand.IsLowLatency, Cand.IsLowLatency,
                 TryCand, Cand, Depth))
    return;

  if (TryCand.IsLowLatency &&
      tryLess(TryCand.LowLatencyOffset, Cand.LowLatencyOffset,
              TryCand, Cand, Depth))
    return;

  if (tryLess(TryCand.VGPRUsage, Cand.VGPRUsage, TryCand, Cand, RegUsage))
    return;

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum < Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
  }
}

SUnit* SIScheduleBlock::pickNode() {
  SISchedCandidate TopCand;

  for (SUnit* SU : TopReadySUs) {
    SISchedCandidate TryCand;
    std::vector<unsigned> pressure;
    std::vector<unsigned> MaxPressure;
    // Predict register usage after this instruction.
    TryCand.SU = SU;
    TopRPTracker.getDownwardPressure(SU->getInstr(), pressure, MaxPressure);
    TryCand.SGPRUsage = pressure[DAG->SGPRSetID];
    TryCand.VGPRUsage = pressure[DAG->VGPRSetID];
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


// Schedule something valid.
void SIScheduleBlock::fastSchedule() {
  TopReadySUs.clear();
  if (Scheduled)
    undoSchedule();

  for (SUnit* SU : SUnits) {
    if (!SU->NumPredsLeft)
      TopReadySUs.push_back(SU);
  }

  while (!TopReadySUs.empty()) {
    SUnit *SU = TopReadySUs[0];
    ScheduledSUnits.push_back(SU);
    NodeScheduled(SU);
  }

  Scheduled = true;
}

// Returns if the register was set between first and last
static bool isDefBetween(unsigned Reg,
                           SlotIndex First, SlotIndex Last,
                           const MachineRegisterInfo *MRI,
                           const LiveIntervals *LIS) {
  for (MachineRegisterInfo::def_instr_iterator
       UI = MRI->def_instr_begin(Reg),
       UE = MRI->def_instr_end(); UI != UE; ++UI) {
    const MachineInstr* MI = &*UI;
    if (MI->isDebugValue())
      continue;
    SlotIndex InstSlot = LIS->getInstructionIndex(MI).getRegSlot();
    if (InstSlot >= First && InstSlot <= Last)
      return true;
  }
  return false;
}

void SIScheduleBlock::initRegPressure(MachineBasicBlock::iterator BeginBlock,
                                      MachineBasicBlock::iterator EndBlock) {
  IntervalPressure Pressure, BotPressure;
  RegPressureTracker RPTracker(Pressure), BotRPTracker(BotPressure);
  LiveIntervals *LIS = DAG->getLIS();
  MachineRegisterInfo *MRI = DAG->getMRI();
  DAG->initRPTracker(TopRPTracker);
  DAG->initRPTracker(BotRPTracker);
  DAG->initRPTracker(RPTracker);

  // Goes though all SU. RPTracker captures what had to be alive for the SUs
  // to execute, and what is still alive at the end
  for (SUnit* SU : ScheduledSUnits) {
    RPTracker.setPos(SU->getInstr());
    RPTracker.advance();
  }

  // Close the RPTracker to finalize live ins/outs.
  RPTracker.closeRegion();

  // Initialize the live ins and live outs.
  TopRPTracker.addLiveRegs(RPTracker.getPressure().LiveInRegs);
  BotRPTracker.addLiveRegs(RPTracker.getPressure().LiveOutRegs);

  // Do not Track Physical Registers, because it messes up
  for (unsigned Reg : RPTracker.getPressure().LiveInRegs) {
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      LiveInRegs.insert(Reg);
  }
  LiveOutRegs.clear();
  // There is several possibilities to distinguish:
  // 1) Reg is not input to any instruction in the block, but is output of one
  // 2) 1) + read in the block and not needed after it
  // 3) 1) + read in the block but needed in another block
  // 4) Reg is input of an instruction but another block will read it too
  // 5) Reg is input of an instruction and then rewritten in the block.
  //    result is not read in the block (implies used in another block)
  // 6) Reg is input of an instruction and then rewritten in the block.
  //    result is read in the block and not needed in another block
  // 7) Reg is input of an instruction and then rewritten in the block.
  //    result is read in the block but also needed in another block
  // LiveInRegs will contains all the regs in situation 4, 5, 6, 7
  // We want LiveOutRegs to contain only Regs whose content will be read after
  // in another block, and whose content was written in the current block,
  // that is we want it to get 1, 3, 5, 7
  // Since we made the MIs of a block to be packed all together before
  // scheduling, then the LiveIntervals were correct, and the RPTracker was
  // able to correctly handle 5 vs 6, 2 vs 3.
  // (Note: This is not sufficient for RPTracker to not do mistakes for case 4)
  // The RPTracker's LiveOutRegs has 1, 3, (some correct or incorrect)4, 5, 7
  // Comparing to LiveInRegs is not sufficient to differenciate 4 vs 5, 7
  // The use of findDefBetween removes the case 4
  for (unsigned Reg : RPTracker.getPressure().LiveOutRegs) {
    if (TargetRegisterInfo::isVirtualRegister(Reg) &&
        isDefBetween(Reg, LIS->getInstructionIndex(BeginBlock).getRegSlot(),
                       LIS->getInstructionIndex(EndBlock).getRegSlot(),
                       MRI, LIS)) {
      LiveOutRegs.insert(Reg);
    }
  }

  // Pressure = sum_alive_registers register size
  // Internally llvm will represent some registers as big 128 bits registers
  // for example, but they actually correspond to 4 actual 32 bits registers.
  // Thus Pressure is not equal to num_alive_registers * constant.
  LiveInPressure = TopPressure.MaxSetPressure;
  LiveOutPressure = BotPressure.MaxSetPressure;

  // Prepares TopRPTracker for top down scheduling.
  TopRPTracker.closeTop();
}

void SIScheduleBlock::schedule(MachineBasicBlock::iterator BeginBlock,
                               MachineBasicBlock::iterator EndBlock) {
  if (!Scheduled)
    fastSchedule();

  // PreScheduling phase to set LiveIn and LiveOut
  initRegPressure(BeginBlock, EndBlock);
  undoSchedule();

  // Schedule for real now

  TopReadySUs.clear();

  for (SUnit* SU : SUnits) {
    if (!SU->NumPredsLeft)
      TopReadySUs.push_back(SU);
  }

  while (!TopReadySUs.empty()) {
    SUnit *SU = pickNode();
    ScheduledSUnits.push_back(SU);
    TopRPTracker.setPos(SU->getInstr());
    TopRPTracker.advance();
    NodeScheduled(SU);
  }

  // TODO: compute InternalAdditionnalPressure
  InternalAdditionnalPressure.resize(TopPressure.MaxSetPressure.size());

  // Check everything is right
#ifndef NDEBUG
  assert(SUnits.size() == ScheduledSUnits.size() &&
            TopReadySUs.empty());
  for (SUnit* SU : SUnits) {
    assert(SU->isScheduled &&
              SU->NumPredsLeft == 0);
  }
#endif

  Scheduled = true;
}

void SIScheduleBlock::undoSchedule() {
  for (SUnit* SU : SUnits) {
    SU->isScheduled = false;
    for (SDep& Succ : SU->Succs) {
      undoReleaseSucc(SU, &Succ, true);
    }
  }
  HasLowLatencyNonWaitedParent.assign(SUnits.size(), 0);
  ScheduledSUnits.clear();
  Scheduled = false;
}

void SIScheduleBlock::undoReleaseSucc(SUnit *SU, SDep *SuccEdge,
                                      bool InOrOutBlock) {
  SUnit *SuccSU = SuccEdge->getSUnit();

  if (BC->isSUInBlock(SuccSU, ID) != InOrOutBlock)
    return;

  if (SuccEdge->isWeak()) {
    ++SuccSU->WeakPredsLeft;
    return;
  }
  ++SuccSU->NumPredsLeft;
}

void SIScheduleBlock::releaseSucc(SUnit *SU, SDep *SuccEdge,
                                  bool InOrOutBlock) {
  SUnit *SuccSU = SuccEdge->getSUnit();

  if (BC->isSUInBlock(SuccSU, ID) != InOrOutBlock)
    return;

  if (SuccEdge->isWeak()) {
    --SuccSU->WeakPredsLeft;
    return;
  }
#ifndef NDEBUG
  if (SuccSU->NumPredsLeft == 0) {
    dbgs() << "*** Scheduling failed! ***\n";
    SuccSU->dump(DAG);
    dbgs() << " has been released too many times!\n";
    llvm_unreachable(nullptr);
  }
#endif

  --SuccSU->NumPredsLeft;
  if (SuccSU->NumPredsLeft == 0 && InOrOutBlock && !SuccSU->isScheduled)
    TopReadySUs.push_back(SuccSU);
}

/// Release Successors of the SU that are in the block or not
void SIScheduleBlock::releaseSuccessors(SUnit *SU, bool InOrOutBlock) {
  for (SDep& Succ : SU->Succs) {
    releaseSucc(SU, &Succ, InOrOutBlock);
  }
}

void SIScheduleBlock::NodeScheduled(SUnit *SU) {
  // Is in TopReadySUs
  assert (!SU->NumPredsLeft);
  std::vector<SUnit*>::iterator I =
    std::find(TopReadySUs.begin(), TopReadySUs.end(), SU);
  if (I == TopReadySUs.end()) {
    dbgs() << "Data Structure Bug in SI Scheduler\n";
    llvm_unreachable(nullptr);
  }
  TopReadySUs.erase(I);

  releaseSuccessors(SU, true);
  // Scheduling this node will trigger a wait,
  // thus propagate to other instructions that they do not need to wait either.
  if (HasLowLatencyNonWaitedParent[NodeNum2Index[SU->NodeNum]])
    HasLowLatencyNonWaitedParent.assign(SUnits.size(), 0);

  if (DAG->IsLowLatencySU[SU->NodeNum]) {
     for (SDep& Succ : SU->Succs) {
      std::map<unsigned, unsigned>::iterator I = NodeNum2Index.find(Succ.getSUnit()->NodeNum);
      if (I != NodeNum2Index.end())
        HasLowLatencyNonWaitedParent[I->second] = 1;
    }
  }
  SU->isScheduled = true;
}

void SIScheduleBlock::finalizeUnits() {
  // We remove links from outside blocks to enable scheduling inside the block
  for (SUnit* SU : SUnits) {
    releaseSuccessors(SU, false);
    if (DAG->IsHighLatencySU[SU->NodeNum])
      HighLatencyBlock = true;
  }
  HasLowLatencyNonWaitedParent.resize(SUnits.size(), 0);
}

// we maintain ascending order of IDs
void SIScheduleBlock::addPred(SIScheduleBlock *Pred) {
  unsigned PredID = Pred->ID;

  // check if not already predecessor
  for (SIScheduleBlock* P : Preds) {
    if (PredID == P->ID)
      return;
  }
  Preds.push_back(Pred);

#ifndef NDEBUG
  for (SIScheduleBlock* S : Succs) {
    if (PredID == S->ID)
      assert(!"Loop in the Block Graph!\n");
  }
#endif
}

void SIScheduleBlock::addSucc(SIScheduleBlock *Succ) {
  unsigned SuccID = Succ->ID;

  // check if not already predecessor
  for (SIScheduleBlock* S : Succs) {
    if (SuccID == S->ID)
      return;
  }
  if (Succ->isHighLatencyBlock())
    ++NumHighLatencySuccessors;
  Succs.push_back(Succ);
#ifndef NDEBUG
  for (SIScheduleBlock* P : Preds) {
    if (SuccID == P->ID)
      assert("Loop in the Block Graph!\n");
  }
#endif
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
  for (SIScheduleBlock* S : Succs) {
    S->printDebug(false);
  }

  if (Scheduled) {
    dbgs() << "LiveInPressure " << LiveInPressure[DAG->SGPRSetID] << " "
           << LiveInPressure[DAG->VGPRSetID] << '\n';
    dbgs() << "LiveOutPressure " << LiveOutPressure[DAG->SGPRSetID] << " "
           << LiveOutPressure[DAG->VGPRSetID] << "\n\n";
    dbgs() << "LiveIns:\n";
    for (unsigned Reg : LiveInRegs)
      dbgs() << PrintVRegOrUnit(Reg, DAG->getTRI()) << " ";

    dbgs() << "\nLiveOuts:\n";
    for (unsigned Reg : LiveOutRegs)
      dbgs() << PrintVRegOrUnit(Reg, DAG->getTRI()) << " ";
  }

  dbgs() << "\nInstructions:\n";
  if (!Scheduled) {
    for (SUnit* SU : SUnits) {
      SU->dump(DAG);
    }
  } else {
    for (SUnit* SU : SUnits) {
      SU->dump(DAG);
    }
  }

   dbgs() << "///////////////////////\n";
}

#endif

// SIScheduleBlockCreator //

SIScheduleBlockCreator::SIScheduleBlockCreator(SIScheduleDAGMI *DAG) :
DAG(DAG) {
  topologicalSort();
}

SIScheduleBlockCreator::~SIScheduleBlockCreator() {
}

std::vector<SIScheduleBlock*>
SIScheduleBlockCreator::getBlocks(SISchedulerBlockCreatorVariant BlockVariant) {
  std::map<SISchedulerBlockCreatorVariant, std::vector<SIScheduleBlock*>>::iterator B = Blocks.find(BlockVariant);
  if (B == Blocks.end()) {
    createBlocksForVariant(BlockVariant);
    scheduleInsideBlocks();
    Blocks[BlockVariant] = CurrentBlocks;
    return CurrentBlocks;
  } else {
    return B->second;
  }
}

bool SIScheduleBlockCreator::isSUInBlock(SUnit *SU, unsigned ID) {
  if (SU->NodeNum >= DAG->SUnits.size())
    return false;
  return CurrentBlocks[Node2CurrentBlock[SU->NodeNum]]->ID == ID;
}

// Code adapted from scheduleDAG.cpp
// Does a topological sort over the SUs.
// Both TopDown and BottomUp
void SIScheduleBlockCreator::topologicalSort() {
  unsigned DAGSize = DAG->SUnits.size();
  std::vector<SUnit*> WorkList;

  DEBUG(dbgs() << "Topological Sort\n");
  WorkList.reserve(DAGSize);

  TopDownIndex2Node.resize(DAGSize);
  TopDownNode2Index.resize(DAGSize);
  BottomUpIndex2Node.resize(DAGSize);
  BottomUpNode2Index.resize(DAGSize);

  WorkList.push_back(&DAG->getExitSU());
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    int NodeNum = SU->NodeNum;
    unsigned Degree = SU->Succs.size();
    TopDownNode2Index[NodeNum] = Degree;
    if (Degree == 0) {
      assert(SU->Succs.empty() && "SUnit should have no successors");
      WorkList.push_back(SU);
    }
  }

  int Id = DAGSize;
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    if (SU->NodeNum < DAGSize) {
      TopDownNode2Index[SU->NodeNum] = --Id;
      TopDownIndex2Node[Id] = SU->NodeNum;
    }
    for (SDep& Pred : SU->Preds) {
      SUnit *SU = Pred.getSUnit();
      if (SU->NodeNum < DAGSize && !--TopDownNode2Index[SU->NodeNum])
        WorkList.push_back(SU);
    }
  }

  WorkList.clear();

  WorkList.push_back(&DAG->getEntrySU());
  for (int i = DAGSize - 1; i >= 0; --i) {
    SUnit *SU = &DAG->SUnits[i];
    int NodeNum = SU->NodeNum;
    unsigned Degree = SU->Preds.size();
    BottomUpNode2Index[NodeNum] = Degree;
    if (Degree == 0) {
      assert(SU->Preds.empty() && "SUnit should have no successors");
      WorkList.push_back(SU);
    }
  }

  Id = DAGSize;
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    if (SU->NodeNum < DAGSize) {
      BottomUpNode2Index[SU->NodeNum] = --Id;
      BottomUpIndex2Node[Id] = SU->NodeNum;
    }
    for (SDep& Succ : SU->Succs) {
      SUnit *SU = Succ.getSUnit();
      if (SU->NodeNum < DAGSize && !--BottomUpNode2Index[SU->NodeNum])
        WorkList.push_back(SU);
    }
  }
#ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    for (SDep& Pred : SU->Preds) {
      if (Pred.getSUnit()->NodeNum >= DAGSize)
        continue;
      assert(TopDownNode2Index[SU->NodeNum] >
             TopDownNode2Index[Pred.getSUnit()->NodeNum] &&
             "Wrong Top Down topological sorting");
    }
  }
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    for (SDep& Succ : SU->Succs) {
      if (Succ.getSUnit()->NodeNum >= DAGSize)
        continue;
      assert(BottomUpNode2Index[SU->NodeNum] >
             BottomUpNode2Index[Succ.getSUnit()->NodeNum] &&
             "Wrong Bottom Up topological sorting");
    }
  }
#endif
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

void SIScheduleBlockCreator::colorHighLatenciesGroups() {
  llvm_unreachable("Not yet Implemented");
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

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[TopDownIndex2Node[i]];
    std::set<unsigned> SUColors;

    // Already given
    if (CurrentColoring[SU->NodeNum]) {
      CurrentTopDownReservedDependencyColoring[SU->NodeNum] = CurrentColoring[SU->NodeNum];
      continue;
    }

   for (SDep& PredDep : SU->Preds) {
      SUnit *Pred = PredDep.getSUnit();
      if (PredDep.isWeak() || Pred->NodeNum >= DAGSize)
        continue;
      if (CurrentTopDownReservedDependencyColoring[Pred->NodeNum] > 0)
        SUColors.insert(CurrentTopDownReservedDependencyColoring[Pred->NodeNum]);
    }
    // color 0 by default
    if (SUColors.empty())
      continue;
    // same color than parents
    if (SUColors.size() == 1 && *SUColors.begin() > DAGSize)
      CurrentTopDownReservedDependencyColoring[SU->NodeNum] = *SUColors.begin();
    else {
      std::map<std::set<unsigned>, unsigned>::iterator Pos =
        ColorCombinations.find(SUColors);
      if (Pos != ColorCombinations.end()) {
          CurrentTopDownReservedDependencyColoring[SU->NodeNum] = Pos->second;
      } else {
        CurrentTopDownReservedDependencyColoring[SU->NodeNum] = NextNonReservedID;
        ColorCombinations[SUColors] = NextNonReservedID++;
      }
    }
  }

  ColorCombinations.clear();

  // Same than before, but BottomUp.

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[BottomUpIndex2Node[i]];
    std::set<unsigned> SUColors;

    // Already given
    if (CurrentColoring[SU->NodeNum]) {
      CurrentBottomUpReservedDependencyColoring[SU->NodeNum] = CurrentColoring[SU->NodeNum];
      continue;
    }

   for (SDep& SuccDep : SU->Succs) {
      SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (CurrentBottomUpReservedDependencyColoring[Succ->NodeNum] > 0)
        SUColors.insert(CurrentBottomUpReservedDependencyColoring[Succ->NodeNum]);
    }
    // keep color 0
    if (SUColors.empty())
      continue;
    // same color than parents
    if (SUColors.size() == 1 && *SUColors.begin() > DAGSize)
      CurrentBottomUpReservedDependencyColoring[SU->NodeNum] = *SUColors.begin();
    else {
      std::map<std::set<unsigned>, unsigned>::iterator Pos =
        ColorCombinations.find(SUColors);
      if (Pos != ColorCombinations.end()) {
          CurrentBottomUpReservedDependencyColoring[SU->NodeNum] = Pos->second;
      } else {
        CurrentBottomUpReservedDependencyColoring[SU->NodeNum] = NextNonReservedID;
        ColorCombinations[SUColors] = NextNonReservedID++;
      }
    }
  }
}

void SIScheduleBlockCreator::colorAccordingToReservedDependencies() {
  unsigned DAGSize = DAG->SUnits.size();
  std::map<std::set<unsigned>, unsigned> ColorCombinations;

  // Every combination of colors given by the top down
  // and bottom up Reserved node dependency

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    std::set<unsigned> SUColors;

    // High latency instructions: already given
    if (CurrentColoring[SU->NodeNum])
      continue;

    SUColors.insert(CurrentTopDownReservedDependencyColoring[SU->NodeNum]);
    SUColors.insert(CurrentBottomUpReservedDependencyColoring[SU->NodeNum]);

    std::map<std::set<unsigned>, unsigned>::iterator Pos =
      ColorCombinations.find(SUColors);
    if (Pos != ColorCombinations.end()) {
      CurrentColoring[SU->NodeNum] = Pos->second;
    } else {
      CurrentColoring[SU->NodeNum] = NextNonReservedID;
      ColorCombinations[SUColors] = NextNonReservedID++;
    }
  }
}

void SIScheduleBlockCreator::colorForceConsecutiveOrderInGroup() {
  llvm_unreachable("Not yet Implemented");
}

void SIScheduleBlockCreator::colorMergeConstantLoadsNextGroup() {
  unsigned DAGSize = DAG->SUnits.size();

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[BottomUpIndex2Node[i]];
    std::set<unsigned> SUColors;

    if (CurrentColoring[SU->NodeNum] <= (int)DAGSize)
      continue;

    // No predecessor: vgpr constant loading
    // low latency instruction usually have a predecessor (the address)
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

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[BottomUpIndex2Node[i]];
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

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[BottomUpIndex2Node[i]];
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

void SIScheduleBlockCreator::createBlocksForVariant(SISchedulerBlockCreatorVariant BlockVariant) {
  unsigned DAGSize = DAG->SUnits.size();
  std::map<unsigned,unsigned> RealID;

  CurrentBlocks.clear();
  CurrentColoring.clear();
  CurrentColoring.resize(DAGSize, 0);

  // Restore links previous scheduling variant has overridden
  DAG->restoreSULinksLeft();

  NextReservedID = 1;
  NextNonReservedID = DAGSize + 1;

  DEBUG(dbgs() << "Coloring the graph\n");

  if (BlockVariant == SISchedulerBlockCreatorVariant::LatenciesAlone) {
    colorHighLatenciesAlone();
    colorComputeReservedDependencies();
    colorAccordingToReservedDependencies();
    colorMergeConstantLoadsNextGroup();
    colorMergeIfPossibleNextGroupOnlyForReserved();
  } else
    llvm_unreachable(nullptr);

  // Put SUs of same color into same block
  Node2CurrentBlock.resize(DAGSize, -1);
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    unsigned Color = CurrentColoring[SU->NodeNum];
    if (RealID.find(Color) == RealID.end()) {
      int ID = CurrentBlocks.size();
      BlockPtrs.push_back(
        make_unique<SIScheduleBlock>(DAG, this, ID));
      CurrentBlocks.push_back(BlockPtrs.rbegin()->get());
      RealID[Color] = ID;
    }
    CurrentBlocks[RealID[Color]]->addUnit(SU);
    Node2CurrentBlock[SU->NodeNum] = RealID[Color];
  }

  // Build dependencies between blocks
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &DAG->SUnits[i];
    int SUID = Node2CurrentBlock[i];
     for (SDep& SuccDep : SU->Succs) {
       SUnit *Succ = SuccDep.getSUnit();
      if (SuccDep.isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (Node2CurrentBlock[Succ->NodeNum] != SUID)
        CurrentBlocks[SUID]->addSucc(CurrentBlocks[Node2CurrentBlock[Succ->NodeNum]]);
    }
    for (SDep& PredDep : SU->Preds) {
      SUnit *Pred = PredDep.getSUnit();
      if (PredDep.isWeak() || Pred->NodeNum >= DAGSize)
        continue;
      if (Node2CurrentBlock[Pred->NodeNum] != SUID)
        CurrentBlocks[SUID]->addPred(CurrentBlocks[Node2CurrentBlock[Pred->NodeNum]]);
    }
  }

  // free root and leafs of all blocks to enable scheduling inside them
  for (unsigned i = 0, e = CurrentBlocks.size(); i != e; ++i) {
    SIScheduleBlock *Block = CurrentBlocks[i];
    Block->finalizeUnits();
  }
  DEBUG(
    dbgs() << "Blocks created:\n\n";
    for (unsigned i = 0, e = CurrentBlocks.size(); i != e; ++i) {
      SIScheduleBlock *Block = CurrentBlocks[i];
      Block->printDebug(true);
    }
  );
}

// two functions taken from Codegen/MachineScheduler.cpp

/// If this iterator is a debug value, increment until reaching the End or a
/// non-debug instruction.
static MachineBasicBlock::const_iterator
nextIfDebug(MachineBasicBlock::const_iterator I,
            MachineBasicBlock::const_iterator End) {
  for(; I != End; ++I) {
    if (!I->isDebugValue())
      break;
  }
  return I;
}

/// Non-const version.
static MachineBasicBlock::iterator
nextIfDebug(MachineBasicBlock::iterator I,
            MachineBasicBlock::const_iterator End) {
  // Cast the return value to nonconst MachineInstr, then cast to an
  // instr_iterator, which does not check for null, finally return a
  // bundle_iterator.
  return MachineBasicBlock::instr_iterator(
    const_cast<MachineInstr*>(
      &*nextIfDebug(MachineBasicBlock::const_iterator(I), End)));
}

void SIScheduleBlockCreator::scheduleInsideBlocks() {
  unsigned DAGSize = CurrentBlocks.size();
  std::vector<int> WorkList;
  std::vector<int> TopDownIndex2Indice;
  std::vector<int> TopDownIndice2Index;

  DEBUG(dbgs() << "\nScheduling Blocks\n\n");

  // We do schedule a valid scheduling such that a Block corresponds
  // to a range of instructions. That will guarantee that if a result
  // produced inside the Block, but also used inside the Block, while it
  // gets used in another Block, then it will be correctly counted in the
  // Live Output Regs of the Block
  DEBUG(dbgs() << "First phase: Fast scheduling for Reg Liveness\n");
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIScheduleBlock *Block = CurrentBlocks[i];
    Block->fastSchedule();
  }

  WorkList.reserve(DAGSize);

  TopDownIndex2Indice.resize(DAGSize);
  TopDownIndice2Index.resize(DAGSize);

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIScheduleBlock *Block = CurrentBlocks[i];
    unsigned Degree = Block->Succs.size();
    TopDownIndice2Index[i] = Degree;
    if (Degree == 0) {
      WorkList.push_back(i);
    }
  }

  int Id = DAGSize;
  while (!WorkList.empty()) {
    int i = WorkList.back();
    SIScheduleBlock *Block = CurrentBlocks[i];
    WorkList.pop_back();
    TopDownIndice2Index[i] = --Id;
    TopDownIndex2Indice[Id] = i;
    for (SIScheduleBlock* Pred : Block->Preds) {
      if (!--TopDownIndice2Index[Pred->ID])
        WorkList.push_back(Pred->ID);
    }
  }

#ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIScheduleBlock *Block = CurrentBlocks[i];
    for (SIScheduleBlock* Pred : Block->Preds) {
      assert(TopDownIndice2Index[i] > TopDownIndice2Index[Pred->ID] &&
      "Wrong Top Down topological sorting");
    }
  }
#endif

  // Note: the following code, and the part restoring previous position
  // is by far the most expensive operation of the Scheduler.

  // Do not update CurrentTop
  MachineBasicBlock::iterator CurrentTopFastSched = DAG->getCurrentTop();
  std::vector<MachineBasicBlock::iterator> PosOld;
  std::vector<MachineBasicBlock::iterator> PosNew;
  PosOld.reserve(DAG->SUnits.size());
  PosNew.reserve(DAG->SUnits.size());

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    int BlockIndice = TopDownIndex2Indice[i];
    SIScheduleBlock *Block = CurrentBlocks[BlockIndice];
    std::vector<SUnit*> SUs = Block->getScheduledUnits();

    for (SUnit* SU : SUs) {
      MachineInstr *MI = SU->getInstr();
      MachineBasicBlock::iterator Pos = MI;
      PosOld.push_back(Pos);
      if (&*CurrentTopFastSched == MI) {
        PosNew.push_back(Pos);
        CurrentTopFastSched = nextIfDebug(++CurrentTopFastSched,
                                          DAG->getCurrentBottom());
      } else {
        // Update the instruction stream.
        DAG->getBB()->splice(CurrentTopFastSched, DAG->getBB(), MI);

        // Update LiveIntervals
        DAG->getLIS()->handleMove(MI, /*UpdateFlags=*/true);
        PosNew.push_back(CurrentTopFastSched);
      }
    }
  }

  // Now we have Block of SUs == Block of MI
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIScheduleBlock *Block = CurrentBlocks[i];
    std::vector<SUnit*> SUs = Block->getScheduledUnits();
    Block->schedule((*SUs.begin())->getInstr(), (*SUs.rbegin())->getInstr());
  }

  DEBUG(dbgs() << "Restoring MI Pos\n");
  // Restore old ordering (which guarantees LIS->handleMove to be happy)
  for (unsigned i = PosOld.size(), e = 0; i != e; --i) {
    MachineBasicBlock::iterator POld = PosOld[i-1];
    MachineBasicBlock::iterator PNew = PosNew[i-1];
    if (PNew != POld) {
      // Update the instruction stream.
      DAG->getBB()->splice(POld, DAG->getBB(), PNew);

      // Update LiveIntervals
      DAG->getLIS()->handleMove(POld, /*UpdateFlags=*/true);
    }
  }

  DEBUG(
    for (unsigned i = 0, e = CurrentBlocks.size(); i != e; ++i) {
      SIScheduleBlock *Block = CurrentBlocks[i];
      Block->printDebug(true);
    }
  );
}

// SIScheduleBlockScheduler //

SIScheduleBlockScheduler::SIScheduleBlockScheduler(SIScheduleDAGMI *DAG,
                                                   SISchedulerBlockSchedulerVariant Variant,
                                                   std::vector<SIScheduleBlock*> Blocks) :
  DAG(DAG), Variant(Variant), Blocks(Blocks), NumBlockScheduled(0), maxVregUsage(0), maxSregUsage(0) {
  // Fill the usage of every output
  // By construction we have that for every Reg input
  // of a block, it is present one time max in the outputs
  // of its Block predecessors. If it is not, it means it Reads
  // a register whose content was already filled at start
  LiveOutRegsNumUsages.resize(Blocks.size());
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIScheduleBlock *Block = Blocks[i];
    for (unsigned Reg : Block->getInRegs()) {
      for (SIScheduleBlock* Pred : Block->Preds) {
        std::map<unsigned, unsigned>::iterator RegPos =
          LiveOutRegsNumUsages[Pred->ID].find(Reg);
        if (RegPos != LiveOutRegsNumUsages[Pred->ID].end()) {
          ++LiveOutRegsNumUsages[Pred->ID][Reg];
          break;
        } else {
          LiveOutRegsNumUsages[Pred->ID][Reg] = 1;
        }
      }
    }
  }

  LastPosHighLatencyParentScheduled.resize(Blocks.size(), 0);
  BlockNumPredsLeft.resize(Blocks.size());
  BlockNumSuccsLeft.resize(Blocks.size());

  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIScheduleBlock *Block = Blocks[i];
    BlockNumPredsLeft[i] = Block->Preds.size();
    BlockNumSuccsLeft[i] = Block->Succs.size();
  }

#ifndef NDEBUG
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIScheduleBlock *Block = Blocks[i];
    assert(Block->ID == i);
  }
#endif

  std::set<unsigned> InRegs = DAG->getInRegs();
  addLiveRegs(InRegs);

  // fill LiveRegsConsumers for regs that were already
  // defined before scheduling
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIScheduleBlock *Block = Blocks[i];
    for (unsigned Reg : Block->getInRegs()) {
      bool Found = false;
      for (SIScheduleBlock* Pred: Block->Preds) {
        std::set<unsigned> PredOutRegs = Pred->getOutRegs();
        std::set<unsigned>::iterator RegPos = PredOutRegs.find(Reg);

        if (RegPos != PredOutRegs.end()) {
          Found = true;
          break;
        }
      }

      if (!Found) {
        if (LiveRegsConsumers.find(Reg) == LiveRegsConsumers.end())
          LiveRegsConsumers[Reg] = 1;
        else
          ++LiveRegsConsumers[Reg];
      }
    }
  }

  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIScheduleBlock *Block = Blocks[i];
    if (BlockNumPredsLeft[i] == 0) {
      ReadyBlocks.push_back(Block);
    }
  }

  if (Variant != SISchedulerBlockSchedulerVariant::BlockLatency)
    llvm_unreachable(nullptr);

  while (SIScheduleBlock *Block = pickBlock()) {
    BlocksScheduled.push_back(Block);
    blockScheduled(Block);
  }

  DEBUG(
    dbgs() << "Block Order:";
    for (SIScheduleBlock* Block : BlocksScheduled) {
      dbgs() << " " << Block->ID;
    }
  );
}

void SIScheduleBlockScheduler::tryCandidate(SIBlockSchedCandidate &Cand,
                                            SIBlockSchedCandidate &TryCand) {
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return;
  }

  // Try not to increase VGPR usage too much, else we may spill
  if (VregCurrentUsage > 180 &&
      tryLess(TryCand.VGPRUsageDiff > 0, Cand.VGPRUsageDiff > 0,
              TryCand, Cand, RegUsage))
    return;
  // Try to hide high latencies
  if (tryLess(TryCand.LastPosHighLatParentScheduled,
              Cand.LastPosHighLatParentScheduled, TryCand, Cand, Latency))
    return;
  // schedules high latencies early so you can hide them better
  if (tryGreater(TryCand.IsHighLatency, Cand.IsHighLatency,
                 TryCand, Cand, Latency))
    return;
  if (tryGreater(TryCand.NumHighLatencySuccessors,
                 Cand.NumHighLatencySuccessors,
                 TryCand, Cand, Successor))
    return;
  if (tryLess(TryCand.VGPRUsageDiff, Cand.VGPRUsageDiff,
              TryCand, Cand, RegUsage))
    return;
}

SIScheduleBlock *SIScheduleBlockScheduler::pickBlock() {
  SIBlockSchedCandidate Cand;
  std::vector<SIScheduleBlock*>::iterator Best;
  SIScheduleBlock *Block;
  if (ReadyBlocks.empty())
    return nullptr;

  DAG->fillVgprSgprCost(LiveRegs.begin(), LiveRegs.end(), VregCurrentUsage, SregCurrentUsage);
  if (VregCurrentUsage > maxVregUsage)
    maxVregUsage = VregCurrentUsage;
  if (VregCurrentUsage > maxSregUsage)
    maxSregUsage = VregCurrentUsage;
  DEBUG(
    dbgs() << "Picking New Blocks\n";
    dbgs() << "Available: ";
    for (SIScheduleBlock* Block : ReadyBlocks)
      dbgs() << Block->ID << " ";
    dbgs() << "\nCurrent Live:\n";
    for (unsigned Reg : LiveRegs)
      dbgs() << PrintVRegOrUnit(Reg, DAG->getTRI()) << " ";
    dbgs() << '\n';
    dbgs() << "Current VGPRs: " << VregCurrentUsage << '\n';
    dbgs() << "Current SGPRs: " << SregCurrentUsage << '\n';
  );

  Cand.Block = nullptr;
  for (std::vector<SIScheduleBlock*>::iterator I = ReadyBlocks.begin(),
       E = ReadyBlocks.end(); I != E; ++I) {
    SIBlockSchedCandidate TryCand;
    TryCand.Block = *I;
    TryCand.IsHighLatency = TryCand.Block->isHighLatencyBlock();
    TryCand.VGPRUsageDiff =
      checkRegUsageImpact(TryCand.Block->getInRegs(),
                          TryCand.Block->getOutRegs())[DAG->VGPRSetID];
    TryCand.NumHighLatencySuccessors = TryCand.Block->NumHighLatencySuccessors;
    TryCand.LastPosHighLatParentScheduled =
      LastPosHighLatencyParentScheduled[TryCand.Block->ID];
    tryCandidate(Cand, TryCand);
    if (TryCand.Reason != NoCand) {
      Cand.setBest(TryCand);
      Best = I;
      DEBUG(dbgs() << "Best Current Choice: " << Cand.Block->ID << " "
                   << getReasonStr(Cand.Reason) << '\n');
    }
  }

  DEBUG(
    dbgs() << "Picking: " << Cand.Block->ID << '\n';
    dbgs() << "Is a block with high latency instruction: "
      << (Cand.IsHighLatency ? "yes\n" : "no\n");
    dbgs() << "Position of last high latency dependency: "
           << Cand.LastPosHighLatParentScheduled << '\n';
    dbgs() << "VGPRUsageDiff: " << Cand.VGPRUsageDiff << '\n';
    dbgs() << '\n';
  );

  Block = Cand.Block;
  ReadyBlocks.erase(Best);
  return Block;
}

// Tracking of currently alive register to determine VGPR Usage

void SIScheduleBlockScheduler::addLiveRegs(std::set<unsigned> &Regs) {
  for (unsigned Reg : Regs) {
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    // if not already in the live set, then add it
    (void) LiveRegs.insert(Reg);
  }
}

void SIScheduleBlockScheduler::decreaseLiveRegs(SIScheduleBlock *Block,
                                       std::set<unsigned> &Regs) {
  for (unsigned Reg : Regs) {
    // For now only track virtual registers
    std::set<unsigned>::iterator Pos = LiveRegs.find(Reg);
    assert (Pos != LiveRegs.end() && // Reg must be live
               LiveRegsConsumers.find(Reg) != LiveRegsConsumers.end() &&
               LiveRegsConsumers[Reg] >= 1);
    --LiveRegsConsumers[Reg];
    if (LiveRegsConsumers[Reg] == 0)
      LiveRegs.erase(Pos);
  }
}

void SIScheduleBlockScheduler::releaseBlockSuccs(SIScheduleBlock *Parent) {
  for (SIScheduleBlock* Block : Parent->Succs) {
    --BlockNumPredsLeft[Block->ID];
    if (BlockNumPredsLeft[Block->ID] == 0) {
      ReadyBlocks.push_back(Block);
    }
    if (Parent->isHighLatencyBlock())
      LastPosHighLatencyParentScheduled[Block->ID] = NumBlockScheduled;
  }
}

void SIScheduleBlockScheduler::blockScheduled(SIScheduleBlock *Block) {
  decreaseLiveRegs(Block, Block->getInRegs());
  addLiveRegs(Block->getOutRegs());
  releaseBlockSuccs(Block);
  for (std::map<unsigned, unsigned>::iterator RegI =
       LiveOutRegsNumUsages[Block->ID].begin(),
       E = LiveOutRegsNumUsages[Block->ID].end(); RegI != E; ++RegI) {
    std::pair<unsigned, unsigned> RegP = *RegI;
    if (LiveRegsConsumers.find(RegP.first) == LiveRegsConsumers.end())
      LiveRegsConsumers[RegP.first] = RegP.second;
    else
      LiveRegsConsumers[RegP.first] += RegP.second;
  }
  ++NumBlockScheduled;
}

std::vector<int>
SIScheduleBlockScheduler::checkRegUsageImpact(std::set<unsigned> &InRegs,
                                     std::set<unsigned> &OutRegs) {
  std::vector<int> DiffSetPressure;
  DiffSetPressure.assign(DAG->getTRI()->getNumRegPressureSets(), 0);

  for (unsigned Reg : InRegs) {
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (LiveRegsConsumers[Reg] > 1)
      continue;
    PSetIterator PSetI = DAG->getMRI()->getPressureSets(Reg);
    for (; PSetI.isValid(); ++PSetI) {
      DiffSetPressure[*PSetI] -= PSetI.getWeight();
    }
  }

  for (unsigned Reg : OutRegs) {
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    PSetIterator PSetI = DAG->getMRI()->getPressureSets(Reg);
    for (; PSetI.isValid(); ++PSetI) {
      DiffSetPressure[*PSetI] += PSetI.getWeight();
    }
  }

  return DiffSetPressure;
}

// SIScheduler //

std::vector<unsigned>
SIScheduler::scheduleVariant(SISchedulerBlockCreatorVariant BlockVariant,
                             SISchedulerBlockSchedulerVariant ScheduleVariant) {
  std::vector<SIScheduleBlock*> Blocks = BlockCreator.getBlocks(BlockVariant);
  SIScheduleBlockScheduler Scheduler(DAG, ScheduleVariant, Blocks);
  std::vector<unsigned> Res;

  Blocks = Scheduler.getBlocks();

  for (unsigned b = 0; b < Blocks.size(); ++b) {
    SIScheduleBlock *Block = Blocks[b];
    std::vector<SUnit*> SUs = Block->getScheduledUnits();

    for (SUnit* SU : SUs)
      Res.push_back(SU->NodeNum);
  }

  return Res;
}

// SIScheduleDAGMI //

SIScheduleDAGMI::SIScheduleDAGMI(MachineSchedContext *C) :
  ScheduleDAGMILive(C, make_unique<GenericScheduler>(C)) {
  SITII = static_cast<const SIInstrInfo*>(TII);
  SITRI = static_cast<const SIRegisterInfo*>(TRI);

  VGPRSetID = SITRI->getVGPR32PressureSet();
  SGPRSetID = SITRI->getSGPR32PressureSet();
}

SIScheduleDAGMI::~SIScheduleDAGMI() {
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
      if (SITII->isLowLatencyInstruction(Pred->getInstr())) {
        IsLowLatencyUser = true;
      }
      if (Pred->NodeNum >= DAGSize)
        continue;
      unsigned PredPos = ScheduledSUnitsInv[Pred->NodeNum];
      if (PredPos >= MinPos)
        MinPos = PredPos + 1;
    }

    if (SITII->isLowLatencyInstruction(SU->getInstr())) {
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
    // the low latency instructions too
    } else if (SU->getInstr()->getOpcode() == AMDGPU::COPY) {
      bool CopyForLowLat = false;
      for (SDep& SuccDep : SU->Succs) {
        SUnit *Succ = SuccDep.getSUnit();
        if (SITII->isLowLatencyInstruction(Succ->getInstr())) {
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

void SIScheduleDAGMI::restoreSULinksLeft() {
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    SUnits[i].WeakPredsLeft = SUnitsLinksBackup[i].WeakPredsLeft;
    SUnits[i].NumPredsLeft = SUnitsLinksBackup[i].NumPredsLeft;
    SUnits[i].WeakSuccsLeft = SUnitsLinksBackup[i].WeakSuccsLeft;
    SUnits[i].NumSuccsLeft = SUnitsLinksBackup[i].NumSuccsLeft;
  }
}

// return the vgpr usage corresponding to some virtual registers
template<typename _Iterator> void
SIScheduleDAGMI::fillVgprSgprCost(_Iterator First, _Iterator End,
                                  unsigned &VgprUsage, unsigned &SgprUsage) {
  VgprUsage = 0;
  SgprUsage = 0;
  for (_Iterator RegI = First; RegI != End; ++RegI) {
    unsigned Reg = *RegI;
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    PSetIterator PSetI = MRI.getPressureSets(Reg);
    for (; PSetI.isValid(); ++PSetI) {
      if (*PSetI == VGPRSetID)
        VgprUsage += PSetI.getWeight();
      else if (*PSetI == SGPRSetID)
        SgprUsage += PSetI.getWeight();
    }
  }
}

void SIScheduleDAGMI::schedule()
{
  SmallVector<SUnit*, 8> TopRoots, BotRoots;
  DEBUG(dbgs() << "Preparing Scheduling\n");

  buildDAGWithRegPressure();
  DEBUG(
    for(SUnit& SU : SUnits)
       SU.dumpAll(this)
  );

  findRootsAndBiasEdges(TopRoots, BotRoots);
  // We reuse several ScheduleDAGMI and ScheduleDAGMILive
  // functions, but to make them happy we must initialize
  // the default Scheduler implementation (even if we do not
  // run it)
  SchedImpl->initialize(this);
  initQueues(TopRoots, BotRoots);

  // fill some stats to help scheduling

  SUnitsLinksBackup = SUnits;
  IsLowLatencySU.clear();
  LowLatencyOffset.clear();
  IsHighLatencySU.clear();

  IsLowLatencySU.resize(SUnits.size(), 0);
  LowLatencyOffset.resize(SUnits.size(), 0);
  IsHighLatencySU.resize(SUnits.size(), 0);

  for (unsigned i = 0, e = (unsigned)SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned BaseLatReg, OffLatReg;
    if (SITII->isLowLatencyInstruction(SU->getInstr())) {
      IsLowLatencySU[i] = 1;
      if (SITII->getMemOpBaseRegImmOfs(SU->getInstr(), BaseLatReg,
                                      OffLatReg, TRI))
        LowLatencyOffset[i] = OffLatReg;
    } else if (SITII->isHighLatencyInstruction(SU->getInstr()))
      IsHighLatencySU[i] = 1;
  }

  SIScheduler Scheduler(this);
  ScheduledSUnits = Scheduler.scheduleVariant(SISchedulerBlockCreatorVariant::LatenciesAlone,
                                              SISchedulerBlockSchedulerVariant::BlockLatency);
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

    scheduleMI(SU, true);

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
