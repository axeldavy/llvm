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
  case NoCand:         return "NOCAND    ";
  case RegUsage:       return "REGUSAGE  ";
  case Latency:        return "LATENCY   ";
  case Successor:      return "SUCCESSOR ";
  case Depth:          return "DEPTH     ";
  case NodeOrder:      return "ORDER     ";
  };
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

// SIBlockSchedule //

void SIBlockSchedule::addUnit(SUnit *SU) {
  SUnits.push_back(SU);
}

#ifndef NDEBUG

void SIBlockSchedule::traceCandidate(const SISchedCandidate &Cand) {

  dbgs() << "  SU(" << Cand.SU->NodeNum << ") " << getReasonStr(Cand.Reason);
  dbgs() << '\n';
}
#endif

void SIBlockSchedule::tryCandidateTopDown(SISchedCandidate &Cand,
                                          SISchedCandidate &TryCand) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return;
  }

  // Schedule low latency instructions as top as possible
  // Order of priority is:
  // . Low latency instructions which do not depend on low latencies we
  //   haven't waited
  // . Other instructions which do not depend on low latencies we haven't
  //   waited
  // . Low latencies
  // . All other instructions
  // Goal is to get: low latency instructions - independant instructions
  //     - (eventually some more low latency instructions)
  //     - instructions that depend on the first low latency instructions.
  // If never in the block there is a lot of constant loads, the SGPR usage
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

SUnit* SIBlockSchedule::pickNode() {
  SISchedCandidate TopCand;

  for (std::vector<SUnit*>::iterator I = TopReadySUs.begin(),
       E = TopReadySUs.end(); I != E; ++I) {
    SISchedCandidate TryCand;
    std::vector<unsigned> pressure = TopRPTracker.getRegSetPressureAtPos();
    std::vector<unsigned> MaxPressure = TopRPTracker.getRegSetPressureAtPos();
    // predict register usage after this instruction
    TopRPTracker.getDownwardPressure((*I)->getInstr(), pressure, MaxPressure);
    TryCand.SU = *I;
    TryCand.SGPRUsage = pressure[DAG->SGPRSetID];
    TryCand.VGPRUsage = pressure[DAG->VGPRSetID];
    TryCand.IsLowLatency = DAG->IsLowlatencySU[(*I)->NodeNum];
    TryCand.LowLatencyOffset = DAG->LowLatencyOffset[(*I)->NodeNum];
    TryCand.HasLowLatencyNonWaitedParent =
      DAG->HasLowLatencyNonWaitedParent[(*I)->NodeNum];
    tryCandidateTopDown(TopCand, TryCand);
    if (TryCand.Reason != NoCand) {
      TopCand.setBest(TryCand);
      //DEBUG(traceCandidate(Cand));
    }
  }

  return TopCand.SU;
}


// Schedule something valid.
void SIBlockSchedule::fastSchedule() {
  TopReadySUs.clear();

  for (std::vector<SUnit*>::iterator I = SUnits.begin(),
       E = SUnits.end(); I != E; ++I) {
    SUnit *SU = *I;

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
static bool findDefBetween(unsigned Reg,
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

void SIBlockSchedule::initRegPressure(MachineBasicBlock::iterator BeginBlock,
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
  for (std::vector<SUnit*>::iterator I = ScheduledSUnits.begin(),
       E = ScheduledSUnits.end(); I != E; ++I) {
    RPTracker.setPos((*I)->getInstr());
    RPTracker.advance();
  }

  // Close the RPTracker to finalize live ins/outs.
  RPTracker.closeRegion();

  // Initialize the live ins and live outs.
  TopRPTracker.addLiveRegs(RPTracker.getPressure().LiveInRegs);
  BotRPTracker.addLiveRegs(RPTracker.getPressure().LiveOutRegs);

  // Do not Track Physical Registers, because it messes up
  for (SmallVector<unsigned, 8>::iterator I =
       RPTracker.getPressure().LiveInRegs.begin(),
       E = RPTracker.getPressure().LiveInRegs.end(); I != E; ++I) {
    unsigned Reg = *I;
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      LiveInRegs.insert(Reg);
  }
  LiveOutRegs.clear();
  LiveOutRegsNumUsages.clear();
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
  for (SmallVector<unsigned, 8>::iterator I =
       RPTracker.getPressure().LiveOutRegs.begin(),
       E = RPTracker.getPressure().LiveOutRegs.end(); I != E; ++I) {
    unsigned Reg = *I;
    if (findDefBetween(Reg, LIS->getInstructionIndex(BeginBlock).getRegSlot(),
                       LIS->getInstructionIndex(EndBlock).getRegSlot(),
                       MRI, LIS) &&
        TargetRegisterInfo::isVirtualRegister(Reg)) {
      LiveOutRegs.insert(Reg);
      // LiveOutRegsNumUsages will be filled later by the main scheduler
      LiveOutRegsNumUsages[Reg] = 0;
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

void SIBlockSchedule::schedule(MachineBasicBlock::iterator BeginBlock,
                               MachineBasicBlock::iterator EndBlock) {
  if (!Scheduled)
    fastSchedule();

  // PreScheduling phase to set LiveIn and LiveOut
  initRegPressure(BeginBlock, EndBlock);
  undoSchedule();

  // Schedule for real now

  TopReadySUs.clear();

  for (std::vector<SUnit*>::iterator I = SUnits.begin(),
       E = SUnits.end(); I != E; ++I) {
    SUnit *SU = *I;

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
  assert(SUnits.size() == ScheduledSUnits.size());
  assert(TopReadySUs.empty());
  for (std::vector<SUnit*>::iterator I = SUnits.begin(),
       E = SUnits.end(); I != E; ++I) {
    SUnit *SU = *I;
    assert (SU->isScheduled);
    assert (SU->NumPredsLeft == 0);
  }
#endif

  Scheduled = true;
}

void SIBlockSchedule::undoSchedule() {
  for (std::vector<SUnit*>::iterator I = SUnits.begin(),
       E = SUnits.end(); I != E; ++I) {
    SUnit *SU = *I;
    SU->isScheduled = false;
    for (SUnit::succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      undoReleaseSucc(SU, &*I, true);
    }
  }
  ScheduledSUnits.clear();
  Scheduled = false;
}

void SIBlockSchedule::undoReleaseSucc(SUnit *SU, SDep *SuccEdge,
                                      bool InOrOutBlock) {
  SUnit *SuccSU = SuccEdge->getSUnit();

  if (DAG->isSUInBlock(SuccSU, ID) != InOrOutBlock)
    return;

  if (SuccEdge->isWeak()) {
    ++SuccSU->WeakPredsLeft;
    return;
  }
  ++SuccSU->NumPredsLeft;
}

void SIBlockSchedule::releaseSucc(SUnit *SU, SDep *SuccEdge,
                                  bool InOrOutBlock) {
  SUnit *SuccSU = SuccEdge->getSUnit();

  if (DAG->isSUInBlock(SuccSU, ID) != InOrOutBlock)
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
void SIBlockSchedule::releaseSuccessors(SUnit *SU, bool InOrOutBlock) {
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    releaseSucc(SU, &*I, InOrOutBlock);
  }
}

void SIBlockSchedule::NodeScheduled(SUnit *SU) {
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
  // scheduling this node will trigger a wait,
  // thus propagate to other instructions that they do not need to wait either
  if (DAG->HasLowLatencyNonWaitedParent[SU->NodeNum])
    DAG->propagateWaitedLatencies();
  SU->isScheduled = true;
}

// we maintain ascending order of IDs
void SIBlockSchedule::addPred(SIBlockSchedule *Pred) {
  unsigned PredID = Pred->ID;

  // check if not already predecessor
  for (std::vector<SIBlockSchedule*>::iterator I = Preds.begin(),
       E = Preds.end(); I != E; ++I) {
    if (PredID == (*I)->ID)
      return;
  }
  Preds.push_back(Pred);
  ++NumPredsLeft;
}

void SIBlockSchedule::addSucc(SIBlockSchedule *Succ) {
  unsigned SuccID = Succ->ID;

  // check if not already predecessor
  for (std::vector<SIBlockSchedule*>::iterator I = Succs.begin(),
       E = Succs.end(); I != E; ++I) {
    if (SuccID == (*I)->ID)
      return;
  }
  if (Succ->isHighLatencyBlock())
    ++NumHighLatencySuccessors;
  Succs.push_back(Succ);
  ++NumSuccsLeft;
}

#ifndef NDEBUG
void SIBlockSchedule::printDebug(bool full) {
  dbgs() << "Block (" << ID << ")\n";
  if (!full)
    return;

  dbgs() << "\nContains High Latency Instruction: "
         << HighLatencyBlock << "\n";
  dbgs() << "\nDepends On:\n";
  for (std::vector<SIBlockSchedule*>::iterator I = Preds.begin(),
       E = Preds.end(); I != E; ++I) {
    (*I)->printDebug(false);
  }

  dbgs() << "\nSuccessors:\n";
  for (std::vector<SIBlockSchedule*>::iterator I = Succs.begin(),
       E = Succs.end(); I != E; ++I) {
    (*I)->printDebug(false);
  }

  if (Scheduled) {
    dbgs() << "LiveInPressure " << LiveInPressure[DAG->SGPRSetID] << " "
           << LiveInPressure[DAG->VGPRSetID] << "\n";
    dbgs() << "LiveOutPressure " << LiveOutPressure[DAG->SGPRSetID] << " "
           << LiveOutPressure[DAG->VGPRSetID] << "\n\n";
    dbgs() << "LiveIns:\n";
    for (std::set<unsigned>::iterator RegI = LiveInRegs.begin(),
         E = LiveInRegs.end(); RegI != E; ++RegI) {
      unsigned Reg = *RegI;
      dbgs() << PrintVRegOrUnit(Reg, DAG->getTRI()) << " ";
    }
    dbgs() << "\nLiveOuts (N usage):\n";
    for (std::map<unsigned, unsigned>::iterator RegI =
         LiveOutRegsNumUsages.begin(), E = LiveOutRegsNumUsages.end();
         RegI != E; ++RegI) {
      std::pair<unsigned, unsigned> RegP = *RegI;
      dbgs() << PrintVRegOrUnit(RegP.first, DAG->getTRI()) << " ("
             << RegP.second <<"), ";
    }
  }

  dbgs() << "\nInstructions:\n";
  if (!Scheduled) {
    for (std::vector<SUnit*>::iterator I = SUnits.begin(),
         E = SUnits.end(); I != E; ++I) {
      (*I)->dump(DAG);
    }
  } else {
    for (std::vector<SUnit*>::iterator I = ScheduledSUnits.begin(),
         E = ScheduledSUnits.end(); I != E; ++I) {
      (*I)->dump(DAG);
    }
  }

   dbgs() << "///////////////////////\n";
}

#endif

// SIScheduleDAGMI //

SIScheduleDAGMI::SIScheduleDAGMI(MachineSchedContext *C) :
  ScheduleDAGMILive(C, make_unique<GenericScheduler>(C)) {
  int i;
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  SITII = static_cast<const SIInstrInfo*>(TII);
  SITRI = static_cast<const SIRegisterInfo*>(TRI);

  for (i=0; i<10; ++i) {
    SGPRsForWaveFronts[i] = SITRI->getNumSGPRsAllowed(ST.getGeneration(), i+1);
    VGPRsForWaveFronts[i] = SITRI->getNumVGPRsAllowed(i+1);
  }
  VGPRSetID = SITRI->getVGPR32PressureSet();
  SGPRSetID = SITRI->getSGPR32PressureSet();
}

SIScheduleDAGMI::~SIScheduleDAGMI() {
}

void SIScheduleDAGMI::prepareSchedule() {
  DEBUG(dbgs() << "Preparing Scheduling\n");
  buildDAGWithRegPressure();
  DEBUG(for (std::vector<SUnit>::iterator I = SUnits.begin(),
             E = SUnits.end(); I != E; ++I)
          (*I).dumpAll(this));
  topologicalSort();
  SmallVector<SUnit*, 8> TopRoots, BotRoots;
  findRootsAndBiasEdges(TopRoots, BotRoots);
  SchedImpl->initialize(this);
  initQueues(TopRoots, BotRoots);
  createBlocks();
  scheduleInsideBlocks();
}

// Code adapted from scheduleDAG.cpp
// Does a topological sort over the SUs.
// Both TopDown and BottomUp
void SIScheduleDAGMI::topologicalSort() {
  unsigned DAGSize = SUnits.size();
  std::vector<SUnit*> WorkList;

  DEBUG(dbgs() << "Topological Sort\n");
  WorkList.reserve(DAGSize);

  TopDownIndex2Node.resize(DAGSize);
  TopDownNode2Index.resize(DAGSize);
  BottomUpIndex2Node.resize(DAGSize);
  BottomUpNode2Index.resize(DAGSize);

  if (&ExitSU)
    WorkList.push_back(&ExitSU);
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
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
    for (SUnit::const_pred_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      SUnit *SU = I->getSUnit();
      if (SU->NodeNum < DAGSize && !--TopDownNode2Index[SU->NodeNum])
        WorkList.push_back(SU);
    }
  }

  WorkList.clear();

  if (&EntrySU)
    WorkList.push_back(&EntrySU);
  for (int i = DAGSize-1; i >= 0; --i) {
    SUnit *SU = &SUnits[i];
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
    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      SUnit *SU = I->getSUnit();
      if (SU->NodeNum < DAGSize && !--BottomUpNode2Index[SU->NodeNum])
        WorkList.push_back(SU);
    }
  }
  #ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::const_pred_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      if (I->getSUnit()->NodeNum >= DAGSize)
        continue;
      assert(TopDownNode2Index[SU->NodeNum] >
             TopDownNode2Index[I->getSUnit()->NodeNum] &&
             "Wrong Top Down topological sorting");
    }
  }
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      if (I->getSUnit()->NodeNum >= DAGSize)
        continue;
      assert(BottomUpNode2Index[SU->NodeNum] >
             BottomUpNode2Index[I->getSUnit()->NodeNum] &&
             "Wrong Bottom Up topological sorting");
    }
  }
#endif
}

// Associate all SUs to Blocks
void SIScheduleDAGMI::createBlocks() {
  unsigned MaxID = 0;
  unsigned MaxHighLatencyID;
  unsigned DAGSize = SUnits.size();
  std::vector<unsigned> Colors;
  std::vector<unsigned> Colors_LatOnly;
  std::vector<unsigned> Colors_FirstPass;
  std::vector<int> RealID;
  std::map<std::set<unsigned>, unsigned> ColorCombinations;

  DEBUG(dbgs() << "Coloring the graph\n");
  Colors.resize(DAGSize, 0);

  // Put all high latency instructions in separate blocks
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    if (SITII->isHighLatencyInstruction(SU->getInstr())) {
      Colors[SU->NodeNum] = ++MaxID;
    }
  }

  Colors_LatOnly = Colors;
  MaxHighLatencyID = MaxID;

  // Traverse TopDown, and give different colors to SUs depending
  // on which combination of High Latencies they depend on.

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[TopDownIndex2Node[i]];
    std::set<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum])
      continue;

    for (SUnit::const_pred_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      SUnit *Pred = I->getSUnit();
      if (I->isWeak() || Pred->NodeNum >= DAGSize)
        continue;
      if (Colors[Pred->NodeNum] > 0)
        SUColors.insert(Colors[Pred->NodeNum]);
    }
    // color 0 by default
    if (SUColors.size() == 0)
      continue;
    // same color than parents
    if (SUColors.size() == 1 &&
        (*SUColors.begin() > MaxHighLatencyID || *SUColors.begin() == 0))
      Colors[SU->NodeNum] = *SUColors.begin();
    else {
      std::map<std::set<unsigned>, unsigned>::iterator Pos =
        ColorCombinations.find(SUColors);
      if (Pos != ColorCombinations.end()) {
          Colors[SU->NodeNum] = Pos->second;
      } else {
        Colors[SU->NodeNum] = ++MaxID;
        ColorCombinations[SUColors] = MaxID;
      }
    }
  }

  ColorCombinations.clear();
  Colors_FirstPass = Colors;
  Colors = Colors_LatOnly;

  // Same than before, but BottomUp.
  // Ignore colors given by the TopDown pass.

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::set<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= MaxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (Colors[Succ->NodeNum] > 0)
        SUColors.insert(Colors[Succ->NodeNum]);
    }
    // keep previous color
    if (SUColors.size() == 0)
      continue;
    // same color than parents
    if (SUColors.size() == 1 &&
        (*SUColors.begin() > MaxHighLatencyID || *SUColors.begin() == 0))
      Colors[SU->NodeNum] = *SUColors.begin();
    else {
      std::map<std::set<unsigned>, unsigned>::iterator Pos =
        ColorCombinations.find(SUColors);
      if (Pos != ColorCombinations.end()) {
          Colors[SU->NodeNum] = Pos->second;
      } else {
        Colors[SU->NodeNum] = ++MaxID;
        ColorCombinations[SUColors] = MaxID;
      }
    }
  }

  ColorCombinations.clear();

  // Every combination of colors given by the two previous pass
  // leads to a new color.
  // Instructions that have the same color are the ones that depend
  // on the same high latency instructions, and that are dependencies
  // for the same high latency instructions

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[TopDownIndex2Node[i]];
    std::set<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= MaxHighLatencyID &&
        Colors[SU->NodeNum] > 0)
      continue;

    SUColors.insert(Colors_FirstPass[SU->NodeNum]);
    SUColors.insert(Colors[SU->NodeNum]);

    // Case the BottomUp pass didn't find any high latency
    // that depend on the SU
    if (SUColors.size() == 1)
      Colors[SU->NodeNum] = *SUColors.begin();
    else {
      std::map<std::set<unsigned>, unsigned>::iterator Pos =
        ColorCombinations.find(SUColors);
      if (Pos != ColorCombinations.end()) {
          Colors[SU->NodeNum] = Pos->second;
      } else {
        Colors[SU->NodeNum] = ++MaxID;
        ColorCombinations[SUColors] = MaxID;
      }
    }
  }

  // 0 for the TopDown pass means it doesn't depend on any
  // high latency instruction.
  // This pass gives these instructions the color of their successors, if only
  // these successors are all of the same color. Such instructions include
  // vgpr/sgpr constant loading, and others.
  // The pass seems to decrease register pressure
  // (quite significantly sometimes)
  // However the high latency instructions (especially the first ones) are
  // less well hidden in general. Some cases get better performance, some get
  // worse performance. It seems worse performance is more often the case.

#if 0
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::set<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    if (Colors_FirstPass[SU->NodeNum] > 0)
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.insert(Colors[Succ->NodeNum]);
    }
    if (SUColors.size() == 0)
      continue;

    if (SUColors.size() == 1)
      Colors[SU->NodeNum] = *SUColors.begin();
  }
#endif

  // lighter pass. Only merges constant loading
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::set<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= MaxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    /* No predecessor: vgpr constant loading */
    /* low latency instruction usually have a predecessor (the address) */
    if (Colors_FirstPass[SU->NodeNum] > 0 ||
        (SU->Preds.size() > 0 &&
         !SITII->isLowLatencyInstruction(SU->getInstr())))
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.insert(Colors[Succ->NodeNum]);
    }
    if (SUColors.size() == 0)
      continue;
    if (SUColors.size() == 1)
      Colors[SU->NodeNum] = *SUColors.begin();
  }

  // Special case: predecessors of high latency instructions that have no
  // other successors are put in the high latency instruction block.
  // This reduces sgpr usage significantly, because the instruction loading
  // the address of the image load will be put in the same block.
  // However since that also means we do not have a lot of instructions
  // (and often none) to hide the low latency, that would benefit from a pass
  // after block scheduling to move before these latencies
  // if sgpr usage after scheduling is low.

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::set<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= MaxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (Colors[Succ->NodeNum] > 0)
        SUColors.insert(Colors[Succ->NodeNum]);
    }
    // keep previous color
    if (SUColors.size() == 0)
      continue;
    if (SUColors.size() == 1 && *SUColors.begin() <= MaxHighLatencyID &&
        *SUColors.begin() > 0)
      Colors[SU->NodeNum] = *SUColors.begin();
  }

  // Other passes could be added, for examples big blocks could be divided
  // into smaller parts, or if a block has some instructions that only increase
  // the register pressure, they could be put in a separate block.

  // Uncomment this if you want to try 1 instruction == 1 block
  // (except for high latencies):
  /*for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[TopDownIndex2Node[i]];

    // High latency instructions
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    Colors[SU->NodeNum] = ++maxID;
  }*/

  // Put SUs of same color into same block
  RealID.resize(MaxID+1, -1);
  Node2Block.resize(DAGSize, -1);
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned Color = Colors[SU->NodeNum];
    if (RealID[Color] == -1) {
      int ID = Blocks.size();
      Blocks.push_back(
        make_unique<SIBlockSchedule>(this, ID,
                                     (Color > 0) &&
                                     (Color <= MaxHighLatencyID)));
      RealID[Color] = ID;
    }
    Blocks[RealID[Color]]->addUnit(SU);
    Node2Block[SU->NodeNum] = RealID[Color];
  }

  // Build dependencies between blocks
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    int SUID = Node2Block[i];
    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (Node2Block[Succ->NodeNum] != SUID)
        Blocks[SUID]->addSucc(Blocks[Node2Block[Succ->NodeNum]].get());
    }
    for (SUnit::const_pred_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      SUnit *Pred = I->getSUnit();
      if (I->isWeak() || Pred->NodeNum >= DAGSize)
        continue;
      if (Node2Block[Pred->NodeNum] != SUID)
        Blocks[SUID]->addPred(Blocks[Node2Block[Pred->NodeNum]].get());
    }
  }

  // free root and leafs of all blocks to enable scheduling inside them
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned Color = Colors[SU->NodeNum];
    Blocks[RealID[Color]]->releaseSuccessors(SU, false);
  }
  DEBUG(dbgs() << "Blocks created:\n\n");
  DEBUG(
    for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
      SIBlockSchedule *Block = Blocks[i].get();
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

void SIScheduleDAGMI::propagateWaitedLatencies() {
  unsigned DAGSize = Blocks.size();

  for (unsigned i = 0, e = (unsigned)SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];

    HasLowLatencyNonWaitedParent[i] = 0;

    for (SUnit::succ_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      SUnit *Pred = I->getSUnit();
      if (Pred->NodeNum >= DAGSize || Pred->isScheduled)
        continue;
      if (SITII->isLowLatencyInstruction(Pred->getInstr()))
        HasLowLatencyNonWaitedParent[i] = 1;
    }
  }
}

void SIScheduleDAGMI::scheduleInsideBlocks() {
  unsigned DAGSize = Blocks.size();
  std::vector<int> WorkList;
  std::vector<int> TopDownIndex2Indice;
  std::vector<int> TopDownIndice2Index;

  DEBUG(dbgs() << "\nScheduling Blocks\n\n");

  // fill some stats to help scheduling (dummy stats for fastSchedule)

  IsLowlatencySU.clear();
  LowLatencyOffset.clear();
  HasLowLatencyNonWaitedParent.clear();

  IsLowlatencySU.resize(SUnits.size());
  LowLatencyOffset.resize(SUnits.size());
  HasLowLatencyNonWaitedParent.resize(SUnits.size());

  // We do schedule a valid scheduling such that a Block corresponds
  // to a range of instructions. That will guarantee that if a result
  // produced inside the Block, but also used inside the Block, while it
  // gets used in another Block, then it will be correctly counted in the
  // Live Output Regs of the Block
  DEBUG(dbgs() << "First phase: Fast scheduling for Reg Liveness\n");
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    Block->fastSchedule();
  }

  WorkList.reserve(DAGSize);

  TopDownIndex2Indice.resize(DAGSize);
  TopDownIndice2Index.resize(DAGSize);

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    unsigned Degree = Block->Succs.size();
    TopDownIndice2Index[i] = Degree;
    if (Degree == 0) {
      WorkList.push_back(i);
    }
  }

  int Id = DAGSize;
  while (!WorkList.empty()) {
    int i = WorkList.back();
    SIBlockSchedule *Block = Blocks[i].get();
    WorkList.pop_back();
    TopDownIndice2Index[i] = --Id;
    TopDownIndex2Indice[Id] = i;
    for (std::vector<SIBlockSchedule*>::iterator I = Block->Preds.begin(),
         E = Block->Preds.end(); I != E; ++I) {
      SIBlockSchedule *Pred = *I;
      if (!--TopDownIndice2Index[Pred->ID])
        WorkList.push_back(Pred->ID);
    }
  }

#ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    for (std::vector<SIBlockSchedule*>::iterator I = Block->Preds.begin(),
         E = Block->Preds.end(); I != E; ++I) {
      SIBlockSchedule *Pred = *I;
      assert(TopDownIndice2Index[i] > TopDownIndice2Index[Pred->ID] &&
      "Wrong Top Down topological sorting");
    }
  }
#endif

  // Do not update CurrentTop
  MachineBasicBlock::iterator CurrentTopFastSched = CurrentTop;
  std::vector<MachineBasicBlock::iterator> PosOld;
  std::vector<MachineBasicBlock::iterator> PosNew;
  PosOld.reserve(SUnits.size());
  PosNew.reserve(SUnits.size());

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    int BlockIndice = TopDownIndex2Indice[i];
    SIBlockSchedule *Block = Blocks[BlockIndice].get();
    std::vector<SUnit*> SUs = Block->getScheduledUnits();

    for (std::vector<SUnit*>::iterator I = SUs.begin(),
         E = SUs.end(); I != E; ++I) {
      MachineInstr *MI = (*I)->getInstr();
      MachineBasicBlock::iterator Pos = MI;
      PosOld.push_back(Pos);
      if (&*CurrentTopFastSched == MI) {
        PosNew.push_back(Pos);
        CurrentTopFastSched = nextIfDebug(++CurrentTopFastSched,
                                          CurrentBottom);
      } else {
        // Update the instruction stream.
        BB->splice(CurrentTopFastSched, BB, MI);

        // Update LiveIntervals
        LIS->handleMove(MI, /*UpdateFlags=*/true);
        PosNew.push_back(CurrentTopFastSched);
      }
    }
  }

  // fill some stats to help scheduling

  IsLowlatencySU.clear();
  LowLatencyOffset.clear();
  HasLowLatencyNonWaitedParent.clear();

  IsLowlatencySU.resize(SUnits.size());
  LowLatencyOffset.resize(SUnits.size());
  HasLowLatencyNonWaitedParent.resize(SUnits.size());

  for (unsigned i = 0, e = (unsigned)SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned BaseLatReg, OffLatReg;
    if (SITII->isLowLatencyInstruction(SU->getInstr())) {
      IsLowlatencySU[i] = 1;
      if (SITII->getMemOpBaseRegImmOfs(SU->getInstr(), BaseLatReg,
                                      OffLatReg, TRI))
        LowLatencyOffset[i] = OffLatReg;
    }
  }

  for (unsigned i = 0, e = (unsigned)SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::succ_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      SUnit *Pred = I->getSUnit();
      if (Pred->NodeNum >= DAGSize)
        continue;
      if (SITII->isLowLatencyInstruction(Pred->getInstr()))
        HasLowLatencyNonWaitedParent[i] = 1;
    }
  }

  // Now we have Block of SUs == Block of MI
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
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
      BB->splice(POld, BB, PNew);

      // Update LiveIntervals
      LIS->handleMove(POld, /*UpdateFlags=*/true);
    }
  }

  // Fill the usage of every output
  // By construction we have that for every Reg input
  // of a block, it is present one time max in the outputs
  // of its Block predecessors. If it is not, it means it Reads
  // a register whose content was already filled at start
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    for (std::set<unsigned>::iterator RegI = Block->getInRegs().begin(),
         E = Block->getInRegs().end(); RegI != E; ++RegI) {
      unsigned Reg = *RegI;
      for (std::vector<SIBlockSchedule*>::iterator I = Block->Preds.begin(),
           E = Block->Preds.end(); I != E; ++I) {
        SIBlockSchedule *Pred = *I;
        std::map<unsigned, unsigned>::iterator RegPos =
          Pred->LiveOutRegsNumUsages.find(Reg);
        if (RegPos != Pred->LiveOutRegsNumUsages.end()) {
          ++Pred->LiveOutRegsNumUsages[Reg];
          break;
        }
      }
    }
  }

  DEBUG(
    for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
      SIBlockSchedule *Block = Blocks[i].get();
      Block->printDebug(true);
    }
  );
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

    for (SUnit::const_pred_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      SUnit *Pred = I->getSUnit();
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
      for (SUnit::const_succ_iterator I = SU->Succs.begin(),
           E = SU->Succs.end(); I != E; ++I) {
        SUnit *Succ = I->getSUnit();
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

// return the vgpr usage corresponding to some virtual registers
template<typename _Iterator> unsigned
SIScheduleDAGMI::VgprCost(_Iterator First, _Iterator End) {
  unsigned cost = 0;
  for (_Iterator RegI = First; RegI != End; ++RegI) {
    unsigned Reg = *RegI;
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    PSetIterator PSetI = MRI.getPressureSets(Reg);
    for (; PSetI.isValid(); ++PSetI) {
      if (*PSetI == VGPRSetID)
        cost += PSetI.getWeight();
    }
  }

  return cost;
}

std::vector<int>
SIScheduleDAGMI::checkRegUsageImpact(std::set<unsigned> &InRegs,
                                     std::set<unsigned> &OutRegs) {
  std::vector<int> DiffSetPressure;
  DiffSetPressure.assign(TRI->getNumRegPressureSets(), 0);

  for (std::set<unsigned>::iterator RegI = InRegs.begin(),
       E = InRegs.end(); RegI != E; ++RegI) {
    unsigned Reg = *RegI;
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (LiveRegsConsumers[Reg] > 1)
      continue;
    PSetIterator PSetI = MRI.getPressureSets(Reg);
    for (; PSetI.isValid(); ++PSetI) {
      DiffSetPressure[*PSetI] -= PSetI.getWeight();
    }
  }

  for (std::set<unsigned>::iterator RegI = OutRegs.begin(),
       E = OutRegs.end(); RegI != E; ++RegI) {
    unsigned Reg = *RegI;
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    PSetIterator PSetI = MRI.getPressureSets(Reg);
    for (; PSetI.isValid(); ++PSetI) {
      DiffSetPressure[*PSetI] += PSetI.getWeight();
    }
  }

  return DiffSetPressure;
}

// We could imagine testing different Block scheduling
// strategies and taking the best one. This set up
// everything need for testing a Block scheduling
// strategy.
void SIScheduleDAGMI::prepareScheduleBlocks() {
  LiveRegs.clear();
  LiveRegsConsumers.clear();
  NumBlockScheduled = 0;
  ReadyBlocks.clear();
  BlockScheduleOrder.clear();
  BlockScheduleOrder.reserve(Blocks.size());

  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    Block->NumPredsLeft = Block->Preds.size();
    Block->NumSuccsLeft = Block->Succs.size();
    Block->LastPosHighLatencyParentScheduled = 0;
  }

  std::set<unsigned> InRegs (RPTracker.getPressure().LiveInRegs.begin(),
                             RPTracker.getPressure().LiveInRegs.end());
  addLiveRegs(InRegs);

  // fill LiveRegsConsumers for regs that were already
  // defined before scheduling
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    for (std::set<unsigned>::iterator RegI = Block->getInRegs().begin(),
         E = Block->getInRegs().end(); RegI != E; ++RegI) {
      unsigned Reg = *RegI;
      bool Found = false;
      for (std::vector<SIBlockSchedule*>::iterator I = Block->Preds.begin(),
           E = Block->Preds.end(); I != E; ++I) {
        SIBlockSchedule *Pred = *I;
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
    SIBlockSchedule *Block = Blocks[i].get();
    if (Block->NumPredsLeft == 0) {
      ReadyBlocks.push_back(Block);
    }
  }
}

// Tracking of currently alive register to determine VGPR Usage

void SIScheduleDAGMI::addLiveRegs(std::set<unsigned> &Regs) {
  for (std::set<unsigned>::iterator RegI = Regs.begin(),
       E = Regs.end(); RegI != E; ++RegI) {
    unsigned Reg = *RegI;
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    // if not already in the live set, then add it
    (void) LiveRegs.insert(Reg);
  }
}

void SIScheduleDAGMI::decreaseLiveRegs(SIBlockSchedule *Block,
                                       std::set<unsigned> &Regs) {
  for (std::set<unsigned>::iterator RegI = Regs.begin(),
       E = Regs.end(); RegI != E; ++RegI) {
    unsigned Reg = *RegI;
    // For now only track virtual registers
    std::set<unsigned>::iterator Pos = LiveRegs.find(Reg);
    assert (Pos != LiveRegs.end()); // Reg must be live
    assert (LiveRegsConsumers.find(Reg) != LiveRegsConsumers.end());
    assert (LiveRegsConsumers[Reg] >= 1);
    --LiveRegsConsumers[Reg];
    if (LiveRegsConsumers[Reg] == 0)
      LiveRegs.erase(Pos);
  }
}

void SIScheduleDAGMI::releaseBlockSuccs(SIBlockSchedule *Parent) {
  for (std::vector<SIBlockSchedule*>::iterator I = Parent->Succs.begin(),
       E = Parent->Succs.end(); I != E; ++I) {
    SIBlockSchedule *Block = *I;
    --Block->NumPredsLeft;
    if (Block->NumPredsLeft == 0) {
      ReadyBlocks.push_back(Block);
    }
    if (Parent->isHighLatencyBlock())
      Block->LastPosHighLatencyParentScheduled = NumBlockScheduled;
  }
}

void SIScheduleDAGMI::blockScheduled(SIBlockSchedule *Block) {
  decreaseLiveRegs(Block, Block->getInRegs());
  addLiveRegs(Block->getOutRegs());
  releaseBlockSuccs(Block);
  for (std::map<unsigned, unsigned>::iterator RegI =
       Block->LiveOutRegsNumUsages.begin(),
       E = Block->LiveOutRegsNumUsages.end(); RegI != E; ++RegI) {
    std::pair<unsigned, unsigned> RegP = *RegI;
    if (LiveRegsConsumers.find(RegP.first) == LiveRegsConsumers.end())
      LiveRegsConsumers[RegP.first] = RegP.second;
    else
      LiveRegsConsumers[RegP.first] += RegP.second;
  }
  ++NumBlockScheduled;
}

unsigned SIScheduleDAGMI::getWaveFrontsForUsage(unsigned SGPRsUsed,
                                                unsigned VGPRsUsed) {
  unsigned i;

  for (i = 9; i > 0; --i) {
    if (SGPRsForWaveFronts[i] >= SGPRsUsed &&
        VGPRsForWaveFronts[i] >= VGPRsUsed)
      break;
  }

  return i+1;
}

void SIScheduleDAGMI::tryCandidate(SIBlockSchedCandidate &Cand,
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

SIBlockSchedule *SIScheduleDAGMI::pickBlock() {
  SIBlockSchedCandidate Cand;
  std::vector<SIBlockSchedule*>::iterator Best;
  SIBlockSchedule *Block;
  if (ReadyBlocks.size() == 0)
    return nullptr;

  VregCurrentUsage = VgprCost(LiveRegs.begin(), LiveRegs.end());
  DEBUG(
    dbgs() << "Picking New Blocks\n";
    dbgs() << "Available: ";
    for (std::vector<SIBlockSchedule*>::iterator I = ReadyBlocks.begin(),
         E = ReadyBlocks.end(); I != E; ++I)
      dbgs() << (*I)->ID << " ";
    dbgs() << "\nCurrent Live:\n";
    for (std::set<unsigned>::iterator I = LiveRegs.begin(),
         E = LiveRegs.end(); I != E; ++I)
      dbgs() << PrintVRegOrUnit(*I, TRI) << " ";
    dbgs() << "\n";
    dbgs() << "Current VGPRs: " << VregCurrentUsage << "\n";
  );

  for (std::vector<SIBlockSchedule*>::iterator I = ReadyBlocks.begin(),
       E = ReadyBlocks.end(); I != E; ++I) {
    SIBlockSchedCandidate TryCand;
    TryCand.Block = *I;
    TryCand.IsHighLatency = TryCand.Block->isHighLatencyBlock();
    TryCand.VGPRUsageDiff =
      checkRegUsageImpact((*I)->getInRegs(), (*I)->getOutRegs())[VGPRSetID];
    TryCand.NumHighLatencySuccessors = TryCand.Block->NumHighLatencySuccessors;
    TryCand.LastPosHighLatParentScheduled =
      TryCand.Block->LastPosHighLatencyParentScheduled;
    tryCandidate(Cand, TryCand);
    if (TryCand.Reason != NoCand) {
      Cand.setBest(TryCand);
      Best = I;
      DEBUG(dbgs() << "Best Current Choice: " << Cand.Block->ID << " "
                   << getReasonStr(Cand.Reason) << "\n");
    }
  }

  DEBUG(
    dbgs() << "Picking: " << Cand.Block->ID << "\n";
    if (Cand.IsHighLatency)
      dbgs() << "Is a block with high latency instruction\n";
    dbgs() << "Position of last high latency dependency: "
           << Cand.LastPosHighLatParentScheduled << "\n";
    dbgs() << "VGPRUsageDiff: " << Cand.VGPRUsageDiff << "\n";
    dbgs() << "\n";
  );

  Block = Cand.Block;
  ReadyBlocks.erase(Best);
  return Block;
}

void SIScheduleDAGMI::scheduleBlocks()
{
  prepareScheduleBlocks();

  // We do a TopDown algorithm trying to hide high latencies as much as
  // possible.
  // Other strategies could be imagined, for example trying to have register
  // usage very low to have high wavefront count, even if high latency
  // instructions wouldn't be as well hidden.
  // Several strategies could be tried, and the best result picked.
  while (SIBlockSchedule *Block = pickBlock()) {
    BlockScheduleOrder.push_back(Block->ID);
    blockScheduled(Block);
  }

  DEBUG(
    dbgs() << "Block Order:";
    for (std::vector<unsigned>::iterator I = BlockScheduleOrder.begin(),
         E = BlockScheduleOrder.end(); I != E; ++I) {
      dbgs() << " " << *I;
    }
  );
  // We could imagine here an algorithm to optimize even more the Block order.
  // Has been tried to put here an algorithm to optimize the score:
  // '(mean number of instructions hiding high latencies +
  //   min number of instructions hiding high latencies)
  // * number of wavefronts
  // The score was optimized by iterating over all blocks, and testing for each
  // of them if there was a position that would increase the score.
  // However while computation time was increased a lot,
  // the performance wasn't improved.
}


void SIScheduleDAGMI::schedule()
{
  prepareSchedule();

  scheduleBlocks();

  ScheduledSUnits.clear();
  ScheduledSUnitsInv.resize(SUnits.size());

  for (unsigned b = 0; b < Blocks.size(); ++b) {
    SIBlockSchedule *Block = Blocks[BlockScheduleOrder[b]].get();
    std::vector<SUnit*> SUs = Block->getScheduledUnits();

    for (std::vector<SUnit*>::iterator I = SUs.begin(),
         E = SUs.end(); I != E; ++I) {
      ScheduledSUnitsInv[(*I)->NodeNum] = ScheduledSUnits.size();
      ScheduledSUnits.push_back((*I)->NodeNum);
    }
  }

  moveLowLatencies();

  // tell the outside world about the result of the scheduling

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

  Blocks.clear();
  TopDownIndex2Node.clear();
  TopDownNode2Index.clear();
  BottomUpIndex2Node.clear();
  BottomUpNode2Index.clear();
  Node2Block.clear();
}
