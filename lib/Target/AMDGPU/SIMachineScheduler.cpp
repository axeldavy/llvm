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
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "misched"

/// This scheduler implements a different scheduling algorithm than GenericScheduler.
///
/// There are several specific architecture behaviours that can't make it to GenericScheduler:
/// . There are different level of pressure for the registers, and the pressure level is the max
/// of the pressure levels for all the register sets.
/// . The more the register pressure, the more the load latencies.
/// . When accessing the result of an SGPR load instruction, you have to wait for all the
/// SGPR load instructions before your current instruction to finish.
/// . When accessing the result of an VGPR load instruction, you have to wait for all the
/// VGPR load instructions previous to the VGPR load instruction you are interested in to finish.
/// . The diversity of the instruction number of inputs/outputs, latency, etc make the generic scheduler
/// generate worse code for some shaders when it tries to reduce reg pressure. The scheduler is 
/// extremely sensitive to the reg pressure levels, and the load latency. Also some other bad behaviours
/// are generated when it doesn't care about reg pressure (for example at the beginning of the shader), like
/// loading at the beginning of the shader a constant in register you won't need until the end of the shader.
///
/// This scheduler implemented here algorithm is:
/// . Build clever mini blocks of instructions
/// . Schedule inside every mini blocks
/// . Schedule the whole by placing the blocks in a good order
///
/// In practice currently:
/// . Do topological sort for top-down and bottom-up
/// . Use the sort to build the blocks. A block is:
///     . An instruction with high latency alone
///     . A set of instructions having the same dependencies on the high latency instructions in the graph for both bottom up and top down
///        + instructions like loading constants needed for these instructions
/// Inside a block:
/// Put all low latency instructions (loading sgpr constant from memory for example) at the beginning of the block
/// Schedule (top down) instructions of the bloc not needing the outputs of the low latency instructions
/// Schedule (bottom-up) the remaining instructions. The reason is that bottom-up seems better for register usage, and top-down better for latency.
///
/// When scheduling all the blocks together:
/// Top down approach to reduce latency
///
/// The advantage of scheduling by blocks is that the blocks are rather self contained (constant loading, etc),
/// thus reducing register usage in the entire scheduling region, and it becomes easier to reduce latency: after a
/// instruction with high latency, we just put an entire block (or several ones) not depending on this instruction to hide latency.
///
/// Once everything is scheduled, a finalization phase can be implemented, that would not decrease the wavefront count, and that would
/// move low latency instructions outside of their block, to merge them to other low latency instructions, or just put them further away
/// of the instructions needing them.
///
/// Improvements to the algorithm can be made by finding better ways of building and scheduling the blocks.
/// For example the blocks could be cut into several independant parts when too big.


// common code //

#ifndef NDEBUG

const char *getReasonStr(SIScheduleCandReason Reason)  {
  switch (Reason) {
  case NoCand:         return "NOCAND    ";
  case RegUsage:       return "REGUSAGE  ";
  case SIWaveFronts:   return "WAVEFRONTS";
  case Latency:        return "LATENCY   ";
  case Weak:           return "WEAK      ";
  case ReadySuccessor: return "READYSUCC ";
  case Height:         return "HEIGHT    ";
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
  // Note: the reason we do care about low latency instructions here, whereas
  // we have special treatment for them to put them first is in the case that
  // we have low latency instructions that need other low latency instructions
  // results.
  // We will get low latency instructions - independant instructions - eventual low latency instructions - etc
  if (tryGreater(TryCand.isLowLatency, Cand.isLowLatency, TryCand, Cand, Depth))
    return;

  if (tryLess(TryCand.VGPRUsage, Cand.VGPRUsage, TryCand, Cand, RegUsage))
    return;

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum < Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
  }
}

void SIBlockSchedule::tryCandidateBottomUp(SISchedCandidate &Cand,
                                           SISchedCandidate &TryCand) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return;
  }

  // Schedule low latency instructions as top as possible
  if (tryLess(TryCand.isLowLatency, Cand.isLowLatency, TryCand, Cand, Depth))
    return;

  if (tryLess(TryCand.VGPRUsage, Cand.VGPRUsage, TryCand, Cand, RegUsage))
    return;

  // if everything would increase register usage, then encourage depth traversal
  //if ( TODO )
  if (tryLess(TryCand.SU->getDepth(), Cand.SU->getDepth(), TryCand, Cand, Depth))
    return;

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum > Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
  }
}

SUnit* SIBlockSchedule::pickNode(bool TopOnly, bool &isTop) {
  SISchedCandidate TopCand;
  SISchedCandidate BotCand;

  for (std::vector<SUnit*>::iterator I = TopReadySUs.begin(), E = TopReadySUs.end(); I != E; ++I) {
    SISchedCandidate TryCand;
    std::vector<unsigned> pressure = TopRPTracker.getRegSetPressureAtPos();
    std::vector<unsigned> MaxPressure = TopRPTracker.getRegSetPressureAtPos();
    TopRPTracker.getDownwardPressure((*I)->getInstr(), pressure, MaxPressure);
    TryCand.SU = *I;
    TryCand.VGPRUsage = pressure[26];
    TryCand.isLowLatency = DAG->SITII->isLowLatencyInstruction(TryCand.SU->getInstr());
    tryCandidateTopDown(TopCand, TryCand);
    if (TryCand.Reason != NoCand) {
      TopCand.setBest(TryCand);
      //DEBUG(traceCandidate(Cand));
    }
  }

  if (TopOnly) {
    isTop = true;
    return TopCand.SU;
  }

  for (std::vector<SUnit*>::iterator I = BottomReadySUs.begin(), E = BottomReadySUs.end(); I != E; ++I) {
    SISchedCandidate TryCand;
    std::vector<unsigned> pressure = BotRPTracker.getRegSetPressureAtPos();
    std::vector<unsigned> MaxPressure = BotRPTracker.getRegSetPressureAtPos();
    BotRPTracker.getUpwardPressure((*I)->getInstr(), pressure, MaxPressure);
    TryCand.SU = *I;
    TryCand.VGPRUsage = pressure[26];
    TryCand.isLowLatency = DAG->SITII->isLowLatencyInstruction(TryCand.SU->getInstr());
    tryCandidateBottomUp(BotCand, TryCand);
    if (TryCand.Reason != NoCand) {
      BotCand.setBest(TryCand);
    }
  }
  // TODO: if one reduce reg usage, and not the other, take it, else take bottom
  isTop = false;
  return BotCand.SU;
}


// Schedule something valid.
// goal is just to be able to compute LiveIns and LiveOuts
void SIBlockSchedule::fastSchedule() {
  TopReadySUs.clear();
  BottomReadySUs.clear();

  for (unsigned i = 0, e = (unsigned)SUnits.size(); i != e; ++i) {
    SUnit *SU = SUnits[i];

    if (!SU->NumSuccsLeft)
      BottomReadySUs.push_back(SU);

    if (!SU->NumPredsLeft)
      TopReadySUs.push_back(SU);
  }

  while (!TopReadySUs.empty()) {
    SUnit *SU = TopReadySUs[0];
    ScheduledSUnits.push_back(SU);
    NodeScheduled(SU);
  }

  scheduled = true;
}

static bool findDefBetween(unsigned Reg,
                           SlotIndex first, SlotIndex last,
                           const MachineRegisterInfo *MRI,
                           const LiveIntervals *LIS) {
  for (MachineRegisterInfo::def_instr_iterator
       UI = MRI->def_instr_begin(Reg),
       UE = MRI->def_instr_end(); UI != UE; ++UI) {
      const MachineInstr* MI = &*UI;
      if (MI->isDebugValue())
        continue;
      SlotIndex InstSlot = LIS->getInstructionIndex(MI).getRegSlot();
      if (InstSlot >= first && InstSlot <= last)
        return true;
  }
  return false;
}

void SIBlockSchedule::initRegPressure(MachineBasicBlock::iterator BeginBlock, MachineBasicBlock::iterator EndBlock) {
  LiveIntervals *LIS = DAG->getLIS();
  MachineRegisterInfo *MRI = DAG->getMRI();
  DAG->initRPTracker(TopRPTracker);
  DAG->initRPTracker(BotRPTracker);
  DAG->initRPTracker(RPTracker);

  // Goes though all SU. RPTracker captures what had to be alive for the SUs
  // to execute, and what is still alive at the end
  for (unsigned i = 0, e = (unsigned)ScheduledSUnits.size(); i != e; ++i) {
    SUnit *SU = ScheduledSUnits[i];
    RPTracker.setPos(SU->getInstr());
    RPTracker.advance();
  }

  // Close the RPTracker to finalize live ins/outs.
  RPTracker.closeRegion();

  // Initialize the live ins and live outs.
  TopRPTracker.addLiveRegs(RPTracker.getPressure().LiveInRegs);
  BotRPTracker.addLiveRegs(RPTracker.getPressure().LiveOutRegs);

  // Do not Track Physical Registers, because it messes up
  for (unsigned i = 0, e = RPTracker.getPressure().LiveInRegs.size(); i != e; ++i) {
    unsigned Reg = RPTracker.getPressure().LiveInRegs[i];
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      LiveInRegs.push_back(Reg);
  }
  LiveOutRegs.clear();
  // There is several possibilities to distinguish:
  // 1) Reg is not input to any instruction in the block, but is output of one
  // 2) 1) + read in the block and not needed after it
  // 3) 1) + read in the block but needed in another block
  // 4) Reg is input of an instruction but another block will read it too
  // 5) Reg is input of an instruction and then rewritten in the block. result is not read in the block (implies used in another block)
  // 6) Reg is input of an instruction and then rewritten in the block. result is read in the block and not needed in another block
  // 7) Reg is input of an instruction and then rewritten in the block. result is read in the block but also needed in another block
  // LiveInRegs will contains all the regs in situation 4, 5, 6, 7
  // We want LiveOutRegs to contain only Regs whose content will be read after in another block, and whose content was written in the
  // current block, that is we want it to get 1, 3, 5, 7
  // Since we made MI of a block to be packed all together before scheduling, then the LiveIntervals
  // were correct, and the RPTracker was able to correctly handle 5 vs 6, 2 vs 3. (Note: This is not sufficient for RPTracker to not do mistakes for case 4)
  // The RPTracker's LiveOutRegs has 1, 3, (some correct or incorrect)4, 5, 7
  // Comparing to LiveInRegs is not sufficient to differenciate 4 vs 5, 7
  // The following code removes the case 4
  for (unsigned i = 0, e = RPTracker.getPressure().LiveOutRegs.size(); i != e; ++i) {
    unsigned Reg = RPTracker.getPressure().LiveOutRegs[i];
    if (findDefBetween(Reg, LIS->getInstructionIndex(BeginBlock).getRegSlot(),
                       LIS->getInstructionIndex(EndBlock).getRegSlot(), MRI, LIS) &&
        TargetRegisterInfo::isVirtualRegister(Reg))
      LiveOutRegs.push_back(Reg);
  }
  LiveOutRegsNumUsages.clear();
  LiveOutRegsNumUsages.resize(LiveOutRegs.size(), 0);

  LiveInPressure = TopPressure.MaxSetPressure;
  LiveOutPressure = BotPressure.MaxSetPressure;

  // This converts currently live regs into live ins/outs.
  TopRPTracker.closeTop();
  BotRPTracker.closeBottom();
}

void SIBlockSchedule::schedule(MachineBasicBlock::iterator BeginBlock, MachineBasicBlock::iterator EndBlock) {
  std::vector<SUnit*> LowLatencyReadySUs;
  std::vector<SUnit*> ScheduledSUnitsBottom;

  if (!scheduled)
    fastSchedule();

  // PreScheduling phase to discover LiveIn and LiveOut
  initRegPressure(BeginBlock, EndBlock);
  undoSchedule();

  // Schedule for real now

  TopReadySUs.clear();
  BottomReadySUs.clear();

  for (unsigned i = 0, e = (unsigned)SUnits.size(); i != e; ++i) {
    SUnit *SU = SUnits[i];

    if (!SU->NumSuccsLeft)
      BottomReadySUs.push_back(SU);

    if (SU->NumPredsLeft)
      continue;

    if (DAG->SITII->isLowLatencyInstruction(SU->getInstr()) && SU->NumSuccsLeft)
      LowLatencyReadySUs.push_back(SU);
    else
      TopReadySUs.push_back(SU);
  }

  // TODO: choose best order for low latency SU
  for (unsigned i = 0, e = (unsigned)LowLatencyReadySUs.size(); i != e; ++i) {
    SUnit *SU = LowLatencyReadySUs[i];
    ScheduledSUnits.push_back(SU);
    TopRPTracker.setPos(SU->getInstr());
    TopRPTracker.advance();
  }

  // Schedule Top down what can be scheduled
  if (LowLatencyReadySUs.size()) {
    while (!TopReadySUs.empty()) {
      bool isTop;
      SUnit *SU = pickNode(true, isTop);
      assert(isTop);
      ScheduledSUnits.push_back(SU);
      TopRPTracker.setPos(SU->getInstr());
      TopRPTracker.advance();
      NodeScheduled(SU);
    }
  }

  // Release low latency successors only now
  for (unsigned i = 0, e = (unsigned)LowLatencyReadySUs.size(); i != e; ++i) {
    SUnit *SU = LowLatencyReadySUs[i];
    releaseSuccessors(SU, true);
    SU->isScheduled = true;
  }

  // Schedule the remaining Bottom Up (mainly)
  while (!TopReadySUs.empty()) {
    bool isTop;
    SUnit *SU = pickNode(false, isTop);
    if (isTop) {
      ScheduledSUnits.push_back(SU);
      TopRPTracker.setPos(SU->getInstr());
      TopRPTracker.advance();
    } else {
      ScheduledSUnitsBottom.push_back(SU);
      MachineBasicBlock::const_iterator Pos = SU->getInstr();
      Pos ++;
      BotRPTracker.setPos(Pos);
      BotRPTracker.recede();
    }
    NodeScheduled(SU);
  }

  for (unsigned i = 0, e = (unsigned)ScheduledSUnitsBottom.size(); i != e; ++i) {
    SUnit *SU = ScheduledSUnitsBottom.back();
    ScheduledSUnits.push_back(SU);
    TopRPTracker.setPos(SU->getInstr());
    TopRPTracker.advance();
    ScheduledSUnitsBottom.pop_back();
  }

  InternalPressure = TopPressure.MaxSetPressure;

  // Check everything is right
#ifndef NDEBUG
  assert(SUnits.size() == ScheduledSUnits.size());
  assert(TopReadySUs.empty() && BottomReadySUs.empty());
  for (unsigned i = 0, e = (unsigned)ScheduledSUnits.size(); i != e; ++i) {
    SUnit *SU = SUnits[i];
    assert (SU->isScheduled);
    assert (SU->NumPredsLeft == 0);
    assert (SU->NumSuccsLeft == 0);
  }
#endif

  scheduled = true;
}

void SIBlockSchedule::undoSchedule() {
  for (unsigned i = 0, e = (unsigned)ScheduledSUnits.size(); i != e; ++i) {
    SUnit *SU = SUnits[i];
    SU->isScheduled = false;
    for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      undoReleaseSucc(SU, &*I, true);
    }
  }
  ScheduledSUnits.clear();
  scheduled = false;
}

void SIBlockSchedule::undoReleaseSucc(SUnit *SU, SDep *SuccEdge, bool InOrOutBlock) {
  SUnit *SuccSU = SuccEdge->getSUnit();

  if (DAG->isSUInBlock(SuccSU, ID) != InOrOutBlock)
    return;

  if (SuccEdge->isWeak()) {
    ++SuccSU->WeakPredsLeft;
    ++SU->WeakSuccsLeft;
    return;
  }
  ++SuccSU->NumPredsLeft;
  ++SU->NumSuccsLeft;
}

void SIBlockSchedule::releaseSucc(SUnit *SU, SDep *SuccEdge, bool InOrOutBlock) {
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

void SIBlockSchedule::releasePred(SUnit *SU, SDep *PredEdge, bool InOrOutBlock) {
  SUnit *PredSU = PredEdge->getSUnit();

  if (DAG->isSUInBlock(PredSU, ID) != InOrOutBlock)
    return;

  if (PredEdge->isWeak()) {
    --PredSU->WeakSuccsLeft;
    return;
  }
#ifndef NDEBUG
  if (PredSU->NumSuccsLeft == 0) {
    dbgs() << "*** Scheduling failed! ***\n";
    PredSU->dump(DAG);
    dbgs() << " has been released too many times!\n";
    llvm_unreachable(nullptr);
  }
#endif

  --PredSU->NumSuccsLeft;
  if (PredSU->NumSuccsLeft == 0 && InOrOutBlock && !PredSU->isScheduled)
    BottomReadySUs.push_back(PredSU);
}

/// Release Predecessors of the SU that are in the block  or not
void SIBlockSchedule::releasePredecessors(SUnit *SU, bool InOrOutBlock) {
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    releasePred(SU, &*I, InOrOutBlock);
  }
}

void SIBlockSchedule::NodeScheduled(SUnit *SU) {
  /* Is in TopReadySUs */
  if (!SU->NumPredsLeft) {
    std::vector<SUnit*>::iterator I = std::find(TopReadySUs.begin(), TopReadySUs.end(), SU);
    if (I == TopReadySUs.end()) {
      dbgs() << "Data Structure Bug in SI Scheduler\n";
      llvm_unreachable(nullptr);
    }
    TopReadySUs.erase(I);
  }
  /* Is in BottomReadySUs */
  if (!SU->NumSuccsLeft) {
    std::vector<SUnit*>::iterator I = std::find(BottomReadySUs.begin(), BottomReadySUs.end(), SU);
    if (I == BottomReadySUs.end()) {
      dbgs() << "Data Structure Bug in SI Scheduler\n";
      llvm_unreachable(nullptr);
    }
    BottomReadySUs.erase(I);
  }
  releaseSuccessors(SU, true);
  releasePredecessors(SU, true);
  SU->isScheduled = true;
}

void SIBlockSchedule::addPred(SIBlockSchedule *pred) {
  unsigned predID = pred->ID;

  // check if not already predecessor
  for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
    if (Preds[i]->ID == predID)
      return;
  }
  Preds.push_back(pred);
  NumPredsLeft++;
}

void SIBlockSchedule::addSucc(SIBlockSchedule *succ) {
  unsigned succID = succ->ID;

  // check if not already predecessor
  for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
    if (Succs[i]->ID == succID)
      return;
  }
  Succs.push_back(succ);
  NumSuccsLeft++;
}

#ifndef NDEBUG
void SIBlockSchedule::printDebug(bool full) {
  dbgs() << "Block (" << ID << ")\n";
  if (!full)
    return;

  dbgs() << "\nContains High Latency Instruction: " << highLatencyBlock << "\n";
  dbgs() << "\nDepends On:\n";
  for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
    SIBlockSchedule *Pred = Preds[i];
    Pred->printDebug(false);
  }

  dbgs() << "\nSuccessors:\n";
  for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
    SIBlockSchedule *Succ = Succs[i];
    Succ->printDebug(false);
  }

  if (scheduled) {
    dbgs() << "LiveInPressure " << LiveInPressure[12] << " "<< LiveInPressure[26] << "\n";
    dbgs() << "LiveOutPressure " << LiveOutPressure[12] << " "<< LiveOutPressure[26] << "\n\n";
    dbgs() << "LiveIns:\n";
    for (unsigned i = 0, e = LiveInRegs.size(); i != e; ++i) {
      unsigned Reg = LiveInRegs[i];
      dbgs() << PrintVRegOrUnit(Reg, DAG->getTRI()) << " ";
    }
    dbgs() << "\nLiveOuts (N usage):\n";
    for (unsigned i = 0, e = LiveOutRegs.size(); i != e; ++i) {
      unsigned Reg = LiveOutRegs[i];
      dbgs() << PrintVRegOrUnit(Reg, DAG->getTRI()) << " (" << LiveOutRegsNumUsages[i] <<"), ";
    }
  }

  dbgs() << "\nInstructions:\n";
  if (!scheduled) {
    for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
      SUnit *SU = SUnits[i];
      SU->dump(DAG);
    }
  } else {
    for (unsigned i = 0, e = ScheduledSUnits.size(); i != e; ++i) {
      SUnit *SU = ScheduledSUnits[i];
      SU->dump(DAG);
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

  for (i=0; i<10; i++) {
    SGPRsForWaveFronts[i] = SITRI->getNumSGPRsAllowed(ST.getGeneration(), i+1);
    VGPRsForWaveFronts[i] = SITRI->getNumVGPRsAllowed(i+1);
  }
  VGPRSetID = 26; // TODO: determine dynamically
  SGPRSetID = 12;
}

SIScheduleDAGMI::~SIScheduleDAGMI() {
}

void SIScheduleDAGMI::prepareSchedule() {
  DEBUG(dbgs() << "Preparing Scheduling\n");
  buildDAGWithRegPressure();
  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
        SUnits[su].dumpAll(this));
  topologicalSort();
  SmallVector<SUnit*, 8> TopRoots, BotRoots;
  findRootsAndBiasEdges(TopRoots, BotRoots);
  SchedImpl->initialize(this);
  initQueues(TopRoots, BotRoots);
  createBlocks();
  scheduleInsideBlocks();
}

// see scheduleDAG
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
      Id--;
      TopDownNode2Index[SU->NodeNum] = Id;
      TopDownIndex2Node[Id] = SU->NodeNum;
    }
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
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
      Id--;
      BottomUpNode2Index[SU->NodeNum] = Id;
      BottomUpIndex2Node[Id] = SU->NodeNum;
    }
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *SU = I->getSUnit();
      if (SU->NodeNum < DAGSize && !--BottomUpNode2Index[SU->NodeNum])
        WorkList.push_back(SU);
    }
  }
  #ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (I->getSUnit()->NodeNum >= DAGSize)
        continue;
      assert(TopDownNode2Index[SU->NodeNum] > TopDownNode2Index[I->getSUnit()->NodeNum] &&
      "Wrong Top Down topological sorting");
    }
  }
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      if (I->getSUnit()->NodeNum >= DAGSize)
        continue;
      assert(BottomUpNode2Index[SU->NodeNum] > BottomUpNode2Index[I->getSUnit()->NodeNum] &&
      "Wrong Bottom Up topological sorting");
    }
  }
#endif
}

void SIScheduleDAGMI::createBlocks() {
  unsigned maxID = 0;
  unsigned maxHighLatencyID;
  unsigned DAGSize = SUnits.size();
  std::vector<unsigned> Colors;
  std::vector<unsigned> Colors_LatOnly;
  std::vector<unsigned> Colors_FirstPass;
  std::vector<int> RealID;
  std::vector<std::vector<unsigned>> ColorCombinations;
  std::vector<unsigned> ResultCombination;

  DEBUG(dbgs() << "Coloring the graph\n");
  Colors.resize(DAGSize, 0);

  // Put all high latency instructions in separate blocks
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    if (SITII->isHighLatencyInstruction(SU->getInstr())) {
      maxID++;
      Colors[SU->NodeNum] = maxID;
    }
  }

  Colors_LatOnly = Colors;
  maxHighLatencyID = maxID;

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[TopDownIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum])
      continue;

    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *Pred = I->getSUnit();
      if (I->isWeak() || Pred->NodeNum >= DAGSize)
        continue;
      if (Colors[Pred->NodeNum] > 0)
        SUColors.push_back(Colors[Pred->NodeNum]);
    }
    // color 0 by default
    if (SUColors.size() == 0)
      continue;
    std::sort(SUColors.begin(), SUColors.end()); 
    SUColors.erase(std::unique(SUColors.begin(), SUColors.end()), SUColors.end());
    // same color than parents
    if (SUColors.size() == 1 && (SUColors[0] > maxHighLatencyID || SUColors[0] == 0))
      Colors[SU->NodeNum] = SUColors[0];
    else {
      bool found = false;
      // Check if already given combination, else attribute new color
      for (unsigned c = 0; c != ResultCombination.size(); ++c) {
        if (ColorCombinations[c] == SUColors) {
          Colors[SU->NodeNum] = ResultCombination[c];
          found = true;
          break;
        }
      }
      if (!found) {
        maxID++;
        Colors[SU->NodeNum] = maxID;
        ColorCombinations.push_back(SUColors);
        ResultCombination.push_back(maxID);
      }
    }
  }

  ColorCombinations.clear();
  ResultCombination.clear();
  Colors_FirstPass = Colors;
  Colors = Colors_LatOnly;

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (Colors[Succ->NodeNum] > 0)
        SUColors.push_back(Colors[Succ->NodeNum]);
    }
    // keep previous color
    if (SUColors.size() == 0)
      continue;
    std::sort(SUColors.begin(), SUColors.end()); 
    SUColors.erase(std::unique(SUColors.begin(), SUColors.end()), SUColors.end());
    if (SUColors.size() == 1 && (SUColors[0] > maxHighLatencyID || SUColors[0] == 0))
      Colors[SU->NodeNum] = SUColors[0];
    else {
      bool found = false;
      // Check if already given combination, else attribute new color
      for (unsigned c = 0; c != ResultCombination.size(); ++c) {
        if (ColorCombinations[c] == SUColors) {
          Colors[SU->NodeNum] = ResultCombination[c];
          found = true;
          break;
        }
      }
      if (!found) {
        maxID++;
        Colors[SU->NodeNum] = maxID;
        ColorCombinations.push_back(SUColors);
        ResultCombination.push_back(maxID);
      }
     }
  }

  ColorCombinations.clear();
  ResultCombination.clear();

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[TopDownIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    SUColors.push_back(Colors_FirstPass[SU->NodeNum]);
    SUColors.push_back(Colors[SU->NodeNum]);
    std::sort(SUColors.begin(), SUColors.end()); 
    SUColors.erase(std::unique(SUColors.begin(), SUColors.end()), SUColors.end());
    if (SUColors.size() == 1)
      Colors[SU->NodeNum] = SUColors[0];
    else {
      bool found = false;
      // Check if already given combination, else attribute new color
      for (unsigned c = 0; c != ResultCombination.size(); ++c) {
        if (ColorCombinations[c] == SUColors) {
          Colors[SU->NodeNum] = ResultCombination[c];
          found = true;
          break;
        }
      }
      if (!found) {
        maxID++;
        Colors[SU->NodeNum] = maxID;
        ColorCombinations.push_back(SUColors);
        ResultCombination.push_back(maxID);
      }
    }
  }

  // 0 at top down means means it doesn't require anything with latency to be executed.
  // This pass gives these instructions the color of their successors,
  // if they are all of the same color. That means vgpr constant loading, etc
  // will be in the bloc of their user.
  // PB: pass not good for hiding latencies for first instruction. Reduces perf

  /*for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    if (Colors_FirstPass[SU->NodeNum] > 0)
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.push_back(Colors[Succ->NodeNum]);
    }
    if (SUColors.size() == 0)
      continue;
    std::sort(SUColors.begin(), SUColors.end()); 
    SUColors.erase(std::unique(SUColors.begin(), SUColors.end()), SUColors.end());
    if (SUColors.size() == 1)
      Colors[SU->NodeNum] = SUColors[0];
  }*/

  // lighter pass. Only merges constant loading
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    if (Colors_FirstPass[SU->NodeNum] > 0 || SU->Preds.size() > 0)
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.push_back(Colors[Succ->NodeNum]);
    }
    if (SUColors.size() == 0)
      continue;
    std::sort(SUColors.begin(), SUColors.end()); 
    SUColors.erase(std::unique(SUColors.begin(), SUColors.end()), SUColors.end());
    if (SUColors.size() == 1)
      Colors[SU->NodeNum] = SUColors[0];
  }

  // Special case: predecessors of high latency instructions that have no other successors
  // are put in the high latency instruction block.
  // This reduces sgpr usage significantly, but also means we do wait just after low latency
  // instructions. A pass could be added after scheduling to move low latency instructions at better locations
  // without increasing sgpr pressure.

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (Colors[Succ->NodeNum] > 0)
        SUColors.push_back(Colors[Succ->NodeNum]);
    }
    // keep previous color
    if (SUColors.size() == 0)
      continue;
    std::sort(SUColors.begin(), SUColors.end()); 
    SUColors.erase(std::unique(SUColors.begin(), SUColors.end()), SUColors.end());
    if (SUColors.size() == 1 && SUColors[0] <= maxHighLatencyID && SUColors[0] > 0)
      Colors[SU->NodeNum] = SUColors[0];
  }

  //TODO: Document
  /*for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[BottomUpIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    if (SU->Preds.size() > SU->Succs.size())
      continue;

    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      SUColors.push_back(Colors[Succ->NodeNum]);
    }
    std::sort(SUColors.begin(), SUColors.end()); 
    SUColors.erase(std::unique(SUColors.begin(), SUColors.end()), SUColors.end());
    if (SUColors.size() == 0)
      continue;
    if (SUColors.size() == 1 && SUColors[0] != Colors[SU->NodeNum]) {
      maxID++;
      Colors[SU->NodeNum] = maxID;
    }
  }

  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[TopDownIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    if (SU->Preds.size() < SU->Succs.size())
      continue;

    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *Pred = I->getSUnit();
      if (I->isWeak() || Pred->NodeNum >= DAGSize)
        continue;
      SUColors.push_back(Colors[Pred->NodeNum]);
    }
    std::sort(SUColors.begin(), SUColors.end()); 
    SUColors.erase(std::unique(SUColors.begin(), SUColors.end()), SUColors.end());
    if (SUColors.size() == 0)
      continue;
    if (SUColors.size() == 1 && SUColors[0] != Colors[SU->NodeNum]) {
      maxID++;
      Colors[SU->NodeNum] = maxID;
    }
  }*/

  // To remove blocks:
  /*for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[TopDownIndex2Node[i]];
    std::vector<unsigned> SUColors;

    // High latency instructions: already given
    if (Colors[SU->NodeNum] <= maxHighLatencyID && Colors[SU->NodeNum] > 0)
      continue;

    {
      maxID++;
      Colors[SU->NodeNum] = maxID;
    }
  }*/

  // Put SU of same color into same block
  RealID.resize(maxID+1, -1);
  Node2Block.resize(DAGSize, -1);
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned color = Colors[SU->NodeNum];
    if (RealID[color] == -1) {
      int ID = Blocks.size();
      Blocks.push_back(make_unique<SIBlockSchedule>(this, ID, (color > 0) && (color <= maxHighLatencyID)));
      RealID[color] = ID;
    }
    Blocks[RealID[color]]->addUnit(SU);
    Node2Block[SU->NodeNum] = RealID[color];
  }

  // Build dependencies between blocks
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    int SUID = Node2Block[i];
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *Succ = I->getSUnit();
      if (I->isWeak() || Succ->NodeNum >= DAGSize)
        continue;
      if (Node2Block[Succ->NodeNum] != SUID)
        Blocks[SUID]->addSucc(Blocks[Node2Block[Succ->NodeNum]].get());
    }
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
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
    unsigned color = Colors[SU->NodeNum];
    Blocks[RealID[color]]->releaseSuccessors(SU, false);
    Blocks[RealID[color]]->releasePredecessors(SU, false);
  }
  DEBUG(dbgs() << "Blocks created:\n\n");
  DEBUG(
    for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
      SIBlockSchedule *Block = Blocks[i].get();
      Block->printDebug(true);
    }
  );
}

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

void SIScheduleDAGMI::scheduleInsideBlocks() {
  DEBUG(dbgs() << "\nScheduling Blocks\n\n");
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

  unsigned DAGSize = Blocks.size();
  std::vector<int> WorkList;
  std::vector<int> TopDownIndex2Indice;
  std::vector<int> TopDownIndice2Index;

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
    Id--;
    TopDownIndice2Index[i] = Id;
    TopDownIndex2Indice[Id] = i;
    for (unsigned j = 0, e = Block->Preds.size(); j != e; ++j) {
      SIBlockSchedule *Pred = Block->Preds[j];
      if (!--TopDownIndice2Index[Pred->ID])
        WorkList.push_back(Pred->ID);
    }
  }

  #ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    for (unsigned j = 0, e = Block->Preds.size(); j != e; ++j) {
      SIBlockSchedule *Pred = Block->Preds[j];
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

    for (unsigned j = 0, ej = SUs.size(); j != ej; ++j) {
      MachineInstr *MI = SUs[j]->getInstr();
      MachineBasicBlock::iterator Pos = MI;
      PosOld.push_back(Pos);
      if (&*CurrentTopFastSched == MI) {
        PosNew.push_back(Pos);
        CurrentTopFastSched = nextIfDebug(++CurrentTopFastSched, CurrentBottom);
      } else {
        // Update the instruction stream.
        BB->splice(CurrentTopFastSched, BB, MI);

        // Update LiveIntervals
        LIS->handleMove(MI, /*UpdateFlags=*/true);
        PosNew.push_back(CurrentTopFastSched);
      }
    }
  }

  // Now we have Block of SUs == Block of MI
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    std::vector<SUnit*> SUs = Block->getScheduledUnits();
    Block->schedule(SUs[0]->getInstr(), SUs[SUs.size()-1]->getInstr());
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
    for (unsigned j = 0, ej = Block->getInRegs().size(); j != ej; ++j) {
      unsigned Reg = Block->getInRegs()[j];
      for (unsigned k = 0, e = Block->Preds.size(); k != e; ++k) {
        SIBlockSchedule *Pred = Block->Preds[k];
        ArrayRef<unsigned> PredOutRegs = Pred->getOutRegs();
        ArrayRef<unsigned>::iterator RegPos = std::find(PredOutRegs.begin(), PredOutRegs.end(), Reg);
        if (RegPos != PredOutRegs.end()) {
          ++Pred->LiveOutRegsNumUsages[RegPos - PredOutRegs.begin()];
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

void SIScheduleDAGMI::moveLowLatencies() {
   unsigned DAGSize = SUnits.size();
   int last_low_latency_user = -1;
   int last_low_latency_pos = -1;
  // TODO: form groups
  // Move low latencies sooner
   for (unsigned i = 0, e = ScheduledSUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[ScheduledSUnits[i]];
    unsigned MinPos = 0;

    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *Pred = I->getSUnit();
      if (SITII->isLowLatencyInstruction(Pred->getInstr())) {
        last_low_latency_user = i;
      }
      if (Pred->NodeNum >= DAGSize)
        continue;
      unsigned PredPos = ScheduledSUnitsInv[Pred->NodeNum];
      if (PredPos >= MinPos)
        MinPos = PredPos + 1;
    }

    if (SITII->isLowLatencyInstruction(SU->getInstr())) {
      unsigned BestPos = last_low_latency_user + 1;
      if (BestPos > i) // instruction is user of low latency too
        BestPos = i;
      if ((int)BestPos <= last_low_latency_pos)
        BestPos = last_low_latency_pos + 1;
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
      last_low_latency_pos = BestPos;
    }
  }
}

std::vector<int> SIScheduleDAGMI::checkRegUsageImpact(ArrayRef<unsigned> InRegs,
                                                      ArrayRef<unsigned> OutRegs) {
  std::vector<int> DiffSetPressure;
  DiffSetPressure.assign(TRI->getNumRegPressureSets(), 0);

  for (unsigned i = 0, e = InRegs.size(); i != e; ++i) {
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(InRegs[i]))
      continue;
    if (LiveRegsConsumers[InRegs[i]] > 1)
      continue;
    PSetIterator PSetI = MRI.getPressureSets(InRegs[i]);
    for (; PSetI.isValid(); ++PSetI) {
      DiffSetPressure[*PSetI] -= PSetI.getWeight();
    }
  }

  for (unsigned i = 0, e = OutRegs.size(); i != e; ++i) {
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(OutRegs[i]))
      continue;
    PSetIterator PSetI = MRI.getPressureSets(OutRegs[i]);
    for (; PSetI.isValid(); ++PSetI) {
      DiffSetPressure[*PSetI] += PSetI.getWeight();
    }
  }

  return DiffSetPressure;
}

void SIScheduleDAGMI::prepareScheduleVariant() {
  LiveRegs.clear();
  LiveRegsConsumers.clear();
  numBlockScheduled = 0;
  ReadyBlocks.clear();
  VGPRSUsedAfterBlock.clear();
  BlockScheduleOrder.clear();
  BlockScheduleOrder.reserve(Blocks.size());
  VGPRSUsedAfterBlock.reserve(Blocks.size());

  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    Block->NumPredsLeft = Block->Preds.size();
    Block->NumSuccsLeft = Block->Succs.size();
    Block->firstPosUserScheduled = 0;
    Block->lastPosHighLatencyParentScheduled = 0;
  }

  addLiveRegs(RPTracker.getPressure().LiveInRegs);

  // fill LiveRegsConsumers for regs that were already defined before scheduling
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    SIBlockSchedule *Block = Blocks[i].get();
    for (unsigned j = 0, ej = Block->getInRegs().size(); j != ej; ++j) {
      unsigned Reg = Block->getInRegs()[j];
      bool found = false;
      for (unsigned k = 0, e = Block->Preds.size(); k != e; ++k) {
        SIBlockSchedule *Pred = Block->Preds[k];
        ArrayRef<unsigned> PredOutRegs = Pred->getOutRegs();
        ArrayRef<unsigned>::iterator RegPos = std::find(PredOutRegs.begin(), PredOutRegs.end(), Reg);
        if (RegPos != PredOutRegs.end()) {
          found = true;
          break;
        }
      }
      
      if (!found) {
        if (LiveRegsConsumers.find(Reg) == LiveRegsConsumers.end())
          LiveRegsConsumers[Reg] = 1; // Add element
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

void SIScheduleDAGMI::addLiveRegs(ArrayRef<unsigned> Regs) {
  for (unsigned I = 0, E = Regs.size(); I != E; ++I) {
    // For now only track virtual registers
    if (!TargetRegisterInfo::isVirtualRegister(Regs[I]))
      continue;
    // if not already in the live set, then add it
    (void) LiveRegs.insert(Regs[I]);
  }
}

void SIScheduleDAGMI::decreaseLiveRegs(ArrayRef<unsigned> Regs) {
  for (unsigned I = 0, E = Regs.size(); I != E; ++I) {
    // For now only track virtual registers
    std::set<unsigned>::iterator Pos = LiveRegs.find(Regs[I]);
    assert (Pos != LiveRegs.end()); // Reg must be live
    assert (LiveRegsConsumers.find(Regs[I]) != LiveRegsConsumers.end());
    assert (LiveRegsConsumers[Regs[I]] >= 1);
    --LiveRegsConsumers[Regs[I]];
    if (LiveRegsConsumers[Regs[I]] == 0) {
      LiveRegs.erase(Pos);
    }
  }
}

void SIScheduleDAGMI::releaseBlockSuccs(SIBlockSchedule *Parent) {
  for (unsigned i = 0, e = Parent->Succs.size(); i != e; ++i) {
    SIBlockSchedule *Block = Blocks[Parent->Succs[i]->ID].get();
    Block->NumPredsLeft--;
    if (Block->NumPredsLeft == 0) {
      ReadyBlocks.push_back(Block);
    }
    if (Parent->isHighLatencyBlock())
      Block->lastPosHighLatencyParentScheduled = numBlockScheduled;
  }
}

void SIScheduleDAGMI::blockScheduled(SIBlockSchedule *Block) {
  if (numBlockScheduled == 0)
    VGPRSUsedAfterBlock[numBlockScheduled] = checkRegUsageImpact(llvm::ArrayRef<unsigned> (), RPTracker.getPressure().LiveInRegs)[VGPRSetID];
  else
    VGPRSUsedAfterBlock[numBlockScheduled] = VGPRSUsedAfterBlock[numBlockScheduled-1];
  VGPRSUsedAfterBlock[numBlockScheduled] += checkRegUsageImpact(Block->getInRegs(), Block->getOutRegs())[VGPRSetID];
  DEBUG(dbgs() << " ***** " << VGPRSUsedAfterBlock[numBlockScheduled] << " *** ");
  assert (VGPRSUsedAfterBlock[numBlockScheduled] <= __INT_MAX__); // check never gets below 0 if signed

  decreaseLiveRegs(Block->getInRegs());
  addLiveRegs(Block->getOutRegs());
  releaseBlockSuccs(Block);
  ArrayRef<unsigned> Regs = Block->getOutRegs();
  for (unsigned I = 0, E = Regs.size(); I != E; ++I) {
    if (LiveRegsConsumers.find(Regs[I]) == LiveRegsConsumers.end())
      LiveRegsConsumers[Regs[I]] = Block->LiveOutRegsNumUsages[I]; // Add element
    else
      LiveRegsConsumers[Regs[I]] += Block->LiveOutRegsNumUsages[I];
  }
  for (unsigned i = 0, e = Block->Preds.size(); i != e; ++i) {
    SIBlockSchedule *Parent = Blocks[Block->Preds[i]->ID].get();
    if (Parent->firstPosUserScheduled == 0)
      Parent->firstPosUserScheduled = numBlockScheduled;
  }
  numBlockScheduled++;
}

float SIScheduleDAGMI::getSchedulingScore()
{
  std::vector<unsigned> sumBlockCosts;
  unsigned minCostLatencyHiding = SUnits.size();
  unsigned meanCostLatencyHiding = 0;
  unsigned numHighLatInstructions = 0;
  unsigned numWavefronts;
  unsigned maxVGPRUsage = 0;

  sumBlockCosts.reserve(Blocks.size());

  for (unsigned b = 0; b < Blocks.size(); b++) {
    SIBlockSchedule *Block = Blocks[BlockScheduleOrder[b]].get();
    if (b == 0)
      sumBlockCosts[b] = Block->getCost();
    else
      sumBlockCosts[b] = sumBlockCosts[b-1] + Block->getCost();
    if (VGPRSUsedAfterBlock[b] > maxVGPRUsage)
      maxVGPRUsage = VGPRSUsedAfterBlock[b];
  }

  for (unsigned b = 0; b < Blocks.size(); b++) {
    unsigned HidingCost;
    SIBlockSchedule *Block = Blocks[BlockScheduleOrder[b]].get();
    if (Block->isHighLatencyBlock() && Block->firstPosUserScheduled > b) {
      HidingCost = sumBlockCosts[Block->firstPosUserScheduled] - sumBlockCosts[b + 1];
      numHighLatInstructions++;
      if (HidingCost < minCostLatencyHiding)
        minCostLatencyHiding = HidingCost;
      meanCostLatencyHiding += HidingCost;
    }
  }

  if (numHighLatInstructions)
    meanCostLatencyHiding /= numHighLatInstructions;

  numWavefronts = getWaveFrontsForUsage(0, maxVGPRUsage);
  DEBUG(dbgs() << "(determined Wavefront Count: " << numWavefronts << ") " << maxVGPRUsage << " " << meanCostLatencyHiding << " " << minCostLatencyHiding << " ");
  

  return (meanCostLatencyHiding + minCostLatencyHiding) * numWavefronts;
}

unsigned SIScheduleDAGMI::getWaveFrontsForUsage(unsigned SGPRsUsed,
                                                unsigned VGPRsUsed) {
  unsigned i;

  for (i = 9; i > 0; i--) {
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

  if (currentVariant == ReguseLatencyTopDown)
    if (tryLess(TryCand.VGPRUsageDiff, Cand.VGPRUsageDiff, TryCand, Cand, RegUsage))
      return;
  if (tryLess(TryCand.lastHighLatParentScheduled, Cand.lastHighLatParentScheduled, TryCand, Cand, Latency))
    return;
  if (tryGreater(TryCand.isHighLatency, Cand.isHighLatency, TryCand, Cand, Latency))
    return;
  if (tryGreater(TryCand.SuccessorReadiness, Cand.SuccessorReadiness, TryCand, Cand, Latency))
    return;
  if (currentVariant == LatencyReguseTopDown)
    if (tryLess(TryCand.VGPRUsageDiff, Cand.VGPRUsageDiff, TryCand, Cand, RegUsage))
      return;
}

SIBlockSchedule *SIScheduleDAGMI::pickBlock() {
  SIBlockSchedCandidate Cand;
  std::vector<SIBlockSchedule*>::iterator Best;
  SIBlockSchedule *Block;
  if (ReadyBlocks.size() == 0)
    return nullptr;

  DEBUG(
    dbgs() << "Picking New Blocks\n";
    dbgs() << "Available: ";
    for (std::vector<SIBlockSchedule*>::iterator I = ReadyBlocks.begin(), E = ReadyBlocks.end(); I != E; ++I)
      dbgs() << (*I)->ID << " ";
    dbgs() << "\nCurrent Live:\n";
    for (std::set<unsigned>::iterator I = LiveRegs.begin(), e = LiveRegs.end(); I != e; I++)
      dbgs() << PrintVRegOrUnit(*I, TRI) << " ";
    dbgs() << "\n";
  );

  for (std::vector<SIBlockSchedule*>::iterator I = ReadyBlocks.begin(), E = ReadyBlocks.end(); I != E; ++I) {
    SIBlockSchedCandidate TryCand;
    TryCand.Block = *I;
    
    TryCand.SuccessorReadiness = 0;//TODO
    for (unsigned j = 0, e = TryCand.Block->Succs.size(); j != e; ++j) {
      SIBlockSchedule *Succ = TryCand.Block->Succs[j];
      TryCand.SuccessorReadiness += Succ->isHighLatencyBlock();
    }
    TryCand.WaveFronts = 0; //TODO
    TryCand.VGPRUsageDiff = checkRegUsageImpact((*I)->getInRegs(), (*I)->getOutRegs())[VGPRSetID];
    TryCand.lastHighLatParentScheduled = TryCand.Block->lastPosHighLatencyParentScheduled;
    TryCand.isHighLatency = TryCand.Block->isHighLatencyBlock();
    tryCandidate(Cand, TryCand);
    if (TryCand.Reason != NoCand) {
      Cand.setBest(TryCand);
      Best = I;
      DEBUG(dbgs() << "Best Current Choice: " << Cand.Block->ID << " " << getReasonStr(Cand.Reason) << "\n");
    }
  }

  DEBUG(
    dbgs() << "Picking: " << Cand.Block->ID << "\n";
    if (Cand.isHighLatency)
      dbgs() << "Is a block with high latency instruction\n";
    dbgs() << "Position of last high latency dependency: " << Cand.lastHighLatParentScheduled << "\n";
    dbgs() << "VGPRUsageDiff: " << Cand.VGPRUsageDiff << "\n";
    dbgs() << "\n";
  );

  Block = Cand.Block;
  ReadyBlocks.erase(Best);
  return Block;
}

const char *SIScheduleDAGMI::getVariantStr(SIScheduleVariant variant)
{
  switch (variant) {
  case LatencyOnlyTopDown:              return "LatencyOnlyTopDown";
  case LatencyReguseTopDown:            return "LatencyReguseTopDown";
  case ReguseLatencyTopDown:            return "ReguseLatencyTopDown";
  case LatencyOnlyBottomUp:             return "LatencyOnlyBottomUp";
  case LatencyReguseBottomUp:           return "LatencyReguseBottomUp";
  case ReguseLatencyBottomUp:           return "ReguseLatencyBottomUp";
  case LatencyReguseDeepSearchBottomUp: return "LatencyReguseDeepSearchBottomUp";
  };
  llvm_unreachable("Unknown reason!");
}

void SIScheduleDAGMI::scheduleWithVariant(SIScheduleVariant variant)
{
  float score;
  prepareScheduleVariant();

  DEBUG(dbgs() << "Variant: " << getVariantStr(variant) << "\n");
  currentVariant = variant;
  while (SIBlockSchedule *Block = pickBlock()) {
    BlockScheduleOrder.push_back(Block->ID);
    blockScheduled(Block);
  }

  DEBUG(
    dbgs() << "Block Order:";
    for (unsigned i = 0, e = BlockScheduleOrder.size(); i != e; ++i) {
      dbgs() << " " << BlockScheduleOrder[i];
    }
  );

  DEBUG(dbgs() << "\nScore to hide latencies (Higher is better): ");
  score = getSchedulingScore();
  DEBUG(dbgs() << score << "\n\n");

  // TODO: call improveSchedule;
  if (score > bestVariantScore) {
    bestVariantScore = score;
    bestVariant = variant;
    bestBlockScheduleOrder = BlockScheduleOrder;
  }
}

void SIScheduleDAGMI::improveSchedule()
{
  // TODO
}
void SIScheduleDAGMI::schedule()
{
  prepareSchedule();
  bestVariant = LatencyOnlyTopDown;
  bestVariantScore = -1.;
  bestBlockScheduleOrder.clear();

  for (unsigned v = LatencyOnlyTopDown; v <= ReguseLatencyTopDown/*LatencyReguseDeepSearchBottomUp*/; v++) {
    scheduleWithVariant(SIScheduleVariant(v));
  }

  DEBUG(dbgs() << "Best Variant: " << getVariantStr(bestVariant) << "\n");
  if (bestVariant == LatencyReguseTopDown) dbgs() << "used1\n";
  if (bestVariant == ReguseLatencyTopDown) dbgs() << "used2\n";
  ScheduledSUnits.clear();
  ScheduledSUnitsInv.resize(SUnits.size()); // do only now as SUnits wasn't filled before

  for (unsigned b = 0; b < Blocks.size(); b++) {
    SIBlockSchedule *Block = Blocks[bestBlockScheduleOrder[b]].get();
    std::vector<SUnit*> SUs = Block->getScheduledUnits();

    for (unsigned i = 0, e = SUs.size(); i != e; ++i) {
      ScheduledSUnitsInv[SUs[i]->NodeNum] = ScheduledSUnits.size();
      ScheduledSUnits.push_back(SUs[i]->NodeNum);
    }
  }

  moveLowLatencies();

  assert(TopRPTracker.getPos() == RegionBegin && "bad initial Top tracker");
  TopRPTracker.setPos(CurrentTop);

  for (unsigned i = 0, e = ScheduledSUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[ScheduledSUnits[i]];

    scheduleMI(SU, true);

    DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") " << *SU->getInstr());
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
