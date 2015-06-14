//===-- SIMachineScheduler.h - SI Scheduler Interface -*- C++ -*-------===//
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

#ifndef LLVM_LIB_TARGET_AMDGPU_SIMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_AMDGPU_SIMACHINESCHEDULER_H

#include "llvm/CodeGen/RegisterPressure.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace llvm {

enum SIScheduleCandReason {
    NoCand, RegUsage, SIWaveFronts, Latency, Weak, ReadySuccessor, Height, Depth, NodeOrder};

struct SISchedulerCandidate {
  // The reason for this candidate.
  SIScheduleCandReason Reason;

  // Set of reasons that apply to multiple candidates.
  uint32_t RepeatReasonSet;

  SISchedulerCandidate()
    :  Reason(NoCand), RepeatReasonSet(0) {}

  bool isRepeat(SIScheduleCandReason R) { return RepeatReasonSet & (1 << R); }
  void setRepeat(SIScheduleCandReason R) { RepeatReasonSet |= (1 << R); }
};

class SIScheduleDAGMI;

class SIBlockSchedule {
  SIScheduleDAGMI *DAG;

  std::vector<SUnit*> SUnits;
  std::vector<SUnit*> TopReadySUs;
  std::vector<SUnit*> BottomReadySUs;
  std::vector<SUnit*> ScheduledSUnits;

  /// The top of the unscheduled zone.
  IntervalPressure TopPressure;
  RegPressureTracker TopRPTracker;

  /// The bottom of the unscheduled zone.
  IntervalPressure BotPressure;
  RegPressureTracker BotRPTracker;

  IntervalPressure Pressure;
  RegPressureTracker RPTracker;

  std::vector<unsigned> InternalPressure;
  std::vector<unsigned> LiveInPressure;
  std::vector<unsigned> LiveOutPressure;
  SmallVector<unsigned,8> LiveInRegs;
  SmallVector<unsigned,8> LiveOutRegs;

  bool scheduled;
  bool highLatencyBlock;

  MachineBasicBlock::iterator BeginBlock;
  MachineBasicBlock::iterator EndBlock;
  bool BlockBeginEndFilled;
public:
  SIBlockSchedule(SIScheduleDAGMI *dag, unsigned ID, bool isHighLatencyBlock);

  void addUnit(SUnit *SU);

  struct SISchedCandidate : SISchedulerCandidate {
    // The best SUnit candidate.
    SUnit *SU;

    // WaveFronts estimated if the best candidate is scheduled
    unsigned VGPRUsage;
    bool isLowLatency;

    SISchedCandidate()
      : SU(nullptr) {}

    bool isValid() const { return SU; }

    // Copy the status of another candidate without changing policy.
    void setBest(SISchedCandidate &Best) {
      assert(Best.Reason != NoCand && "uninitialized Sched candidate");
      SU = Best.SU;
      Reason = Best.Reason;
      VGPRUsage = Best.VGPRUsage;
      isLowLatency = Best.isLowLatency;
    }
  };

  // The block Predecessors and Successors must be all registered
  // before scheduling
  void fastSchedule();
  void schedule();
  bool isScheduled() {return scheduled;}
  bool isHighLatencyBlock() {return highLatencyBlock;}
  void fillMIBoundaries(MachineBasicBlock::iterator Begin, MachineBasicBlock::iterator End) {
    BeginBlock = Begin;
    EndBlock = End;
    BlockBeginEndFilled = true;
  }
  // TODO: should take into accounts some instructions (rcp, etc) are 4 times slower
  int getCost() {return SUnits.size();}

  std::vector<SUnit*> getScheduledUnits() {return ScheduledSUnits;}
  std::vector<unsigned> &getInternalRegUsage() { return InternalPressure;} // This is bigger the the internal register usage, since it may count LiveIn + LiveOut
  std::vector<unsigned> &getInRegUsage() {return LiveInPressure;}
  std::vector<unsigned> &getOutRegUsage() {return LiveOutPressure;}
  ArrayRef<unsigned> getInRegs() {return LiveInRegs;}
  ArrayRef<unsigned> getOutRegs() {return LiveOutRegs;}
  SmallVector<unsigned,8> LiveOutRegsNumUsages;

  unsigned ID;
  // Store at which pos the last High latency parent was scheduled
  unsigned lastPosHighLatencyParentScheduled;

  std::vector<SIBlockSchedule*> Preds;  // All blocks predecessors.
  std::vector<SIBlockSchedule*> Succs;  // All blocks successors.

  unsigned NumPredsLeft;
  unsigned NumSuccsLeft;

  /// Add block pred, which has instruction predecessor of SU.
  void addPred(SIBlockSchedule *pred);
  void addSucc(SIBlockSchedule *succ);

  void releaseSuccessors(SUnit *SU, bool InOrOutBlock);
  void releasePredecessors(SUnit *SU, bool InOrOutBlock);

  void printDebug(bool full);

private:
  void undoSchedule();
  void undoReleaseSucc(SUnit *SU, SDep *SuccEdge, bool InOrOutBlock);
  void releaseSucc(SUnit *SU, SDep *SuccEdge, bool InOrOutBlock);
  void releasePred(SUnit *SU, SDep *PredEdge, bool InOrOutBlock);
  void NodeScheduled(SUnit *SU);
  void tryCandidateTopDown(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  void tryCandidateBottomUp(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  SUnit* pickNode(bool TopOnly, bool &isTop);
  void traceCandidate(const SISchedCandidate &Cand);
  void initRegPressure();
};

class SIScheduleDAGMI : public ScheduleDAGMILive {
  unsigned CurrentWaveFronts;
  unsigned SGPRsForWaveFronts[10];
  unsigned VGPRsForWaveFronts[10];
  unsigned VGPRSetID;
  unsigned SGPRSetID;

  /// Topological sort
  /// Index2Node - Maps topological index to the node number.
  std::vector<int> TopDownIndex2Node;
  std::vector<int> BottomUpIndex2Node;
  /// Node2Index - Maps the node number to its topological index.
  std::vector<int> TopDownNode2Index;
  std::vector<int> BottomUpNode2Index;

  /// Block number associated to the Node
  std::vector<int> Node2Block;

  std::vector<std::unique_ptr<SIBlockSchedule>> Blocks;
  std::vector<SIBlockSchedule*> ReadyBlocks;
  std::vector<unsigned> ScheduledSUnits;
  std::vector<unsigned> ScheduledSUnitsInv;

  unsigned numBlockScheduled;
  std::set<unsigned> LiveRegs;
  std::map<unsigned, unsigned> LiveRegsConsumers; // num of schedulable unscheduled blocks reading the register

public:
  SIScheduleDAGMI(MachineSchedContext *C);

  ~SIScheduleDAGMI() override;

  void schedule() override;

  struct SIBlockSchedCandidate : SISchedulerCandidate {
    // The best Block candidate.
    SIBlockSchedule *Block;

    // WaveFronts estimated if the best candidate is scheduled
    unsigned WaveFronts;

    unsigned SuccessorReadiness;

    unsigned lastHighLatParentScheduled;
    bool isHighLatency;
    int VGPRUsageDiff;

    SIBlockSchedCandidate()
      : Block(nullptr) {}

    bool isValid() const { return Block; }

    // Copy the status of another candidate without changing policy.
    void setBest(SIBlockSchedCandidate &Best) {
      assert(Best.Reason != NoCand && "uninitialized Sched candidate");
      Block = Best.Block;
      Reason = Best.Reason;
      VGPRUsageDiff = Best.VGPRUsageDiff;
      WaveFronts = Best.WaveFronts;
      SuccessorReadiness = Best.SuccessorReadiness;
      lastHighLatParentScheduled = Best.lastHighLatParentScheduled;
      isHighLatency = Best.isHighLatency;
    }
  };

  bool isSUInBlock(SUnit *SU, unsigned ID) {
    if (SU->NodeNum >= SUnits.size())
      return false;
    return Blocks[Node2Block[SU->NodeNum]]->ID == ID;
  }

  void initRPTracker(RegPressureTracker &RPTracker) {
    RPTracker.init(&MF, RegClassInfo, LIS, BB, RegionBegin);
  }

  LiveIntervals *getLIS() {return LIS;}
  MachineRegisterInfo *getMRI() {return &MRI;}
  const TargetRegisterInfo *getTRI() {return TRI;}

private:
  SIBlockSchedule *pickBlock();
  void releaseBlockSuccs(SIBlockSchedule *Parent);
  void addLiveRegs(ArrayRef<unsigned> Regs);
  void decreaseLiveRegs(ArrayRef<unsigned> Regs);
  void blockScheduled(SIBlockSchedule *Block);
  // check register pressure change by scheduling a block with these LiveIn and LiveOut
  std::vector<int> checkRegUsageImpact(ArrayRef<unsigned> InRegs, ArrayRef<unsigned> OutRegs);
  void prepareSchedule();
  void topologicalSort();
  void createBlocks();
  void scheduleInsideBlocks();
  unsigned getWaveFrontsForUsage(unsigned SGPRsUsed, unsigned VGPRsUsed);
  //unsigned getWaveFrontsAfterBlock(SIBlockSchedule* Block);
  //unsigned determineWaveFronts();
  //unsigned computeSuccessorReadiness(SIBlockSchedule* Block);
  void tryCandidate(SIBlockSchedCandidate &Cand, SIBlockSchedCandidate &TryCand);
  void traceCandidate(const SIBlockSchedCandidate &Cand);
  void moveInstruction(MachineInstr *MI, MachineBasicBlock::iterator InsertPos);
  void moveLowLatencies();
public:
  const SIInstrInfo *SITII;
  const SIRegisterInfo *SITRI;
};

} // namespace llvm

#endif /* SIMACHINESCHEDULER_H_ */
 
