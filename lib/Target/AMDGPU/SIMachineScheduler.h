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

public:
  SIBlockSchedule(SIScheduleDAGMI *dag, unsigned ID, bool isHighLatencyBlock):
    DAG(dag), SUnits(), TopReadySUs(), BottomReadySUs(), ScheduledSUnits(),
    TopRPTracker(TopPressure), BotRPTracker(BotPressure), RPTracker(Pressure),
    scheduled(false), highLatencyBlock(isHighLatencyBlock), ID(ID),
    lastPosHighLatencyParentScheduled(0), firstPosUserScheduled(0),
    Preds(), Succs(), NumPredsLeft(0), NumSuccsLeft(0) {};

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
  // fast schedule with no particular requirement
  void fastSchedule();

  // Complete schedule that will take try to minimize reg pressure, and will
  // fill correct track of liveins and liveouts.
  // Needs all MIs to be groupped begin BeginBlock and EndBlock.
  // The MIs can be moved after the scheduling. It is just used to allow precise computation.
  void schedule(MachineBasicBlock::iterator BeginBlock, MachineBasicBlock::iterator EndBlock);
  bool isScheduled() {return scheduled;}
  bool isHighLatencyBlock() {return highLatencyBlock;}

  // TODO: should take into accounts some instructions (rcp, etc) are 4 times slower
  int getCost() {return SUnits.size();}

  std::vector<SUnit*> getScheduledUnits() {return ScheduledSUnits;}
  // TODO: find a way to have better computation here. It returns the maximum Pressure inside the block, but
  // it counts potentially some liveins and liveouts.
  std::vector<unsigned> &getInternalRegUsage() { return InternalPressure;}
  // Pressure = sum_alive_registers register size
  // Internally llvm will represent some registers as big 128 bits registers for example,
  // but they actually correspond to 4 actual 32 bits registers for example.
  // Thus Pressure is not equal to num_alive_registers * constant.
  std::vector<unsigned> &getInRegUsage() {return LiveInPressure;}
  std::vector<unsigned> &getOutRegUsage() {return LiveOutPressure;}
  ArrayRef<unsigned> getInRegs() {return LiveInRegs;}
  ArrayRef<unsigned> getOutRegs() {return LiveOutRegs;}
  // How many blocks use the outputs registers
  SmallVector<unsigned,8> LiveOutRegsNumUsages;

  unsigned ID;
  // Store at which pos the last High latency parent was scheduled
  unsigned lastPosHighLatencyParentScheduled;
  // Position of the first block using the outputs
  unsigned firstPosUserScheduled;

  std::vector<SIBlockSchedule*> Preds;  // All blocks predecessors.
  std::vector<SIBlockSchedule*> Succs;  // All blocks successors.

  unsigned NumPredsLeft;
  unsigned NumSuccsLeft;

  /// Add block pred, which has instruction predecessor of SU.
  void addPred(SIBlockSchedule *pred);
  void addSucc(SIBlockSchedule *succ);

  // InOrOutBlock: restrict to links pointing inside the block (true),
  // or restrict to links pointing outside the block (false).
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
  void initRegPressure(MachineBasicBlock::iterator BeginBlock, MachineBasicBlock::iterator EndBlock);
};

class SIScheduleDAGMI : public ScheduleDAGMILive {
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

  // For moveLowLatencies. After all Scheduling variants are tested.
  std::vector<unsigned> ScheduledSUnits;
  std::vector<unsigned> ScheduledSUnitsInv;

  // For Scheduling
  std::vector<SIBlockSchedule*> ReadyBlocks;
  unsigned numBlockScheduled;
  std::set<unsigned> LiveRegs;
  std::map<unsigned, unsigned> LiveRegsConsumers; // num of schedulable unscheduled blocks reading the register
  std::vector<unsigned> BlockScheduleOrder;
  std::vector<unsigned> VGPRSUsedAfterBlock;

public:
  SIScheduleDAGMI(MachineSchedContext *C);

  ~SIScheduleDAGMI() override;

  // entry point for the schedule
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

  // To init Block's RPTracker
  void initRPTracker(RegPressureTracker &RPTracker) {
    RPTracker.init(&MF, RegClassInfo, LIS, BB, RegionBegin);
  }

  LiveIntervals *getLIS() {return LIS;}
  MachineRegisterInfo *getMRI() {return &MRI;}
  const TargetRegisterInfo *getTRI() {return TRI;}

private:
  // To prepare the schedule
  void prepareSchedule();
  void topologicalSort();
  void createBlocks();
  void scheduleInsideBlocks();
  // After scheduling is done, improve low latency placements
  void moveLowLatencies();
  // To help the schedule
  // check register pressure change by scheduling a block with these LiveIn and LiveOut
  std::vector<int> checkRegUsageImpact(ArrayRef<unsigned> InRegs, ArrayRef<unsigned> OutRegs);
  // reinits what needs to be.
  void prepareScheduleVariant();
  void addLiveRegs(ArrayRef<unsigned> Regs);
  void decreaseLiveRegs(ArrayRef<unsigned> Regs);
  void releaseBlockSuccs(SIBlockSchedule *Parent);
  void blockScheduled(SIBlockSchedule *Block);
  // higher is better
  float getSchedulingScore();
  // More Wavefronts = better latency hidding
  unsigned getWaveFrontsForUsage(unsigned SGPRsUsed, unsigned VGPRsUsed);
  
  enum SIScheduleVariant {
    LatencyOnlyTopDown, LatencyReguseTopDown, ReguseLatencyTopDown,
    LatencyOnlyBottomUp, LatencyReguseBottomUp, ReguseLatencyBottomUp,
    LatencyReguseDeepSearchBottomUp
  };

  SIScheduleVariant currentVariant;
  SIScheduleVariant bestVariant;
  float bestVariantScore;
  std::vector<unsigned> bestBlockScheduleOrder;

  void tryCandidate(SIBlockSchedCandidate &Cand, SIBlockSchedCandidate &TryCand);
  void traceCandidate(const SIBlockSchedCandidate &Cand);
  SIBlockSchedule *pickBlock();
  const char *getVariantStr(SIScheduleVariant variant);
  void scheduleWithVariant(SIScheduleVariant variant);

  // Once a schedule Variant is run, compute current wavefront count, and try
  // move latency instructions to keep this wavefront count the same, but with better
  // latency overall
  void improveSchedule();

public:
  const SIInstrInfo *SITII;
  const SIRegisterInfo *SITRI;
};

} // namespace llvm

#endif /* SIMACHINESCHEDULER_H_ */
 
