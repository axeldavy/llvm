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

#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterPressure.h"

using namespace llvm;

namespace llvm {

enum SIScheduleCandReason {
    NoCand, RegUsage, Latency, Successor, Depth, NodeOrder};

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
  std::vector<SUnit*> ScheduledSUnits;

  /// The top of the unscheduled zone.
  IntervalPressure TopPressure;
  RegPressureTracker TopRPTracker;

  std::vector<unsigned> InternalAdditionnalPressure;
  std::vector<unsigned> LiveInPressure;
  std::vector<unsigned> LiveOutPressure;
  std::set<unsigned> LiveInRegs;
  std::set<unsigned> LiveOutRegs;

  bool Scheduled;
  bool HighLatencyBlock;

public:
  SIBlockSchedule(SIScheduleDAGMI *Dag, unsigned ID, bool IsHighLatencyBlock):
    DAG(Dag), SUnits(), TopReadySUs(), ScheduledSUnits(),
    TopRPTracker(TopPressure), Scheduled(false),
    HighLatencyBlock(IsHighLatencyBlock), ID(ID),
    Preds(), Succs(), NumPredsLeft(0), NumSuccsLeft(0) {};

  // Unique ID, the index of the Block in the SIScheduleDAGMI Blocks table
  unsigned ID;

  /// functions for Block construction
  void addUnit(SUnit *SU);
  // Add block pred, which has instruction predecessor of SU.
  void addPred(SIBlockSchedule *Pred);
  void addSucc(SIBlockSchedule *Succ);
  // InOrOutBlock: restrict to links pointing inside the block (true),
  // or restrict to links pointing outside the block (false).
  void releaseSuccessors(SUnit *SU, bool InOrOutBlock);

  std::vector<SIBlockSchedule*> Preds;  // All blocks predecessors.
  std::vector<SIBlockSchedule*> Succs;  // All blocks successors.

  unsigned NumPredsLeft;
  unsigned NumSuccsLeft;
  unsigned NumHighLatencySuccessors;

  bool isHighLatencyBlock() {return HighLatencyBlock;}
  // This is approximative.
  // Ideally should take into accounts some instructions (rcp, etc)
  // are 4 times slower
  int getCost() {return SUnits.size();}

  // The block Predecessors and Successors must be all registered
  // before schedule()
  // fast schedule with no particular requirement
  void fastSchedule();
  std::vector<SUnit*> getScheduledUnits() {return ScheduledSUnits;}

  // Complete schedule that will take try to minimize reg pressure and
  // low latencies, and will fill correct track of liveins and liveouts.
  // Needs all MIs to be groupped between BeginBlock and EndBlock.
  // The MIs can be moved after the scheduling.
  // It is just used to allow precise computation.
  void schedule(MachineBasicBlock::iterator BeginBlock,
                MachineBasicBlock::iterator EndBlock);
  bool isScheduled() {return Scheduled;}


  // Needs the block to be scheduled inside
  // TODO: find a way to compute it.
  std::vector<unsigned> &getInternalAdditionnalRegUsage() {
    return InternalAdditionnalPressure;
  }

  std::set<unsigned> &getInRegs() {return LiveInRegs;}
  std::set<unsigned> &getOutRegs() {return LiveOutRegs;}

  // How many blocks use the outputs registers
  std::map<unsigned, unsigned> LiveOutRegsNumUsages;

  // Store at which pos the last High latency parent was scheduled - Used only
  // during Block scheduling, not valid after
  unsigned LastPosHighLatencyParentScheduled;

  void printDebug(bool Full);

private:
  struct SISchedCandidate : SISchedulerCandidate {
    // The best SUnit candidate.
    SUnit *SU;

    unsigned SGPRUsage;
    unsigned VGPRUsage;
    bool IsLowLatency;
    unsigned LowLatencyOffset;
    bool HasLowLatencyNonWaitedParent;

    SISchedCandidate()
      : SU(nullptr) {}

    bool isValid() const { return SU; }

    // Copy the status of another candidate without changing policy.
    void setBest(SISchedCandidate &Best) {
      assert(Best.Reason != NoCand && "uninitialized Sched candidate");
      SU = Best.SU;
      Reason = Best.Reason;
      SGPRUsage = Best.SGPRUsage;
      VGPRUsage = Best.VGPRUsage;
      IsLowLatency = Best.IsLowLatency;
      LowLatencyOffset = Best.LowLatencyOffset;
      HasLowLatencyNonWaitedParent = Best.HasLowLatencyNonWaitedParent;
    }
  };

  void undoSchedule();
  void undoReleaseSucc(SUnit *SU, SDep *SuccEdge, bool InOrOutBlock);
  void releaseSucc(SUnit *SU, SDep *SuccEdge, bool InOrOutBlock);
  void NodeScheduled(SUnit *SU);
  void tryCandidateTopDown(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  void tryCandidateBottomUp(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  SUnit* pickNode();
  void traceCandidate(const SISchedCandidate &Cand);
  void initRegPressure(MachineBasicBlock::iterator BeginBlock,
                       MachineBasicBlock::iterator EndBlock);
};

class SIScheduleDAGMI : public ScheduleDAGMILive {
  const SIInstrInfo *SITII;
  const SIRegisterInfo *SITRI;

  unsigned SGPRsForWaveFronts[10];
  unsigned VGPRsForWaveFronts[10];

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
  unsigned NumBlockScheduled;
  std::set<unsigned> LiveRegs;
  // num of schedulable unscheduled blocks reading the register
  std::map<unsigned, unsigned> LiveRegsConsumers;
  std::vector<unsigned> BlockScheduleOrder;
  unsigned VregCurrentUsage;

  // for optimizing
  std::multiset<unsigned> VGPRUsages;
  std::multiset<unsigned> CostsHidingHighLatencies;

public:
  SIScheduleDAGMI(MachineSchedContext *C);

  ~SIScheduleDAGMI() override;

  // entry point for the schedule
  void schedule() override;

  struct SIBlockSchedCandidate : SISchedulerCandidate {
    // The best Block candidate.
    SIBlockSchedule *Block;

    bool IsHighLatency;
    int VGPRUsageDiff;
    int VGPRUsageNext;
    unsigned NumHighLatencySuccessors;
    unsigned LastPosHighLatParentScheduled;

    SIBlockSchedCandidate()
      : Block(nullptr) {}

    bool isValid() const { return Block; }

    // Copy the status of another candidate without changing policy.
    void setBest(SIBlockSchedCandidate &Best) {
      assert(Best.Reason != NoCand && "uninitialized Sched candidate");
      Block = Best.Block;
      Reason = Best.Reason;
      IsHighLatency = Best.IsHighLatency;
      VGPRUsageDiff = Best.VGPRUsageDiff;
      NumHighLatencySuccessors = Best.NumHighLatencySuccessors;
      LastPosHighLatParentScheduled = Best.LastPosHighLatParentScheduled;
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

  void propagateWaitedLatencies();

private:
  // To prepare the schedule
  void prepareSchedule();
  void topologicalSort();
  void createBlocks();
  void scheduleInsideBlocks();
  // After scheduling is done, improve low latency placements
  void moveLowLatencies();
  // To help the schedule, check register pressure change
  // by scheduling a block with these LiveIn and LiveOut
  std::vector<int> checkRegUsageImpact(std::set<unsigned> &InRegs,
                                       std::set<unsigned> &OutRegs);
  // reinits what needs to be.
  void prepareScheduleBlocks();
  void addLiveRegs(std::set<unsigned> &Regs);
  void decreaseLiveRegs(SIBlockSchedule *Block, std::set<unsigned> &Regs);
  void releaseBlockSuccs(SIBlockSchedule *Parent);
  void blockScheduled(SIBlockSchedule *Block);
  template<typename _Iterator> unsigned VgprCost(_Iterator First,
                                                 _Iterator End);
  // More Wavefronts = better latency hidding
  unsigned getWaveFrontsForUsage(unsigned SGPRsUsed, unsigned VGPRsUsed);

  void tryCandidate(SIBlockSchedCandidate &Cand,
                    SIBlockSchedCandidate &TryCand);
  void traceCandidate(const SIBlockSchedCandidate &Cand);
  SIBlockSchedule *pickBlock();
  void scheduleBlocks();

public:
  unsigned VGPRSetID;
  unsigned SGPRSetID;
  // some stats for scheduling inside blocks
  std::vector<unsigned> IsLowlatencySU;
  std::vector<unsigned> LowLatencyOffset;
  std::vector<unsigned> HasLowLatencyNonWaitedParent;
};

} // namespace llvm

#endif /* SIMACHINESCHEDULER_H_ */

