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
    NoCand,
    RegUsage,
    Latency,
    Successor,
    Depth,
    NodeOrder
};

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
class SIScheduleBlockCreator;

class SIScheduleBlock {
  SIScheduleDAGMI *DAG;
  SIScheduleBlockCreator *BC;

  std::vector<SUnit*> SUnits;
  std::map<unsigned, unsigned> NodeNum2Index;
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

  std::vector<unsigned> HasLowLatencyNonWaitedParent;

public:
  SIScheduleBlock(SIScheduleDAGMI *DAG, SIScheduleBlockCreator *BC,
                  unsigned ID):
    DAG(DAG), BC(BC), SUnits(), TopReadySUs(), ScheduledSUnits(),
    TopRPTracker(TopPressure), Scheduled(false),
    HighLatencyBlock(false), ID(ID),
    Preds(), Succs() {};

  ~SIScheduleBlock() {};

  // Unique ID, the index of the Block in the SIScheduleDAGMI Blocks table
  unsigned ID;

  /// functions for Block construction
  void addUnit(SUnit *SU);

  // when all SUs have been added
  void finalizeUnits();

  // Add block pred, which has instruction predecessor of SU.
  void addPred(SIScheduleBlock *Pred);
  void addSucc(SIScheduleBlock *Succ);

  std::vector<SIScheduleBlock*> Preds;  // All blocks predecessors.
  std::vector<SIScheduleBlock*> Succs;  // All blocks successors.

  unsigned NumHighLatencySuccessors = 0;

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

  // InOrOutBlock: restrict to links pointing inside the block (true),
  // or restrict to links pointing outside the block (false).
  void undoReleaseSucc(SUnit *SU, SDep *SuccEdge, bool InOrOutBlock);
  void releaseSucc(SUnit *SU, SDep *SuccEdge, bool InOrOutBlock);
  void releaseSuccessors(SUnit *SU, bool InOrOutBlock);

  void NodeScheduled(SUnit *SU);
  void tryCandidateTopDown(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  void tryCandidateBottomUp(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  SUnit* pickNode();
  void traceCandidate(const SISchedCandidate &Cand);
  void initRegPressure(MachineBasicBlock::iterator BeginBlock,
                       MachineBasicBlock::iterator EndBlock);
};

enum SISchedulerBlockCreatorVariant {
    LatenciesAlone
};

class SIScheduleBlockCreator {
  SIScheduleDAGMI *DAG;
  // unique_ptr handles freeing memory for us
  std::vector<std::unique_ptr<SIScheduleBlock>> BlockPtrs;
  std::map<SISchedulerBlockCreatorVariant,
           std::vector<SIScheduleBlock*>> Blocks;
  std::vector<SIScheduleBlock*> CurrentBlocks;
  std::vector<int> Node2CurrentBlock;

  /// Topological sort
  /// Index2Node - Maps topological index to the node number.
  std::vector<int> TopDownIndex2Node;
  std::vector<int> BottomUpIndex2Node;
  /// Node2Index - Maps the node number to its topological index.
  std::vector<int> TopDownNode2Index;
  std::vector<int> BottomUpNode2Index;

  // 0 -> color not given
  // 1 to SUnits.size() -> Reserved group (you should only add elements to them)
  // above -> other groups
  int NextReservedID;
  int NextNonReservedID;
  std::vector<int> CurrentColoring;
  std::vector<int> CurrentTopDownReservedDependencyColoring;
  std::vector<int> CurrentBottomUpReservedDependencyColoring;

public:
  SIScheduleBlockCreator(SIScheduleDAGMI *DAG);
  ~SIScheduleBlockCreator();

  std::vector<SIScheduleBlock*>
  getBlocks(SISchedulerBlockCreatorVariant BlockVariant);

  bool isSUInBlock(SUnit *SU, unsigned ID);

private:
  void topologicalSort();

  // Give a Reserved color to every high latency
  void colorHighLatenciesAlone();

  // Creates groups of high latencies with a Reserved color
  void colorHighLatenciesGroups();

  // Compute coloring for topdown and bottom traversals with
  // different colors depending on dependencies on Reserved colors
  void colorComputeReservedDependencies();

  // Gives Color to all non-colored SUs according to Reserved groups dependencies
  void colorAccordingToReservedDependencies();

  // Cut groups into groups with SUs in consecutive order (except for Reserved groups)
  void colorForceConsecutiveOrderInGroup();

  // Merge Constant loads that have all their users into another group to the group
  // (TODO: else if all their users depend on the same group, put them there)
  void colorMergeConstantLoadsNextGroup();

  // Merge SUs that have all their users into another group to the group
  void colorMergeIfPossibleNextGroup();

  // Merge SUs that have all their users into another group to the group, but only for Reserved groups
  void colorMergeIfPossibleNextGroupOnlyForReserved();

  void createBlocksForVariant(SISchedulerBlockCreatorVariant BlockVariant);

  void scheduleInsideBlocks();
};

enum SISchedulerBlockSchedulerVariant {
    BlockLatency
};

class SIScheduleBlockScheduler {
  SIScheduleDAGMI *DAG;
  SISchedulerBlockSchedulerVariant Variant;
  std::vector<SIScheduleBlock*> Blocks;

  std::vector<std::map<unsigned, unsigned>> LiveOutRegsNumUsages;
  std::set<unsigned> LiveRegs;
  // num of schedulable unscheduled blocks reading the register
  std::map<unsigned, unsigned> LiveRegsConsumers;

  std::vector<unsigned> LastPosHighLatencyParentScheduled;

  std::vector<SIScheduleBlock*> BlocksScheduled;
  unsigned NumBlockScheduled;
  std::vector<SIScheduleBlock*> ReadyBlocks;

  unsigned VregCurrentUsage;
  unsigned SregCurrentUsage;

  // Currently is only approximation,
  // it will contain the maximum register usage
  // between two blocks instead of the maximum
  // register usage in general
  unsigned maxVregUsage;
  unsigned maxSregUsage;

  std::vector<unsigned> BlockNumPredsLeft;
  std::vector<unsigned> BlockNumSuccsLeft;

public:
  SIScheduleBlockScheduler(SIScheduleDAGMI *DAG,
                           SISchedulerBlockSchedulerVariant Variant,
                           std::vector<SIScheduleBlock*> Blocks);
  ~SIScheduleBlockScheduler() {};

  std::vector<SIScheduleBlock*> getBlocks() {return BlocksScheduled;};

  unsigned getVGPRUsage() {return maxVregUsage;};
  unsigned getSGPRUsage() {return maxSregUsage;};

private:
  struct SIBlockSchedCandidate : SISchedulerCandidate {
    // The best Block candidate.
    SIScheduleBlock *Block;

    bool IsHighLatency;
    int VGPRUsageDiff;
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

  void tryCandidate(SIBlockSchedCandidate &Cand,
                    SIBlockSchedCandidate &TryCand);
  SIScheduleBlock *pickBlock();

  void addLiveRegs(std::set<unsigned> &Regs);
  void decreaseLiveRegs(SIScheduleBlock *Block, std::set<unsigned> &Regs);
  void releaseBlockSuccs(SIScheduleBlock *Parent);
  void blockScheduled(SIScheduleBlock *Block);

  // Check register pressure change
  // by scheduling a block with these LiveIn and LiveOut
  std::vector<int> checkRegUsageImpact(std::set<unsigned> &InRegs,
                                       std::set<unsigned> &OutRegs);

  void schedule();
};

class SIScheduler {
  SIScheduleDAGMI *DAG;
  SIScheduleBlockCreator BlockCreator;

public:
  SIScheduler(SIScheduleDAGMI *DAG) : DAG(DAG), BlockCreator(DAG) {};

  ~SIScheduler() {};

  std::vector<unsigned>
  scheduleVariant(SISchedulerBlockCreatorVariant BlockVariant,
                  SISchedulerBlockSchedulerVariant ScheduleVariant);
};

class SIScheduleDAGMI : public ScheduleDAGMILive {
  const SIInstrInfo *SITII;
  const SIRegisterInfo *SITRI;

  std::vector<SUnit> SUnitsLinksBackup;

  // For moveLowLatencies. After all Scheduling variants are tested.
  std::vector<unsigned> ScheduledSUnits;
  std::vector<unsigned> ScheduledSUnitsInv;

public:
  SIScheduleDAGMI(MachineSchedContext *C);

  ~SIScheduleDAGMI() override;

  // entry point for the schedule
  void schedule() override;

  // To init Block's RPTracker
  void initRPTracker(RegPressureTracker &RPTracker) {
    RPTracker.init(&MF, RegClassInfo, LIS, BB, RegionBegin);
  }

  MachineBasicBlock *getBB() {return BB;}
  MachineBasicBlock::iterator getCurrentTop() {return CurrentTop;};
  MachineBasicBlock::iterator getCurrentBottom() {return CurrentBottom;};
  LiveIntervals *getLIS() {return LIS;}
  MachineRegisterInfo *getMRI() {return &MRI;}
  const TargetRegisterInfo *getTRI() {return TRI;}
  SUnit& getEntrySU() {return EntrySU;};
  SUnit& getExitSU() {return ExitSU;};

  void restoreSULinksLeft();

  template<typename _Iterator> void fillVgprSgprCost(_Iterator First,
                                                     _Iterator End,
                                                     unsigned &VgprUsage,
                                                     unsigned &SgprUsage);
  std::set<unsigned> getInRegs() {
    std::set<unsigned> InRegs (RPTracker.getPressure().LiveInRegs.begin(),
                               RPTracker.getPressure().LiveInRegs.end());
    return InRegs;
  };

private:
  // After scheduling is done, improve low latency placements
  void moveLowLatencies();

public:
  unsigned VGPRSetID;
  unsigned SGPRSetID;
  // Some stats for scheduling inside blocks
  std::vector<unsigned> IsLowLatencySU;
  std::vector<unsigned> LowLatencyOffset;
  std::vector<unsigned> IsHighLatencySU;
};

} // namespace llvm

#endif /* SIMACHINESCHEDULER_H_ */

