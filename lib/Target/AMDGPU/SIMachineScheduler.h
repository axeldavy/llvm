//===-- SIMachineScheduler.h - SI Scheduler Interface -----------*- C++ -*-===//
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/MC/LaneBitmask.h"
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace llvm {

class SISchedulerRPTracker {
  const MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  unsigned VGPRSetID, SGPRSetID;

  // Register -> Base of LaneBitmask to describe all possible
  // LaneBitmask used by Items Ins and Outs.
  // Except specified otherwise, RegisterMaskPairs in this class
  // are always having the LaneMask be one of the element of
  // LaneMaskBasisForReg[Reg]
  std::map<unsigned, SmallVector<LaneBitmask, 8>> LaneMaskBasisForReg;

  //   Static information about the Items:

  // Index of Item Successors or Predecessors.
  std::vector<SmallVector<unsigned, 8>> ItemSuccs;
  std::vector<SmallVector<unsigned, 8>> ItemPreds;

  std::vector<unsigned> TopoIndexToItem;
  std::vector<unsigned> TopoItemToIndex;

  // Blocks's getInRegs, but with RegisterMaskPair with Mask
  // in LaneMaskBasisForReg[Reg] if exists.
  std::vector<SmallVector<RegisterMaskPair, 8>> InRegsForItem;
  std::vector<SmallVector<RegisterMaskPair, 8>> OutRegsForItem;

  // Item num -> Number of usages for each Item output.
  std::vector<std::map<RegisterMaskPair, unsigned>> OutRegsNumUsages;

  //   Variable information during the schedule:

  // Items ready for schedule
  SmallVector<unsigned, 16> ReadyItems;

  // Current live regs, and the valid LaneBitmask
  std::map<unsigned, LaneBitmask> LiveRegs;

  // Number of schedulable unscheduled blocks reading the register.
  std::map<RegisterMaskPair, unsigned> RemainingRegsConsumers;

  std::vector<unsigned> ItemNumPredsLeft;

  unsigned CurrentVGPRUsage, CurrentSGPRUsage;

public:
  // Here the RegisterMaskPair can have arbitrary LaneMask (not
  // elements of LaneMaskBasisForReg, which is not yet built).
  SISchedulerRPTracker(
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
  );

  void itemScheduled(unsigned ID);

  const SmallVector<unsigned, 16> &getReadyItems() {
    return ReadyItems;
  }

  void getCurrentRegUsage(unsigned &VGPR, unsigned &SGPR);
  // Check register pressure change
  // by scheduling a item
  void checkRegUsageImpact(unsigned ID, int &DiffVGPR, int &DiffSGPR);

  void printDebugLives();

  std::set<unsigned> findPathRegUsage(int SearchDepthLimit,
                                      int VGPRDiffGoal,
                                      int SGPRDiffGoal,
                                      bool PriorityVGPR);

private:
  // Convert Reg/Mask to a list of Reg/Mask, with Mask in
  // LaneMaskBasisForReg.
  SmallVector<RegisterMaskPair, 8> getPairsForReg(unsigned Reg,
                                                  LaneBitmask Mask);
  // ToAppend: where to append the result.
  void getPairsForReg(SmallVector<RegisterMaskPair, 8> &ToAppend,
                      unsigned Reg, LaneBitmask Mask);
  // Idem for a list of Reg/Mask
  SmallVector<RegisterMaskPair, 8> getPairsForRegs(
    const SmallVectorImpl<RegisterMaskPair> &Regs);

  void fillTopoData();

  void addLiveRegs(const SmallVectorImpl<RegisterMaskPair> &Regs);
  void decreaseLiveRegs(const SmallVectorImpl<RegisterMaskPair> &Regs);
  void releaseItemSuccs(unsigned ID);
};

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
  SIScheduleCandReason Reason = NoCand;

  // Set of reasons that apply to multiple candidates.
  uint32_t RepeatReasonSet = 0;

  SISchedulerCandidate() = default;

  bool isRepeat(SIScheduleCandReason R) { return RepeatReasonSet & (1 << R); }
  void setRepeat(SIScheduleCandReason R) { RepeatReasonSet |= (1 << R); }
};

class SIScheduleDAGMI;
class SIScheduleBlockCreator;

enum SIScheduleBlockLinkKind {
  NoData,
  Data
};

class SIScheduleBlock {
  SIScheduleDAGMI *DAG;
  SIScheduleBlockCreator *BC;

  std::unique_ptr<SISchedulerRPTracker> RPTracker;

  std::vector<SUnit*> SUnits;
  std::map<unsigned, unsigned> NodeNum2Index;
  std::vector<SUnit*> ScheduledSUnits;

  // Pressure: number of said class of registers needed to
  // store the live virtual and real registers.
  // We do care only of SGPR32 and VGPR32 and do track only virtual registers.
  // Pressure of additional registers required inside the block.
  std::vector<unsigned> InternalAdditionnalPressure;
  // Pressure of input and output registers
  unsigned LiveInVGPRPressure;
  unsigned LiveInSGPRPressure;
  unsigned LiveOutVGPRPressure;
  unsigned LiveOutSGPRPressure;
  // Registers required by the block, and outputs.
  // We do track only virtual registers.
  // Note that some registers are not 32 bits,
  // and thus the pressure is not equal
  // to the number of live registers.
  SmallVector<RegisterMaskPair, 8> LiveInRegs;
  SmallVector<RegisterMaskPair, 8> LiveOutRegs;

  bool HighLatencyBlock = false;

  std::vector<unsigned> HasLowLatencyNonWaitedParent;

  // Unique ID, the index of the Block in the SIScheduleDAGMI Blocks table.
  unsigned ID;

  std::vector<SIScheduleBlock*> Preds;  // All blocks predecessors.
  // All blocks successors, and the kind of link
  std::vector<std::pair<SIScheduleBlock*, SIScheduleBlockLinkKind>> Succs;
  unsigned NumHighLatencySuccessors = 0;

public:
  SIScheduleBlock(SIScheduleDAGMI *DAG, SIScheduleBlockCreator *BC,
                  unsigned ID):
    DAG(DAG), BC(BC), LiveInVGPRPressure(0), LiveInSGPRPressure(0),
    LiveOutVGPRPressure(0), LiveOutSGPRPressure(0), LiveInRegs(),
    LiveOutRegs(), ID(ID) {}

  ~SIScheduleBlock() = default;

  unsigned getID() const { return ID; }

  /// Functions for Block construction.
  void addUnit(SUnit *SU);

  // When all SUs have been added, and liveIns/Outs computed.
  void finalize();

  // Add block pred, which has instruction predecessor of SU.
  void addPred(SIScheduleBlock *Pred);
  void addSucc(SIScheduleBlock *Succ, SIScheduleBlockLinkKind Kind);
  void addLiveIns(SmallVector<RegisterMaskPair, 8> Ins);
  void addLiveOuts(SmallVector<RegisterMaskPair, 8> Outs);

  const std::vector<SIScheduleBlock*>& getPreds() const { return Preds; }
  ArrayRef<std::pair<SIScheduleBlock*, SIScheduleBlockLinkKind>>
    getSuccs() const { return Succs; }

  unsigned Height;  // Maximum topdown path length to block without outputs
  unsigned Depth;   // Maximum bottomup path length to block without inputs

  unsigned getNumHighLatencySuccessors() const {
    return NumHighLatencySuccessors;
  }

  bool isHighLatencyBlock() { return HighLatencyBlock; }

  // This is approximative.
  // Ideally should take into accounts some instructions (rcp, etc)
  // are 4 times slower.
  int getCost() { return SUnits.size(); }

  std::vector<SUnit*> getScheduledUnits() { return ScheduledSUnits; }

  // Needs the block to be scheduled inside
  // TODO: find a way to compute it.
  std::vector<unsigned> &getInternalAdditionnalRegUsage() {
    return InternalAdditionnalPressure;
  }

  const SmallVector<RegisterMaskPair, 8> &getInRegs() const {
    return LiveInRegs;
  }

  const SmallVector<RegisterMaskPair, 8> &getOutRegs() const {
    return LiveOutRegs;
  }

  void printDebug(bool Full);

private:
  struct SISchedCandidate : SISchedulerCandidate {
    // The best SUnit candidate.
    SUnit *SU = nullptr;

    unsigned SGPRUsage;
    unsigned VGPRUsage;
    bool IsLowLatency;
    unsigned LowLatencyOffset;
    bool HasLowLatencyNonWaitedParent;

    SISchedCandidate() = default;

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

  void nodeScheduled(SUnit *SU);
  void tryCandidateTopDown(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  void tryCandidateBottomUp(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  SUnit* pickNode();
  void traceCandidate(const SISchedCandidate &Cand);
  void schedule();
};

struct SIScheduleBlocks {
  std::vector<SIScheduleBlock*> Blocks;
  std::vector<int> TopDownIndex2Block;
  std::vector<int> TopDownBlock2Index;
};

enum SISchedulerBlockCreatorVariant {
  LatenciesAlone,
  LatenciesGrouped,
  LatenciesAlonePlusConsecutive
};

class SIScheduleBlockCreator {
  SIScheduleDAGMI *DAG;
  // unique_ptr handles freeing memory for us.
  std::vector<std::unique_ptr<SIScheduleBlock>> BlockPtrs;
  std::map<SISchedulerBlockCreatorVariant,
           SIScheduleBlocks> Blocks;
  std::vector<SIScheduleBlock*> CurrentBlocks;
  std::vector<int> Node2CurrentBlock;

  // Topological sort
  // Maps topological index to the node number.
  std::vector<int> TopDownIndex2Block;
  std::vector<int> TopDownBlock2Index;
  std::vector<int> BottomUpIndex2Block;

  // 0 -> Color not given.
  // 1 to SUnits.size() -> Reserved group (you should only add elements to them).
  // Above -> Other groups.
  int NextReservedID;
  int NextNonReservedID;
  std::vector<int> CurrentColoring;
  std::vector<int> CurrentTopDownReservedDependencyColoring;
  std::vector<int> CurrentBottomUpReservedDependencyColoring;

public:
  SIScheduleBlockCreator(SIScheduleDAGMI *DAG);
  ~SIScheduleBlockCreator();

  SIScheduleBlocks
  getBlocks(SISchedulerBlockCreatorVariant BlockVariant);

  bool isSUInBlock(SUnit *SU, unsigned ID);

private:
  // Give a Reserved color to every high latency.
  void colorHighLatenciesAlone();

  // Create groups of high latencies with a Reserved color.
  void colorHighLatenciesGroups();

  // Compute coloring for topdown and bottom traversals with
  // different colors depending on dependencies on Reserved colors.
  void colorComputeReservedDependencies();

  // Give color to all non-colored SUs according to Reserved groups dependencies.
  void colorAccordingToReservedDependencies();

  // Divides Blocks having no bottom up or top down dependencies on Reserved groups.
  // The new colors are computed according to the dependencies on the other blocks
  // formed with colorAccordingToReservedDependencies.
  void colorEndsAccordingToDependencies();

  // Cut groups into groups with SUs in consecutive order (except for Reserved groups).
  void colorForceConsecutiveOrderInGroup();

  // Merge Constant loads that have all their users into another group to the group.
  // (TODO: else if all their users depend on the same group, put them there)
  void colorMergeConstantLoadsNextGroup();

  // Merge SUs that have all their users into another group to the group
  void colorMergeIfPossibleNextGroup();

  // Merge SUs that have all their users into another group to the group,
  // but only for Reserved groups.
  void colorMergeIfPossibleNextGroupOnlyForReserved();

  // Merge SUs that have all their users into another group to the group,
  // but only if the group is no more than a few SUs.
  void colorMergeIfPossibleSmallGroupsToNextGroup();

  // Divides Blocks with important size.
  // Idea of implementation: attribute new colors depending on topdown and
  // bottom up links to other blocks.
  void cutHugeBlocks();

  // Put in one group all instructions with no users in this scheduling region
  // (we'd want these groups be at the end).
  void regroupNoUserInstructions();

  void createBlocksForVariant(SISchedulerBlockCreatorVariant BlockVariant);

  void topologicalSort();

  void fillStats();

  LaneBitmask getLaneBitmaskForDef(const SUnit *SU, unsigned Reg);
  LaneBitmask getLaneBitmaskForUse(const SUnit *SU, unsigned Reg);
  void removeUseFromDef(SmallVectorImpl<RegisterMaskPair> &Uses,
                        unsigned Reg, const SUnit *SU);
  void addDefFromUse(SmallVectorImpl<RegisterMaskPair> &Defs,
                     unsigned Reg, const SUnit *SUDef, const SUnit *SUUse);
};

enum SISchedulerBlockSchedulerVariant {
  BlockLatencyRegUsage,
  BlockRegUsageLatency,
  BlockRegUsage
};

class SIScheduleBlockScheduler {
  SIScheduleDAGMI *DAG;
  SISchedulerBlockSchedulerVariant Variant;
  std::vector<SIScheduleBlock*> Blocks;

  std::unique_ptr<SISchedulerRPTracker> RPTracker;

  std::vector<unsigned> LastPosHighLatencyParentScheduled;
  int LastPosWaitedHighLatency;

  std::set<unsigned> CurrentPathRegUsage;

  std::vector<SIScheduleBlock*> BlocksScheduled;
  unsigned NumBlockScheduled;

  // Currently is only approximation.
  unsigned maxVregUsage;
  unsigned maxSregUsage;

public:
  SIScheduleBlockScheduler(SIScheduleDAGMI *DAG,
                           SISchedulerBlockSchedulerVariant Variant,
                           SIScheduleBlocks BlocksStruct);
  ~SIScheduleBlockScheduler() = default;

  std::vector<SIScheduleBlock*> getBlocks() { return BlocksScheduled; }

  unsigned getVGPRUsage() { return maxVregUsage; }
  unsigned getSGPRUsage() { return maxSregUsage; }

private:

  struct SIBlockSchedCandidate : SISchedulerCandidate {
    // The best Block candidate.
    SIScheduleBlock *Block = nullptr;

    bool IsHighLatency;
    int VGPRUsageDiff;
    unsigned NumSuccessors;
    unsigned NumHighLatencySuccessors;
    unsigned LastPosHighLatParentScheduled;
    unsigned Height;

    SIBlockSchedCandidate() = default;

    bool isValid() const { return Block; }

    // Copy the status of another candidate without changing policy.
    void setBest(SIBlockSchedCandidate &Best) {
      assert(Best.Reason != NoCand && "uninitialized Sched candidate");
      Block = Best.Block;
      Reason = Best.Reason;
      IsHighLatency = Best.IsHighLatency;
      VGPRUsageDiff = Best.VGPRUsageDiff;
      NumSuccessors = Best.NumSuccessors;
      NumHighLatencySuccessors = Best.NumHighLatencySuccessors;
      LastPosHighLatParentScheduled = Best.LastPosHighLatParentScheduled;
      Height = Best.Height;
    }
  };

  bool tryCandidateLatency(SIBlockSchedCandidate &Cand,
                           SIBlockSchedCandidate &TryCand);
  bool tryCandidateRegUsage(SIBlockSchedCandidate &Cand,
                            SIBlockSchedCandidate &TryCand);
  SIScheduleBlock *pickBlock();

  void blockScheduled(SIScheduleBlock *Block);

  void schedule();
};

struct SIScheduleBlockResult {
  std::vector<unsigned> SUs;
  unsigned MaxSGPRUsage;
  unsigned MaxVGPRUsage;
};

class SIScheduler {
  SIScheduleDAGMI *DAG;
  SIScheduleBlockCreator BlockCreator;

public:
  SIScheduler(SIScheduleDAGMI *DAG) : DAG(DAG), BlockCreator(DAG) {}

  ~SIScheduler() = default;

  struct SIScheduleBlockResult
  scheduleVariant(SISchedulerBlockCreatorVariant BlockVariant,
                  SISchedulerBlockSchedulerVariant ScheduleVariant);
};

class SIScheduleDAGMI final : public ScheduleDAGMILive {
  const SIInstrInfo *SITII;
  const SIRegisterInfo *SITRI;

  std::vector<SUnit> SUnitsLinksBackup;

  // For moveLowLatencies. After all Scheduling variants are tested.
  std::vector<unsigned> ScheduledSUnits;
  std::vector<unsigned> ScheduledSUnitsInv;

  unsigned VGPRSetID;
  unsigned SGPRSetID;

public:
  SIScheduleDAGMI(MachineSchedContext *C);

  ~SIScheduleDAGMI() override;

  // Entry point for the schedule.
  void schedule() override;

  MachineBasicBlock *getBB() { return BB; }
  MachineBasicBlock::iterator getCurrentTop() { return CurrentTop; }
  MachineBasicBlock::iterator getCurrentBottom() { return CurrentBottom; }
  LiveIntervals *getLIS() { return LIS; }
  MachineRegisterInfo *getMRI() { return &MRI; }
  const TargetRegisterInfo *getTRI() { return TRI; }
  ScheduleDAGTopologicalSort *getTopo() { return &Topo; }
  SUnit& getEntrySU() { return EntrySU; }
  SUnit& getExitSU() { return ExitSU; }

  const SmallVector<RegisterMaskPair, 8> &getInRegs() const {
    return RPTracker.getPressure().LiveInRegs;
  }

  const SmallVector<RegisterMaskPair, 8> &getOutRegs() const {
    return RPTracker.getPressure().LiveOutRegs;
  };

  unsigned getVGPRSetID() const { return VGPRSetID; }
  unsigned getSGPRSetID() const { return SGPRSetID; }

  bool shouldTrackLaneMasks() const { return ShouldTrackLaneMasks; }

private:
  void topologicalSort();
  // After scheduling is done, improve low latency placements.
  void moveLowLatencies();

public:
  // Some stats for scheduling inside blocks.
  std::vector<unsigned> IsLowLatencySU;
  std::vector<unsigned> LowLatencyOffset;
  std::vector<unsigned> IsHighLatencySU;
  // Topological sort
  // Maps topological index to the node number.
  std::vector<int> TopDownIndex2SU;
  std::vector<int> BottomUpIndex2SU;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SIMACHINESCHEDULER_H
