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
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace llvm {

class SISchedStrategy : public MachineSchedStrategy {

  ScheduleDAGMILive *DAG;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  const MachineSchedContext *Context;
  const TargetSchedModel *SchedModel;
  ReadyQueue Available;

  unsigned CurrentWaveFronts;
  unsigned SGPRsForWaveFronts[10];
  unsigned VGPRsForWaveFronts[10];
  unsigned VGPRSetID;
  unsigned SGPRSetID;

public:
  SISchedStrategy() :
    DAG(nullptr), TII(nullptr), TRI(nullptr), MRI(nullptr),
    Available(1, "TopQA") {}

  virtual ~SISchedStrategy() {}

  void initialize(ScheduleDAGMI *dag) override;

  SUnit *pickNode(bool &IsTopNode) override;

  void schedNode(SUnit *SU, bool IsTopNode) override;

  void releaseTopNode(SUnit *SU) override {
    Available.push(SU);
  }

  void releaseBottomNode(SUnit *SU) override {
  }

  void registerRoots() override;

  enum SICandReason {
    NoCand, SIWaveFronts, Cluster, Weak, NextDefUse, ReadySuccessor, NodeOrder};

  struct SISchedCandidate {
    // The best SUnit candidate.
    SUnit *SU;

    // The reason for this candidate.
    SICandReason Reason;

    // Set of reasons that apply to multiple candidates.
    uint32_t RepeatReasonSet;

    // WaveFronts estimated if the best candidate is scheduled
    unsigned WaveFronts;

    unsigned SuccessorReadiness;

    SISchedCandidate()
      : SU(nullptr), Reason(NoCand), RepeatReasonSet(0) {}

    bool isValid() const { return SU; }

    // Copy the status of another candidate without changing policy.
    void setBest(SISchedCandidate &Best) {
      assert(Best.Reason != NoCand && "uninitialized Sched candidate");
      SU = Best.SU;
      Reason = Best.Reason;
      WaveFronts = Best.WaveFronts;
      SuccessorReadiness = Best.SuccessorReadiness;
    }

    bool isRepeat(SICandReason R) { return RepeatReasonSet & (1 << R); }
    void setRepeat(SICandReason R) { RepeatReasonSet |= (1 << R); }
  };

private:
  void tryCandidate(SISchedCandidate &Cand, SISchedCandidate &TryCand);
  unsigned getWaveFrontsForUsage(unsigned SGPRsUsed, unsigned VGPRsUsed);
  unsigned getWaveFrontsAfterInstruction(SUnit *SU);
  unsigned determineWaveFronts();
  unsigned computeSuccessorReadiness(SUnit *SU);

  const char *getReasonStr(SICandReason Reason);
  void traceCandidate(const SISchedCandidate &Cand);
};

} // namespace llvm

#endif /* SIMACHINESCHEDULER_H_ */
 
