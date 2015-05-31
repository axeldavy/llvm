//===-- R600MachineScheduler.cpp - R600 Scheduler Interface -*- C++ -*-----===//
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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "misched"

// This scheduler is an adapted version of GenericScheduler
// to better fit SI needs

void SISchedStrategy::initialize(ScheduleDAGMI *dag) {
  int i;

  assert(dag->hasVRegLiveness() && "SiSchedStrategy needs vreg liveness");
  DAG = static_cast<ScheduleDAGMILive*>(dag);
  const AMDGPUSubtarget &ST = DAG->MF.getSubtarget<AMDGPUSubtarget>();
  TII = static_cast<const SIInstrInfo*>(DAG->TII);
  TRI = static_cast<const SIRegisterInfo*>(DAG->TRI);
  SchedModel = DAG->getSchedModel();

  Available.clear();
  for (i=0; i<10; i++) {
    SGPRsForWaveFronts[i] = TRI->getNumSGPRsAllowed(ST.getGeneration(), i+1);
    VGPRsForWaveFronts[i] = TRI->getNumVGPRsAllowed(i+1);
  }
  VGPRSetID = 25; // TODO: determine dynamically
  SGPRSetID = 12;
  CurrentWaveFronts = determineWaveFronts();
}

#ifndef NDEBUG
const char *SISchedStrategy::getReasonStr(SICandReason Reason) {
  switch (Reason) {
  case NoCand:         return "NOCAND    ";
  case SIWaveFronts:   return "WAVEFRONTS";
  case Cluster:        return "CLUSTER   ";
  case Weak:           return "WEAK      ";
  case NextDefUse:     return "DEF-USE   ";
  case ReadySuccessor: return "READYSUCC ";
  case NodeOrder:      return "ORDER     ";
  };
  llvm_unreachable("Unknown reason!");
}

void SISchedStrategy::traceCandidate(const SISchedCandidate &Cand) {

  dbgs() << "  SU(" << Cand.SU->NodeNum << ") " << getReasonStr(Cand.Reason);
  if (Cand.Reason == SIWaveFronts)
    dbgs() << " " << Cand.WaveFronts << " wavefronts ";
  else
    dbgs() << "          ";
  dbgs() << '\n';
}
#endif

static bool tryLess(int TryVal, int CandVal,
                    SISchedStrategy::SISchedCandidate &TryCand,
                    SISchedStrategy::SISchedCandidate &Cand,
                    SISchedStrategy::SICandReason Reason) {
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
                       SISchedStrategy::SISchedCandidate &TryCand,
                       SISchedStrategy::SISchedCandidate &Cand,
                       SISchedStrategy::SICandReason Reason) {
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

void SISchedStrategy::tryCandidate(SISchedCandidate &Cand,
                                   SISchedCandidate &TryCand) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return;
  }

  // Avoid decreasing wavefronts
  if (tryGreater(TryCand.WaveFronts, Cand.WaveFronts, TryCand, Cand, SIWaveFronts))
    return;

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  const SUnit *NextClusterSU = DAG->getNextClusterSucc();
  if (tryGreater(TryCand.SU == NextClusterSU, Cand.SU == NextClusterSU,
                 TryCand, Cand, Cluster))
    return;

  // Weak edges are for clustering and other constraints.
  if (tryLess(TryCand.SU->WeakPredsLeft,
              Cand.SU->WeakPredsLeft,
              TryCand, Cand, Weak)) {
    return;
  }

  // Prefer immediate defs/users of the last scheduled instruction. This is a
  // local pressure avoidance strategy that also makes the machine code
  // readable.
  //if (tryGreater(Zone.isNextSU(TryCand.SU), Zone.isNextSU(Cand.SU),
  //               TryCand, Cand, NextDefUse))
  //  return;

  // Prefer instructions whose result will directly enable new instructions
  if (tryGreater(TryCand.SuccessorReadiness, Cand.SuccessorReadiness,
                 TryCand, Cand, ReadySuccessor))
    return;

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum < Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
  }
}

SUnit* SISchedStrategy::pickNode(bool &IsTopNode) {
  ReadyQueue &Q = Available;
  SISchedCandidate Cand;
  SUnit *SU;

  DEBUG(Q.dump());

  if (DAG->top() == DAG->bottom()) {
    assert(Available.empty() && "ReadyQ garbage");
    return nullptr;
  }

  do {
    for (ReadyQueue::iterator I = Q.begin(), E = Q.end(); I != E; ++I) {
      SISchedCandidate TryCand;
      TryCand.SU = *I;
      TryCand.SuccessorReadiness = computeSuccessorReadiness(TryCand.SU );
      TryCand.WaveFronts = getWaveFrontsAfterInstruction(TryCand.SU);
      tryCandidate(Cand, TryCand);
      if (TryCand.Reason != NoCand) {
          Cand.setBest(TryCand);
          DEBUG(traceCandidate(Cand));
      }
    }
    SU = Cand.SU;

    if (!SU)
      SU = *Available.begin();
    IsTopNode = true;
  } while (SU->isScheduled);

  Available.remove(Available.find(SU));

  DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") " << *SU->getInstr());
  DEBUG(dbgs() << "Estimated next WaveFronts: " << getWaveFrontsAfterInstruction(SU) << "\n");
  DEBUG(dbgs() << "Ready Successor score: " << computeSuccessorReadiness(SU) << "\n");
  return SU;
}

void SISchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  assert(IsTopNode);

  CurrentWaveFronts = determineWaveFronts();
  DEBUG(dbgs() << "Estimated Wavefronts: " << CurrentWaveFronts << "\n");
}

void SISchedStrategy::registerRoots() {
}

unsigned SISchedStrategy::getWaveFrontsForUsage(unsigned SGPRsUsed,
                                                unsigned VGPRsUsed) {
  unsigned i;

  for (i = 9; i > 0; i--) {
    if (SGPRsForWaveFronts[i] >= SGPRsUsed &&
        VGPRsForWaveFronts[i] >= VGPRsUsed)
      break;
  }

  if (i+1 < CurrentWaveFronts)
    return i+1;
  return CurrentWaveFronts;
}

unsigned SISchedStrategy::getWaveFrontsAfterInstruction(SUnit *SU) {
  RegPressureTracker &Temp = const_cast<RegPressureTracker&>(DAG->getTopRPTracker());
  std::vector<unsigned> pressure;
  std::vector<unsigned> MaxPressure;
  Temp.getDownwardPressure(SU->getInstr(), pressure, MaxPressure);
  unsigned SGPRsUsed = pressure[SGPRSetID];
  unsigned VGPRsUsed = pressure[VGPRSetID];

  return getWaveFrontsForUsage(SGPRsUsed, VGPRsUsed);
}

unsigned SISchedStrategy::determineWaveFronts() {
  RegPressureTracker &Temp = const_cast<RegPressureTracker&>(DAG->getTopRPTracker());
  std::vector<unsigned> &pressure = Temp.getRegSetPressureAtPos();
  unsigned SGPRsUsed = pressure[SGPRSetID];
  unsigned VGPRsUsed = pressure[VGPRSetID];

  DEBUG(dbgs() << "Reg usage: SGPRs: " << SGPRsUsed << " VGPRs: " << VGPRsUsed << "\n");

  return getWaveFrontsForUsage(SGPRsUsed, VGPRsUsed);
}

unsigned SISchedStrategy::computeSuccessorReadiness(SUnit *SU) {
  unsigned numSuccessors = 0;
  unsigned numAlmostReadySuccs = 0;
  unsigned minPredecessorsSuccs = 32;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    SUnit *SuccSU = (*I).getSUnit();
    if ((*I).isWeak())
      continue;
    numSuccessors++;
    if (SuccSU->NumPredsLeft == 1)
      numAlmostReadySuccs++;
    if (SuccSU->NumPredsLeft < minPredecessorsSuccs)
      minPredecessorsSuccs = SuccSU->NumPredsLeft;
  }
  return numSuccessors + 10 * minPredecessorsSuccs + 100 * numAlmostReadySuccs;
}
