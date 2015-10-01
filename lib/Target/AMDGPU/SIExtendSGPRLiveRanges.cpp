//===-- SIExtendSGPRLiveRanges.cpp - Extend SGPR live ranges ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file Extend the SGPRs live ranges

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-extend-sgpr-live-ranges"

namespace {

class SIExtendSGPRLiveRanges : public MachineFunctionPass {
public:
  static char ID;

public:
  SIExtendSGPRLiveRanges() : MachineFunctionPass(ID) {
    initializeSIExtendSGPRLiveRangesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Extend SGPR live ranges";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.setPreservesCFG();

    //AU.addPreserved<SlotIndexes>(); // XXX - This might be OK
    AU.addPreserved<LiveIntervals>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIExtendSGPRLiveRanges, DEBUG_TYPE,
                      "SI Extend SGPR Live Ranges", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveVariables)
INITIALIZE_PASS_END(SIExtendSGPRLiveRanges, DEBUG_TYPE,
                    "SI Extend SGPR Live Ranges", false, false)

char SIExtendSGPRLiveRanges::ID = 0;

char &llvm::SIExtendSGPRLiveRangesID = SIExtendSGPRLiveRanges::ID;

FunctionPass *llvm::createSIExtendSGPRLiveRangesPass() {
  return new SIExtendSGPRLiveRanges();
}

bool SIExtendSGPRLiveRanges::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const SIRegisterInfo *TRI = static_cast<const SIRegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  bool MadeChange = false;

  std::vector<std::pair<unsigned, LiveRange *>> SGPRLiveRanges;

  LiveIntervals *LIS = &getAnalysis<LiveIntervals>();
  LiveVariables *LV = getAnalysisIfAvailable<LiveVariables>();
  MachineBasicBlock *Entry = MF.begin();

  for (MachineBasicBlock *MBB : depth_first(Entry)) {
    for (const MachineInstr &MI : *MBB) {
      if (MI.getOpcode() != AMDGPU::S_BUFFER_LOAD_DWORD_IMM && MI.getOpcode() != AMDGPU::S_BUFFER_LOAD_DWORD_IMM_ci)
        continue;
      for (const MachineOperand &MO : MI.uses()) {
        if (!MO.isReg())
          continue;
        unsigned Use = MO.getReg();
        if (TargetRegisterInfo::isVirtualRegister(Use)) {
          if (TRI->isSGPRClass(MRI.getRegClass(Use))) {
            LiveRange &LR = LIS->getInterval(Use);
            SGPRLiveRanges.push_back(std::make_pair(Use, &LR));
          }
        }
      }
    }
  }
  {
    MachineBasicBlock &NCD = *MF.rbegin();

    for (std::pair<unsigned, LiveRange*> RegLR : SGPRLiveRanges) {
      unsigned Reg = RegLR.first;
      LiveRange *LR = RegLR.second;

      assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
             "Not expecting to extend live range of physreg");

      // FIXME: Need to figure out how to update LiveRange here so this pass
      // will be able to preserve LiveInterval analysis.
      MachineInstr *NCDSGPRUse =
        BuildMI(NCD, NCD.getFirstTerminator(), DebugLoc(),
                TII->get(AMDGPU::SGPR_USE))
        .addReg(Reg, RegState::Implicit);

      MadeChange = true;

      SlotIndex SI = LIS->InsertMachineInstrInMaps(NCDSGPRUse);
      LIS->extendToIndices(*LR, SI.getRegSlot());

      if (LV) {
        // TODO: This won't work post-SSA
        LV->HandleVirtRegUse(Reg, &NCD, NCDSGPRUse);
      }

      DEBUG(NCDSGPRUse->dump());
    }
  }

  return MadeChange;
}
