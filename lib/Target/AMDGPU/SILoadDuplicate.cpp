//===-- SILoadDuplicate.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass duplicates SMRD loads when the users are in different BB.
// This helps reduce spilling.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-load-duplicate"

namespace {

class SILoadDuplicate : public MachineFunctionPass {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  AliasAnalysis *AA;

  std::vector<MachineInstr *> MarkedForDeletion;

  void duplicateInBB(MachineFunction &MF, MachineInstr &MI, int UseBB);
  bool isDuplicatableLoad(MachineInstr &MI);
  bool duplicateLoad(MachineFunction &MF, MachineInstr &MI);


public:
  static char ID;

  SILoadDuplicate()
      : MachineFunctionPass(ID), TII(nullptr), TRI(nullptr), MRI(nullptr),
        AA(nullptr) {}

  SILoadDuplicate(const TargetMachine &TM_) : MachineFunctionPass(ID) {
    initializeSILoadDuplicatePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Load Duplicator";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SILoadDuplicate, DEBUG_TYPE,
                      "SI Load Duplicator", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(SILoadDuplicate, DEBUG_TYPE,
                    "SI Load Duplicator", false, false)

char SILoadDuplicate::ID = 0;

char &llvm::SILoadDuplicateID = SILoadDuplicate::ID;

FunctionPass *llvm::createSILoadDuplicatePass(TargetMachine &TM) {
  return new SILoadDuplicate(TM);
}

bool SILoadDuplicate::isDuplicatableLoad(MachineInstr &MI) {
  // TODO: Support more instructions (in particular when an
  // offset is stored in an sgpr, and then we read a buffer
  // at this offset, we should duplicate both instructions.
  return MI.getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORD_IMM &&
         MI.isInvariantLoad(AA);
}

// Replace all uses of the result of MI in UseBB
void SILoadDuplicate::duplicateInBB(MachineFunction &MF, MachineInstr &MI, int UseBB) {
  unsigned OldReg = MI.getOperand(0).getReg();
  unsigned NewReg = MRI->createVirtualRegister(MRI->getRegClass(OldReg));
  MachineBasicBlock *MBB;
  std::vector<MachineInstr *> Uses;

  // It is ok to clear more kill flags than needed (in particular
  // not adding the new ones that could get added by the change)
  // but we must not forget to clear the ones needed.
  // Clear the kill flags for the MI's output
  MRI->clearKillFlags(OldReg);
  // We will introduce a new user of 'MI.getOperand(1).getReg()',
  // thus clear its kill flag to enable using it later
  MRI->clearKillFlags(MI.getOperand(1).getReg());

  // Find all users in this block
  for (MachineInstr &User : MRI->use_instructions(OldReg)) {
    int UserBB = User.getParent()->getNumber();
    if (UserBB == UseBB)
      Uses.push_back(&User);
  }

  MBB = (*Uses.begin())->getParent();

  // Duplicate the instruction
  MachineInstr *NewMI = MF.CloneMachineInstr(&MI);
  NewMI->getOperand(0).setReg(NewReg);
  MBB->insert(MBB->getFirstNonPHI(), NewMI);

  // Replace the register for all the users in this block
  for (MachineInstr *User : Uses) {
    for (MachineOperand &Op : User->operands())
      if (Op.isReg() && Op.getReg() == OldReg)
        Op.setReg(NewReg);
  }

  // Remove the initial instruction if there is no
  // more users
  if (MRI->use_empty(OldReg))
    MarkedForDeletion.push_back(&MI);
}

bool SILoadDuplicate::duplicateLoad(MachineFunction &MF, MachineInstr &MI) {
  std::set<int> UseBBs;
  int LoadBB = MI.getParent()->getNumber();
  unsigned LoadReg = MI.getOperand(0).getReg();

  // For all users that are in different BB, duplicate the instruction.
  for (MachineInstr &User : MRI->use_instructions(LoadReg)) {
    int UserBB = User.getParent()->getNumber();
    if (UserBB != LoadBB) {
      DEBUG(dbgs() << "User: " << User << " from: " << MI  << "\n");
      UseBBs.insert(UserBB);
    }
  }
  if (UseBBs.empty())
    return false;

  for (int UseBB : UseBBs) {
    duplicateInBB(MF, MI, UseBB);
  }
  return true;
}

bool SILoadDuplicate::runOnMachineFunction(MachineFunction &MF) {
  const TargetSubtargetInfo &STM = MF.getSubtarget();
  TRI = static_cast<const SIRegisterInfo *>(STM.getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(STM.getInstrInfo());
  MRI = &MF.getRegInfo();

  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  DEBUG(dbgs() << "Running SILoadDuplicate\n");

  assert(MRI->isSSA());

  bool Modified = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (isDuplicatableLoad(MI))
        Modified |= duplicateLoad(MF, MI);
    }
  }

  for (MachineInstr *MI : MarkedForDeletion)
    MI->eraseFromParent();
  MarkedForDeletion.clear();

  return Modified;
}
