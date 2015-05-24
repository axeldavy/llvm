//===-- SILoadBufferOptimizer.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass tries to fuse buffer load instructions with close by immediate offsets.
// This will fuse operations such as
//  s_buffer_load_dword s4 s[0:3] 0x2
//  s_buffer_load_dword s5 s[0:3] 0x3
// ==>
//  s_buffer_load_dwordx2 s[4:5] s[0:3] 0x2
//
// The pass works by steps:
//  - iterate over the instructions until finding a buffer load
//  - Form a consecutive group of 2, 4, 8 or 16 loads by looking at future instructions
//  - Do the merge
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-load-buffer-opt"

namespace {

class SILoadBufferOptimizer : public MachineFunctionPass {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

  MachineBasicBlock::iterator mergeLoads(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         unsigned Len);

  MachineBasicBlock::iterator optimizeLoad(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator I,
                                           bool *Modified);

  void updateRegDefsUses(unsigned SrcReg,
                         unsigned DstReg,
                         unsigned SubIdx);

  void setBufferSize(unsigned *Len,
                     unsigned *Opc,
                     TargetRegisterClass const **SuperRC);

public:
  static char ID;

  SILoadBufferOptimizer() :
    MachineFunctionPass(ID),
    TII(nullptr),
    TRI(nullptr),
    MRI(nullptr),
    LIS(nullptr) {

  }

  SILoadBufferOptimizer(const TargetMachine &TM_) : MachineFunctionPass(ID) {
    initializeSILoadBufferOptimizerPass(*PassRegistry::getPassRegistry());
  }

  bool optimizeBlock(MachineBasicBlock &MBB);

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Load Buffer Optimizer";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<SlotIndexes>();
    AU.addPreserved<LiveIntervals>();
    AU.addPreserved<LiveVariables>();
    AU.addRequired<LiveIntervals>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SILoadBufferOptimizer, DEBUG_TYPE,
                      "SI Load Buffer Optimizer", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_END(SILoadBufferOptimizer, DEBUG_TYPE,
                    "SI Load Buffer Optimizer", false, false)

char SILoadBufferOptimizer::ID = 0;

char &llvm::SILoadBufferOptimizerID = SILoadBufferOptimizer::ID;

FunctionPass *llvm::createSILoadBufferOptimizerPass(TargetMachine &TM) {
  return new SILoadBufferOptimizer(TM);
}

void SILoadBufferOptimizer::updateRegDefsUses(unsigned SrcReg,
                                             unsigned DstReg,
                                             unsigned SubIdx) {
  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(SrcReg),
         E = MRI->reg_end(); I != E; ) {
    MachineOperand &O = *I;
    ++I;
    O.substVirtReg(DstReg, SubIdx, *TRI);
  }
}

const unsigned SubForNum[16] = {
    AMDGPU::sub0,
    AMDGPU::sub1,
    AMDGPU::sub2,
    AMDGPU::sub3,
    AMDGPU::sub4,
    AMDGPU::sub5,
    AMDGPU::sub6,
    AMDGPU::sub7,
    AMDGPU::sub8,
    AMDGPU::sub9,
    AMDGPU::sub10,
    AMDGPU::sub11,
    AMDGPU::sub12,
    AMDGPU::sub13,
    AMDGPU::sub14,
    AMDGPU::sub15
};

void SILoadBufferOptimizer::setBufferSize(unsigned *Len,
                                          unsigned *Opc,
                                          TargetRegisterClass const **SuperRC)  {

  unsigned length = *Len;

  if (length >= 16) {
    *Len = 16;
    *Opc = AMDGPU::S_BUFFER_LOAD_DWORDX16_IMM;
    *SuperRC = &AMDGPU::SReg_512RegClass;
  } else if (length >= 8) {
    *Len = 8;
    *Opc = AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM;
    *SuperRC = &AMDGPU::SReg_256RegClass;
  } else if (length >= 4) {
    *Len = 4;
    *Opc = AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM;
    *SuperRC = &AMDGPU::SReg_128RegClass;
  } else {
    *Len = 2;
    *Opc = AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM;
    *SuperRC = &AMDGPU::SReg_64RegClass;
  }
}

MachineBasicBlock::iterator SILoadBufferOptimizer::mergeLoads(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned Len)
{
  int OffsetIdx = AMDGPU::getNamedOperandIdx(AMDGPU::S_BUFFER_LOAD_DWORD_IMM,
                                             AMDGPU::OpName::offset);
  unsigned n, Opc;
  MachineInstr &MI = *I;
  const MachineOperand *AddrReg = TII->getNamedOperand(MI, AMDGPU::OpName::sbase);
  const unsigned RefOffset = MI.getOperand(OffsetIdx).getImm() & 0xff;
  const TargetRegisterClass *SuperRC;

  MachineBasicBlock::iterator I2 = I;
  I2++;

  setBufferSize(&Len, &Opc, &SuperRC);

  unsigned DestReg = MRI->createVirtualRegister(SuperRC);
  const MCInstrDesc &Desc = TII->get(Opc);
  DebugLoc DL = I->getDebugLoc();
  unsigned DestReg0 = TII->getNamedOperand(*I, AMDGPU::OpName::dst)->getReg();
  MachineInstrBuilder Load_Buffer_2 = BuildMI(MBB, I, DL, Desc, DestReg)
                                        .addOperand(*AddrReg)
                                        .addImm(RefOffset);

  updateRegDefsUses(DestReg0, DestReg, AMDGPU::sub0);

  LIS->RemoveMachineInstrFromMaps(I);
  I->eraseFromParent();
  MachineBasicBlock::iterator Inext = I2;

  for (n = 1; n < Len; n++) {
    I2 = Inext;
    Inext++;
    unsigned DestReg1
        = TII->getNamedOperand(*I2, AMDGPU::OpName::dst)->getReg();
    updateRegDefsUses(DestReg1, DestReg, SubForNum[n]);
    LIS->ReplaceMachineInstrInMaps(I2, Load_Buffer_2);

    I2->eraseFromParent();
  }

  LiveInterval &AddrRegLI = LIS->getInterval(AddrReg->getReg());
  LIS->shrinkToUses(&AddrRegLI);

  LIS->getInterval(DestReg);

  return Load_Buffer_2.getInstr();
}

// TODO: do not restrict to S_BUFFER_LOAD_DWORD_IMM instructions

MachineBasicBlock::iterator SILoadBufferOptimizer::optimizeLoad(MachineBasicBlock &MBB, MachineBasicBlock::iterator I, bool *Modified) {
  unsigned lenSerie=1, currentOffset;
  int AddrIdx = AMDGPU::getNamedOperandIdx(AMDGPU::S_BUFFER_LOAD_DWORD_IMM,
                                           AMDGPU::OpName::sbase);
  int OffsetIdx = AMDGPU::getNamedOperandIdx(AMDGPU::S_BUFFER_LOAD_DWORD_IMM,
                                             AMDGPU::OpName::offset);
  MachineInstr &MI = *I;
  const MachineOperand &AddrReg0 = MI.getOperand(AddrIdx);
  const unsigned refOffset = MI.getOperand(OffsetIdx).getImm() & 0xff; /* imm offsets are 8 bits */
  MachineBasicBlock::iterator I2 = I;
  I2++;

  for (; I2 != MBB.end(); I2++) {
    MachineInstr &MI2 = *I2;
    unsigned Opc = MI2.getOpcode();
    const MachineOperand &AddrReg1 = MI2.getOperand(AddrIdx);

    /* the scheduler is asked to group consecutive loads. Stop
     * if it isn't a load or consecutive */
    if (Opc != AMDGPU::S_BUFFER_LOAD_DWORD_IMM)
        break;

    /* Check same base */
    if (AddrReg0.getReg() != AddrReg1.getReg() ||
        AddrReg0.getSubReg() != AddrReg1.getSubReg())
        break;

    currentOffset = MI2.getOperand(OffsetIdx).getImm() & 0xff;
    if (currentOffset != refOffset + lenSerie)
        break;
    lenSerie++;
  }
  if (lenSerie == 1)
     return ++I;

  *Modified = true;
  return mergeLoads(MBB, I, lenSerie);
}

bool SILoadBufferOptimizer::optimizeBlock(MachineBasicBlock &MBB) {
  bool Modified = false;

  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end();) {
    MachineInstr &MI = *I;
    unsigned Opc = MI.getOpcode();

    if (Opc != AMDGPU::S_BUFFER_LOAD_DWORD_IMM) {
      I++;
      continue;
    }

    I = optimizeLoad(MBB, I, &Modified);
  }

  return Modified;
}

bool SILoadBufferOptimizer::runOnMachineFunction(MachineFunction &MF) {
  const TargetSubtargetInfo &STM = MF.getSubtarget();
  TRI = static_cast<const SIRegisterInfo *>(STM.getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(STM.getInstrInfo());
  MRI = &MF.getRegInfo();

  LIS = &getAnalysis<LiveIntervals>();

  DEBUG(dbgs() << "Running SILoadBufferOptimizer\n");

  assert(!MRI->isSSA());

  bool Modified = false;

  for (MachineBasicBlock &MBB : MF)
    Modified |= optimizeBlock(MBB);

  return Modified;
}
