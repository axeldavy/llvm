; RUN: llc -march=amdgcn --misched=ilpmax -mcpu=SI -verify-machineinstrs < %s | FileCheck -strict-whitespace %s
; RUN: llc -march=amdgcn --misched=ilpmax -mcpu=tonga -verify-machineinstrs < %s | FileCheck -strict-whitespace %s
; we use ilpmax scheduler to have the second s_load be between the two buffer_load

; CHECK-LABEL: {{^}}main:
; CHECK: s_load_dwordx4
; CHECK: s_waitcnt lgkmcnt(0)
; CHECK: buffer_load
; CHECK: s_load_dwordx4
; CHECK: s_waitcnt lgkmcnt(0)
; CHECK: buffer_load
; CHECK: s_waitcnt vmcnt(1)
; CHECK: s_waitcnt vmcnt(0)
; CHECK: s_endpgm
define void @main([6 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* byval, [17 x <4 x i32>] addrspace(2)* byval, [34 x <8 x i32>] addrspace(2)* byval, [16 x <16 x i8>] addrspace(2)*
byval, i32 inreg, i32 inreg, i32, i32, i32, i32) #0 {
main_body:
  %11 = getelementptr [16 x <16 x i8>], [16 x <16 x i8>] addrspace(2)* %4, i64 0, i64 0
  %12 = load <16 x i8>, <16 x i8> addrspace(2)* %11, align 16, !tbaa !0
  %13 = add i32 %5, %7
  %14 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %12, i32 0, i32 %13)
  %15 = extractelement <4 x float> %14, i32 0
  %16 = extractelement <4 x float> %14, i32 1
  %17 = extractelement <4 x float> %14, i32 2
  %18 = extractelement <4 x float> %14, i32 3
  %19 = getelementptr [16 x <16 x i8>], [16 x <16 x i8>] addrspace(2)* %4, i64 0, i64 1
  %20 = load <16 x i8>, <16 x i8> addrspace(2)* %19, align 16, !tbaa !0
  %21 = add i32 %5, %7
  %22 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %20, i32 0, i32 %21)
  %23 = extractelement <4 x float> %22, i32 0
  %24 = extractelement <4 x float> %22, i32 1
  %25 = extractelement <4 x float> %22, i32 2
  %26 = extractelement <4 x float> %22, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %15, float %16, float %17, float %18)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float %23, float %24, float %25, float %26)
  ret void
}

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.vs.load.input(<16 x i8>, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="1" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", null}
