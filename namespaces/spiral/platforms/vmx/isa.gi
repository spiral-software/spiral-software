
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(SIMD_VMX, SIMD_ISA, rec(
#    gen := self >> CellCGenSIMD,
    compileStrategy := self >> CellCompileStrategyVector,
#    hackUnparser := self >> CellVHackUnparser(VectorDefaults.expandConstants),

    #NOTE: unparser should point to SIMD_ISA's unparser
    unparser := SSEUnparser,
    arch := "AltiVec",
    info := "AltiVec, VMX, Cell BE PPU and SPU",
    file := "vmx"
));

Class(AltiVec_4x32f, SIMD_VMX, rec(
    active := true,
    isFixedPoint := false,
    info := "AltiVec 4 x 32-bit float",
    v := 4,
    ctype := "float",
    vtype := "vector float",
    stype := "__attribute__ ((aligned(16))) float",
    instr := [vunpacklo_4x32f_av, vunpackhi_4x32f_av, vperm_4x32f, vuperm_4x32f],
    #instr := [vunpacklo_4x32f_spu, vunpackhi_4x32f_spu, vperm_4x32f_spu, vuperm_4x32f_spu], # <- cheat-code!
    bits := 32,
    isFloat := true,
    isFix := false,
    header := "#include <altivec.h>\n\n",
    infix_op := false,
    infix_assign := true,
    vadd := "vec_add",
    vsub := "vec_sub",
    vmul := "vec_madd",
    vmuladd := true,
    vconst := vconstpr_av,
    vconstv := "(vector float)",
    vconst1 := "(vector float)",
    splopts := rec(precision := "single")
));


#Class(AltiVec_8x16i, rec(
#    active := false,
#    isFixedPoint := true,
#    info := "AltiVec 8 x 16-bit integer",
#    v := 8,
#    ctype := "int16",
#    vtype := "vector int16",
#    stype := "__attribute__ ((aligned(16))) int16",
#    instr := [vunpacklo_8x16i, vunpackhi_8x16i, vperm_8x16i, vuperm_8x16i],
#    bits := 16,
#    isFloat := true,
#    isFix := false,
#    header := "",
#    infix_op := false,
#    infix_assign := true,
#    vadd := "vec_add",
#    vsub := "vec_sub",
#    vmul := "vec_mul",
#    vconst := ""
#));
#
#Class(AltiVec_16x8i, rec(
#    active := false,
#    isFixedPoint := true,
#    info := "AltiVec 16 x 8-bit integer",
#    v := 16,
#    ctype := "int8",
#    vtype := "vector int8",
#    stype := "__attribute__ ((aligned(16))) int8",
#    instr := [vunpacklo_16x8i, vunpackhi_16x8i, vperm_16x8i, vuperm_16x8i],
#    bits := 8,
#    isFloat := true,
#    isFix := false,
#    header := "",
#    infix_op := false,
#    infix_assign := true,
#    vadd := "vec_add",
#    vsub := "vec_sub",
#    vmul := "vec_mul",
#    vconst := ""
#));

SIMD_ISA_DB.addISA(AltiVec_4x32f);
#SIMD_ISA_DB.addISA(AltiVec_8x16i);
#SIMD_ISA_DB.addISA(AltiVec_16x8i);
