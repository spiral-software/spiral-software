
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(altivec_common, SIMD_VMX, rec(
    unparser := altivecUnparser,
    #compileStrategy := self >> IndicesCS2,
    useDeref := true,

    vadd    := "vec_add",
    vsub    := "vec_sub",
    vmul    := "vec_madd",
    vmuladd := true,

    autolib := rec(
       includes := () -> ["<vec_types.h>", "<altivec.h>", "<spu2vmx.h>" ],
       timerIncludes := () -> ["<include/sp_powerpc_timer.h>" ],
    ),

    backendConfig := rec(
                        profile := default_profiles.linux_altivec_gcc,
                        measureFunction := _StandardMeasureVerify
                     ),
));

Class(altivec_4x32f, altivec_common, rec(
    active := false,
    isFixedPoint := false,
    info := "altivec 4 x 32-bit float",
    v := 4,
    ctype := "float",
    vtype := "vector float",
    stype := "__attribute__ ((aligned(16))) float",
    instr := [vunpacklo_4x32f_av, vunpackhi_4x32f_av, vperm_4x32f, vuperm_4x32f],
    bits := 32,
    isFloat := true,
    isFix := false,
    includes := () -> ["<altivec.h>", "<omega32.h>"],
    infix_op := false,
    infix_assign := true,
    vzero   := TVect(TReal, 4).zero(),# vzero_4x32f,
    vconst := vconstpr_av,
    vconstv := "(vector float)",
    vconst1 := "(vector float)",
    splopts := rec(precision := "single"),

));


SIMD_ISA_DB.addISA(altivec_4x32f);
