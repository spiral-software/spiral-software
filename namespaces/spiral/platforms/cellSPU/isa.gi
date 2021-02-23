
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(spu_common, SIMD_VMX, rec(
    file := "cell",
    unparser := CellUnparser,
    #compileStrategy := self >> IndicesCS2_FMA,
    compileStrategy := self >> IndicesCS2,
    useDeref := true,

    vadd    := "spu_add",
    vsub    := "spu_sub",
    vmul    := "spu_mul",
    vlshift := "spu_sl",
    vrshift := "spu_sr",
    vbinand := "spu_and",

    autolib := rec(
       #includes := () -> ["<vec_types.h>", "<altivec.h>", "<spu2vmx.h>" ], # This is for PowerPC (PPE)
       includes := () -> ["<spu_intrinsics.h>", "<include/sp_general_malloc.h>",
                   "<simdmath.h>", "<simdmath/negated2.h>", "<simdmath/negatef4.h>"],
       timerIncludes := () -> ["<include/sp_spe_timer.h>" ],
    ),

    backendConfig := rec(
                        profile := default_profiles.linux_cellmultiSPU_gcc,
                        measureFunction := _StandardMeasureVerify
                     ),
));


Class(spu_4x32f, spu_common, rec(
    active := true,
    isFixedPoint := false,
    info := "SPU 4 x 32-bit float",
    v := 4,
    t := TVect(TReal, 4),
    globalUnrolling := 256,
    ctype := "float",
    vtype := "vector float",
    stype := "__attribute__ ((aligned(16))) float",
    instr := [ vperm_4x32f_spu, vuperm_4x32f_spu ],
    bits := 32,
    isFloat := true,
    isFix := false,
    includes := () -> ["<spu_intrinsics.h>", "<omega32.h>", "<simdmath.h>", "<simdmath/negated2.h>", "<simdmath/negatef4.h>"],
    infix_op := false,
    infix_assign := true,
    vmuladd := false,
    dupload := (y, x) -> assign(y, vuperm_4x32f_spu(promote_spu4x32f(x, 0), [1,1,1,1])),
    #duploadn takes a vector and replicates scalar number n in a vector
    duploadn := (y, x, n) -> assign(y, vuperm_4x32f_spu(x, [n,n,n,n])),
    vzero   := vzero_4x32f,
    vconst := vconstpr_av,
    vconstv := "(vector float)",
    vconst1 := "(vector float)spu_splats",
    splopts := rec(precision := "single"),
    countrec := rec( ops := [[add, sub], [mul], [fma, fms, nfma], [vperm_4x32f_spu]],
                     printstrings := ["[adds]", "[mults]", "[fmas]", "[vperms]"],
                     type := "TVect",
                     arithcost := (self, opcount) >> opcount[1]+opcount[2]+(2*opcount[3])
                     ),

    #NOTE:
    # - If operating on a NAN has no penalty, we shouldn't bother promoting and
    # shuffling to zero, for instance, where we could simply load directly.

    #NOTE:
    # - Find out if these are the fastest/most efficient (lowest register
    #   usage). For instance, how does a
    #   (promote,promote,perm,promote,promote,perm,perm) compare to a (promote,promote,perm,insert,insert)?
    # - Also, should we use temp variables here or not? (might help compiler propagation passes?)
    # - How to reuse loadc code here?

    #----------------------------------------
    #NOTE: Find out if ObjId(x)=nth should hold for all conditions. Doesn't seem to.
    #   (y,x) -> Checked(ObjId(x)=nth, assign(y, vperm_4x32f_spu(promote_spu4x32f(x, 0), vzero_spu(), [1, 128, 128, 128]))),
    svload := [[ # load using subvectors of length 1
              (y,x,opts) -> assign(y, vuperm_4x32f_spu(promote_spu4x32f(x[1], 0), [1, 128, 128, 128])),
              (y,x,opts) -> assign(y, vperm_4x32f_spu(promote_spu4x32f(x[1], 0), promote_spu4x32f(x[2], 0), [1,   5, 128, 128])),
              (y,x,opts) -> assign(y, insert_spu4x32f(x[3], vperm_4x32f_spu(promote_spu4x32f(x[1], 0), promote_spu4x32f(x[2], 0), [1,   5, 128, 128]), 2)),
              (y,x,opts) -> assign(y, insert_spu4x32f(x[4], insert_spu4x32f(x[3], insert_spu4x32f(x[2], promote_spu4x32f(x[1], 0), 1), 2), 3) ),
              ],
              [ # load using subvectors of length 2
              (y,x,opts) -> assign(y, vperm_4x32f_spu(promote_spu4x32f(x[1], 0), promote_spu4x32f(nth(x[1].loc, x[1].idx+1), 0), [1, 5, 128, 128])),
              (y,x,opts) -> assign(y, insert_spu4x32f(nth(x[2].loc, x[2].idx+1), insert_spu4x32f(x[2], insert_spu4x32f(nth(x[1].loc, x[1].idx+1), promote_spu4x32f(x[1], 0), 1), 2), 3) ),
              ]],
    #----------------------------------------
    #NOTE: loadu3 and loadu4 need work, prolly mostly in the unparser.
#     loadc := [      # load contiguous unaligned
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_4x32f_spu(promote_spu4x32f(x, 0), [1, 128, 128, 128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vperm_4x32f_spu(promote_spu4x32f(x, 0), promote_spu4x32f(nth(x.loc, x.idx+1), 0), [1, 5, 128, 128]))),

# #NOTE: This might be the best way to do it. Need to get it to work (past the ptr, ptr+1 problems).
# #(y,x) -> assign(y, vperm_4x32f_spu(bin_or(slqwbyte_spu4x32f(x.toPtr(TVect(TDouble, 4))), rlmaskqwbyte_spu4x32f(x.toPtr(TVect(TDouble, 4)))), vzero_spu(), [1,2,3,128])),
# #(y,x) -> assign(y, bin_or(slqwbyte_spu4x32f(x.toPtr(TVect(TDouble, 4))), rlmaskqwbyte_spu4x32f(x.toPtr(TVect(TDouble, 4))))),


# #NOTE: This is kind of a go-between for now. This bad because it's hacked via the unparser, but good because it only does 2 mem references
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_4x32f_spu(vloadu4_spu4x32f(x), [1, 2, 3, 128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vloadu4_spu4x32f(x)))
#     ],


###########################################################################
###### Vas, please check
###########################################################################

    # keep the n lower scalars and zero the other ones
    optional_mask :=  (c, n, opts) -> When(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and not(n=4),
        vuperm_4x32f_spu(c, List([1..4],x->When(x<=n, x, 128))),
        c),

    # load contiguous with unaligned loads
    loadc := (self, sv, opts) >> Cond(sv=1,
        ((y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_4x32f_spu(promote_spu4x32f(x, 0), [1, 128, 128, 128])))),
        sv=2,
        (y,x) -> Checked(ObjId(x)=nth, assign(y, vperm_4x32f_spu(promote_spu4x32f(x, 0), promote_spu4x32f(nth(x.loc, x.idx+1), 0), [1, 5, 128, 128]))),
#NOTE: loadu3 and loadu4 need work, prolly mostly in the unparser.
#NOTE: This is kind of a go-between for now. This bad because it's hacked via the unparser, but good because it only does 2 mem references
        sv=3,
        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_4x32f_spu(vloadu4_spu4x32f(x), [1, 2, 3, 128]))),
        sv=4,
        (y,x) -> Checked(ObjId(x)=nth, assign(y, vloadu4_spu4x32f(x))),
        Error("bad arguments for loadc")),

    #load contiguous + known alignment -> using 2 aligned load to be smarter
    loadc_align := (self, sv, align, opts)>>
    ((y,x,addr) -> let(
        v1 := nth(nth(x,add(addr,-align)).toPtr(TVect(TReal, 4)),0),
        v2 := nth(nth(x,add(addr,4-align)).toPtr(TVect(TReal, 4)), 0),
        m := x-> self.optional_mask(x, sv, opts),
        Cond(align=0,
            assign(y, m(v1)),
            assign(y, vperm_4x32f_spu(v1, v2, List([1..4], x-> When(x<=sv, align+x, 128))))
            ))),

###########################################################################
###### </Vas>
###########################################################################


    #----------------------------------------
    svstore := [[ # store using subvectors of length 1
            (y,x,opts) ->        assign(y[1], extract_spu4x32f(x, 0)),

            (y,x,opts) -> chain( assign(y[1], extract_spu4x32f(x, 0)),
                            assign(y[2], extract_spu4x32f(x, 1)) ),

            (y,x,opts) -> chain( assign(y[1], extract_spu4x32f(x, 0)),
                            assign(y[2], extract_spu4x32f(x, 1)),
                            assign(y[3], extract_spu4x32f(x, 2)) ),

            (y,x,opts) -> chain( assign(y[1], extract_spu4x32f(x, 0)),
                            assign(y[2], extract_spu4x32f(x, 1)),
                            assign(y[3], extract_spu4x32f(x, 2)),
                            assign(y[4], extract_spu4x32f(x, 3)) ),
              ],

              [ # store using subvectors of length 2
            (y,x,opts) -> chain( assign(y[1],                      extract_spu4x32f(x, 0)),
                            assign(nth(y[1].loc, y[1].idx+1), extract_spu4x32f(x, 1)) ),

            (y,x,opts) -> chain( assign(y[1],                      extract_spu4x32f(x, 0)),
                            assign(nth(y[1].loc, y[1].idx+1), extract_spu4x32f(x, 1)),
                            assign(y[2],                      extract_spu4x32f(x, 2)),
                            assign(nth(y[2].loc, y[2].idx+1), extract_spu4x32f(x, 3)) ),
              ]],
    #----------------------------------------
    storec := [    # store contiguous unaligned
            (y,x) ->        assign(y, extract_spu4x32f(x, 0)),
            (y,x) -> chain( assign(y,                   extract_spu4x32f(x, 0)),
                            assign(nth(y.loc, y.idx+1), extract_spu4x32f(x, 1)) ),
            (y,x) -> chain( assign(y,                   extract_spu4x32f(x, 0)),
                            assign(nth(y.loc, y.idx+1), extract_spu4x32f(x, 1)),
                            assign(nth(y.loc, y.idx+2), extract_spu4x32f(x, 2)) ),
            (y,x) -> chain( assign(y,                   extract_spu4x32f(x, 0)),
                            assign(nth(y.loc, y.idx+1), extract_spu4x32f(x, 1)),
                            assign(nth(y.loc, y.idx+2), extract_spu4x32f(x, 2)),
                            assign(nth(y.loc, y.idx+3), extract_spu4x32f(x, 3)) ),
        ],
    reverse := (y,x) -> assign(vref(y,0,4), vuperm_4x32f_spu(vref(x,0,4), [4, 3, 2, 1])),

    #----------------------------------------
    # support for VS and VS.transpose()

    shl1 :=  (y,x,opts) -> assign(y, vuperm_4x32f_spu(x, [128,1,2,3])),
    shl2 :=  (y,x,opts) -> assign(y, vperm_4x32f_spu(x[1], x[2], [4,5,6,7])),
    shr1 :=  (y,x,opts) -> assign(y, vuperm_4x32f_spu(x, [2, 3, 4, 128])),
    shr2 :=  (y,x,opts) -> assign(y, vperm_4x32f_spu(x[1], x[2], [2,3,4,5])),

    # support for VO1dsJ(n, v)
    shrev := (y,x,opts) -> assign(y, vperm_4x32f_spu(x[1], x[2], [1, 8, 7, 6])),

    swap_cx := (y, x, opts) -> assign(y, vuperm_2x64f_spu(x, [2,1,4,3])),
    RCVIxJ2 := (y, x, opts) -> assign(y, vuperm_2x64f_spu(x, [2,1,4,3]))

));

Class(spu_2x64f, spu_common, rec(
    active  := true,
    isFixedPoint := false,
    info    := "SPU 2 x 64-bit double",
    v       := 2,
    t := TVect(TReal, 2),
    globalUnrolling := 256,
    ctype   := "double",
    vtype   := "vector double",
    stype   := "__attribute__ ((aligned(16))) double",
    instr   := [ vperm_2x64f_spu, vuperm_2x64f_spu ],
    bits    := 64,
    isFloat := true,
    isFix := false,
    includes := () -> ["<spu_intrinsics.h>", "<omega32.h>"],
    infix_op := false,
    infix_assign := true,
    vmuladd := false,

    #NOTE: fix these
    #dupload := (y, x) -> assign(y, vuperm_4x32f_spu(promote_spu4x32f(x, 0), [1,1,1,1])),
    #duploadn takes a vector and replicates scalar number n in a vector
    #uploadn := (y, x, n) -> assign(y, vuperm_4x32f_spu(x, [n,n,n,n])),
    vzero   := vzero_2x64f,

    vconstv := "(vector double)",
    vconst1 := "(vector double)spu_splats",
    splopts := rec(precision := "double"),


    #NOTE: fix these
    #upload := (y, x) -> assign(y, vperm_2x64f_spu(promote_spu2x64f(x, 0), vzero_2x64f(), [1,1])),
    #uploadn := (y, x, n) -> assign(y, vuperm_2x64f_spu(x, [n,n])),

    # NOTE: Check if each of these are correct/efficient.
    svload := [[ # load using subvectors of length 1
              (y,x,opts) -> assign(y, vuperm_2x64f_spu(promote_spu2x64f(x[1], 0), [1, 128])),
              (y,x,opts) -> assign(y, vperm_2x64f_spu(promote_spu2x64f(x[1], 0), promote_spu2x64f(x[2], 0), [1,   3])),
              ],
              [ # load using subvectors of length 2
              (y,x,opts) -> assign(y, vperm_2x64f_spu(promote_spu2x64f(x[1], 0), promote_spu2x64f(nth(x[1].loc, x[1].idx+1), 0), [1, 3])),
              ]],

#     loadc := [      # load contiguous unaligned
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_2x64f_spu(promote_spu2x64f(x, 0), [1, 128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vperm_2x64f_spu(promote_spu2x64f(x, 0), promote_spu2x64f(nth(x.loc, x.idx+1), 0), [1, 3]))),
#        ],


###########################################################################
###### NOTE: Vas, please check
###########################################################################

    # keep the n lower scalars and zero the other ones
    optional_mask :=  (c, n, opts) -> When(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and not(n=2),
        vuperm_2x64f_spu(c, List([1..2],x->When(x<=n, x, 128))),
        c),

    # load contiguous with unaligned loads
    loadc := (self, sv, opts) >> Cond(sv=1,
        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_2x64f_spu(promote_spu2x64f(x, 0), [1, 128]))),
        sv=2,
        (y,x) -> Checked(ObjId(x)=nth, assign(y, vperm_2x64f_spu(promote_spu2x64f(x, 0), promote_spu2x64f(nth(x.loc, x.idx+1), 0), [1, 3]))),
        Error("bad arguments for loadc")),

    #load contiguous + known alignment -> using 2 aligned load to be smarter
    loadc_align := (self, sv, align, opts)>>
    ((y,x,addr) -> let(
        v1 := nth(nth(x,add(addr,-align)).toPtr(TVect(TReal, 2)),0),
        v2 := nth(nth(x,add(addr,2-align)).toPtr(TVect(TReal, 2)), 0),
        m := x-> self.optional_mask(x, sv, opts),
        Cond(align=0,
            assign(y, m(v1)),
            assign(y, vperm_2x64f_spu(v1, v2, List([1..2], x-> When(x<=sv, align+x, 128))))
            ))),

###########################################################################
###### </Vas>
###########################################################################


    svstore := [[ # store using subvectors of length 1
               (y,x,opts) ->        assign(y[1], extract_spu2x64f(x, 0)),

               (y,x,opts) -> chain( assign(y[1], extract_spu2x64f(x, 0)),
                               assign(y[2], extract_spu2x64f(x, 1)) ),
                ],
                [ # store using subvectors of length 2
                (y,x,opts) -> chain( assign(y[1],                      extract_spu2x64f(x, 0)),
                                assign(nth(y[1].loc, y[1].idx+1), extract_spu2x64f(x, 1)) ),
               ]],

    storec := [    # store contiguous unaligned
              (y,x) ->        assign(y, extract_spu2x64f(x, 0)),
              (y,x) -> chain( assign(y,                   extract_spu2x64f(x, 0)),
                              assign(nth(y.loc, y.idx+1), extract_spu2x64f(x, 1)) ),
              ],
    reverse := (y,x) -> assign(vref(y,0,2), vuperm_2x64f_spu(vref(x,0,2), [2, 1])),

    # Complex multiplication
    mul_cx := (self, opts) >> Cond(
               opts.vector.SIMD in ["SSE"],
               Error("opts.vector.SIMD is in SSE, while I am in SPU. You've got problems!"),
               (y, x, c) ->
                    let(u1 := var.fresh_t("U", TVectDouble(2)), u2 := var.fresh_t("U", TVectDouble(2)),
                        u3 := var.fresh_t("U", TVectDouble(2)), u4 := var.fresh_t("U", TVectDouble(2)),
                        decl([u1, u2, u3, u4], chain(
                            assign(u1, mul(x, vuperm_2x64f_spu(c, [1,1]))),
                            assign(u2, vuperm_2x64f_spu(x, [2,1])),
                            assign(u3, mul(u2, vuperm_2x64f_spu(c, [2,2]))),
                            assign(u4, vuperm_2x64f_spu(u3, [2,1])),
                            assign(y, add(u1, u4)))))
                ),

    #----------------------------------------
    # support for VS and VS.transpose()

    shl1 :=  (y,x,opts) -> assign(y, vperm_2x64f_spu(x[1], x[2], [2, 3])),
    shl2 :=  (y,x,opts) -> assign(y, vperm_2x64f_spu(x[1], x[2], [2, 128])),
    shr1 :=  (y,x,opts) -> assign(y, vperm_2x64f_spu(x[1], x[2], [4, 1])),
    shr2 :=  (y,x,opts) -> assign(y, vperm_2x64f_spu(x[1], x[2], [128, 1])),

    # support for VO1dsJ(n, v)
    shrev := (y,x,opts) -> assign(y, vperm_2x64f_spu(x[1], x[2], [1, 2])),

    swap_cx := (y, x, opts) -> assign(y, vuperm_2x64f_spu(x, [2,1])),
    RCVIxJ2 := (y, x, opts) -> assign(y, vuperm_2x64f_spu(x, [2,1]))

));

Class(ppu_4x32f, spu_4x32f, rec(
    includes := () -> ["<omega32.h>", "<vec_types.h>", "<altivec.h>", "<spu2vmx.h>"],
    backendConfig := rec(
                        profile := default_profiles.linux_cellPPU_gcc,
                        measureFunction := _StandardMeasureVerify
                     )
));

Class(spu_8x16i, spu_common, rec(
    active          := false,
    isFixedPoint    := true,
    saturatedArithmetic := true,
    info            := "SPU 8 x 16-bit int",
    v               := 8,
    ctype           := "short int",
    vtype           := "vector signed short",
    stype           := "__attribute__ ((aligned(16))) short int",
    # NOTE: What is svtype?
    svtype          := ["short int", "int", "signed long long", "signed long long"],
    instr           := [ vperm_8x16i_spu, vuperm_8x16i_spu ],
    bits            := 16,
    isFloat         := false,
    isFix           := true,
    includes        := () -> ["<include/omega16i.h>", "<include/mm_malloc.h>", "<spu_intrinsics.h>"],
    infix_op        := false,
    infix_assign    := true,
    vzero           := vzero_8x16i,
    vconstv         := "(vector signed short)",
    vconst1         := "(vector signed short)spu_splats",
    #NOTE: What is vconsthex? - it's to set 8 values. Value in unparser should take care of it?
    #vconsthex       := p->Chain(Print("_mm_set_epi16("), PrintCS(p), Print(")")),
    #NOTE: what are the following two?
    #vconst1init     := c -> PrintCS(List([1..8], i->c)),
    #vconst          := vconstprfp,
    splopts         := rec(customDataType := "signed short"),

    #NOTE: These REALLY need to be rewritten. They're correct, but end up doing
    #extra loads, shifts, rotates like nobody's business.
#     loadc := [      # load contiguous unaligned
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_8x16i_spu(promote_spu8x16i(x, 0), [1, 128, 128, 128, 128, 128, 128, 128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vperm_8x16i_spu(promote_spu8x16i(x, 0), promote_spu8x16i(nth(x.loc, x.idx+1), 0), [1,5,128,128,128,128,128,128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_8x16i_spu(vloadu8_spu8x16i(x), [1,2,3,128,128,128,128,128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_8x16i_spu(vloadu8_spu8x16i(x), [1,2,3,  4,128,128,128,128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_8x16i_spu(vloadu8_spu8x16i(x), [1,2,3,  4,  5,128,128,128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_8x16i_spu(vloadu8_spu8x16i(x), [1,2,3,  4,  5,  6,128,128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vuperm_8x16i_spu(vloadu8_spu8x16i(x), [1,2,3,  4,  5,  6,  7,128]))),
#        (y,x) -> Checked(ObjId(x)=nth, assign(y, vloadu8_spu8x16i(x)))
#     ],


###########################################################################
###### Vas, please check
###########################################################################

    # keep the n lower scalars and zero the other ones
    optional_mask :=  (c, n, opts) -> When(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and not(n=8),
        vuperm_8x16i_spu(c, List([1..8],x->When(x<=n, x, 128))),
        c),

    #NOTE: These REALLY need to be rewritten. They're correct, but end up doing
    #extra loads, shifts, rotates like nobody's business.
    # load contiguous with unaligned loads
    loadc := (self, sv, opts) >> ((y,x) -> assign(y, self.optional_mask(vloadu8_spu8x16i(x.toPtr(TVect(TReal, 8))), sv, opts))),

    #load contiguous + known alignment -> using 2 aligned load to be smarter
    loadc_align := (self, sv, align, opts)>>
    ((y,x,addr) -> let(
        v1 := nth(nth(x,add(addr,-align)).toPtr(TVect(TReal, 8)),0),
        v2 := nth(nth(x,add(addr,8-align)).toPtr(TVect(TReal, 8)), 0),
        m := x-> self.optional_mask(x, sv, opts),
        Cond(align=0,
            assign(y, m(v1)),
            assign(y, vperm_8x16i_spu(v1, v2, List([1..8], x-> When(x<=sv, align+x, 128))))
            ))),

###########################################################################
###### </Vas>
###########################################################################







    #NOTE: See if there's a better way to do this. Also look at assembly to see how good/bad this is.
    storec := [    # store contiguous unaligned
            (y,x) ->        assign(y, extract_spu8x16i(x, 0)),

            (y,x) -> chain( assign(y,                   extract_spu8x16i(x, 0)), assign(nth(y.loc, y.idx+1), extract_spu8x16i(x, 1))),

            (y,x) -> chain( assign(y,                   extract_spu8x16i(x, 0)), assign(nth(y.loc, y.idx+1), extract_spu8x16i(x, 1)),
                            assign(nth(y.loc, y.idx+2), extract_spu8x16i(x, 2))),

            (y,x) -> chain( assign(y,                   extract_spu8x16i(x, 0)), assign(nth(y.loc, y.idx+1), extract_spu8x16i(x, 1)),
                            assign(nth(y.loc, y.idx+2), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+3), extract_spu8x16i(x, 3))),

            (y,x) -> chain( assign(y,                   extract_spu8x16i(x, 0)), assign(nth(y.loc, y.idx+1), extract_spu8x16i(x, 1)),
                            assign(nth(y.loc, y.idx+2), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+3), extract_spu8x16i(x, 3)),
                            assign(nth(y.loc, y.idx+4), extract_spu8x16i(x, 2))),

            (y,x) -> chain( assign(y,                   extract_spu8x16i(x, 0)), assign(nth(y.loc, y.idx+1), extract_spu8x16i(x, 1)),
                            assign(nth(y.loc, y.idx+2), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+3), extract_spu8x16i(x, 3)),
                            assign(nth(y.loc, y.idx+4), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+5), extract_spu8x16i(x, 5))),

            (y,x) -> chain( assign(y,                   extract_spu8x16i(x, 0)), assign(nth(y.loc, y.idx+1), extract_spu8x16i(x, 1)),
                            assign(nth(y.loc, y.idx+2), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+3), extract_spu8x16i(x, 3)),
                            assign(nth(y.loc, y.idx+4), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+5), extract_spu8x16i(x, 5)),
                            assign(nth(y.loc, y.idx+6), extract_spu8x16i(x, 2))),

            (y,x) -> chain( assign(y,                   extract_spu8x16i(x, 0)), assign(nth(y.loc, y.idx+1), extract_spu8x16i(x, 1)),
                            assign(nth(y.loc, y.idx+2), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+3), extract_spu8x16i(x, 3)),
                            assign(nth(y.loc, y.idx+4), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+5), extract_spu8x16i(x, 5)),
                            assign(nth(y.loc, y.idx+6), extract_spu8x16i(x, 2)), assign(nth(y.loc, y.idx+7), extract_spu8x16i(x, 5))),
        ],
    reverse := (y,x) -> assign(vref(y,0,8), vuperm_8x16i_spu(vref(x,0,4), [8, 7, 6, 5, 4, 3, 2, 1]))
));

SIMD_ISA_DB.addISA(spu_4x32f);
SIMD_ISA_DB.addISA(spu_2x64f);
SIMD_ISA_DB.addISA(ppu_4x32f);
