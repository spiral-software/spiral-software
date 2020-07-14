
# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details


# Below we define includes mandated by SSE Intel C+ compiler
# Due to known incompatibilities between gcc and icc, you can turn
# off hasXXX if compiler doesnt support it
# No crosscompilation support. 

_hasSSE4_2 := arg -> LocalConfig.cpuinfo.SIMD().hasSSE4_2() and  LocalConfig.compilerinfo.SIMD().hasSSE4_2();
_hasSSE4_1 := arg -> LocalConfig.cpuinfo.SIMD().hasSSE4_1() and  LocalConfig.compilerinfo.SIMD().hasSSE4_1();
_hasSSSE3  := arg -> LocalConfig.cpuinfo.SIMD().hasSSSE3()  and  LocalConfig.compilerinfo.SIMD().hasSSSE3();
_hasSSE3   := arg -> LocalConfig.cpuinfo.SIMD().hasSSE3()   and  LocalConfig.compilerinfo.SIMD().hasSSE3();
_hasSSE2   := arg -> LocalConfig.cpuinfo.SIMD().hasSSE2()   and  LocalConfig.compilerinfo.SIMD().hasSSE2();
_hasSSE    := arg -> LocalConfig.cpuinfo.SIMD().hasSSE()    and  LocalConfig.compilerinfo.SIMD().hasSSE();
_hasMMX    := arg -> LocalConfig.cpuinfo.SIMD().hasMMX()    and  LocalConfig.compilerinfo.SIMD().hasMMX();

_MM_MALLOC := () -> When(not LocalConfig.osinfo.isDarwin(), ["<include/mm_malloc.h>"], []);
_MMINTRIN  := () -> When(_hasMMX(),    ["<mmintrin.h>"], []);
_XMMINTRIN := () -> When(_hasSSE(),    ["<xmmintrin.h>"], []);
_EMMINTRIN := () -> When(_hasSSE2(),   ["<emmintrin.h>"], []);
_PMMINTRIN := () -> When(_hasSSE3(),   ["<pmmintrin.h>"], []);
_TMMINTRIN := () -> When(_hasSSSE3(),  ["<tmmintrin.h>"], []);
_SMMINTRIN := () -> When(_hasSSE4_1(), ["<smmintrin.h>"], []);
_NMMINTRIN := () -> When(_hasSSE4_2(), ["<nmmintrin.h>"], []);

#F ==============================================================================================
#F SIMD_Intel  --  Base class for Intel ISAs
#F
Class(SIMD_Intel, SIMD_ISA, rec(
    info := "Intel SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, MMX architecture definition base",
    arch := "Intel_SSE",
    file := "sse",
    unparser := SSEUnparser,

    # Should vector values be inlined or declared as constants
    declareConstants := false,

    # This is applied as the final pass of compileStrategy. Below we apply ISA specific strength
    # reduction rules, in particular some unaligned stores can only be eliminated in the final pass
    fixProblems := (self, c, opts) >> BUA(c, MergedRuleSet(RulesStrengthReduce, RulesSSEPostProcess), opts),

    autolib := rec(
	includes := () -> _NMMINTRIN() :: _SMMINTRIN() :: _TMMINTRIN() :: _PMMINTRIN() ::
	                  _EMMINTRIN() :: _XMMINTRIN(), 
        timerIncludes := () -> ["<include/sp_rdtsc.h>"]),

    unsigned := meth(self) self.isSigned:=false; return self; end,

    vzero := self >> self.t.zero(),

    intelCommonIncludes := self >> _NMMINTRIN() :: _SMMINTRIN() :: _TMMINTRIN() :: 
                                   _PMMINTRIN() :: _EMMINTRIN() :: _XMMINTRIN() :: _MMINTRIN(),

    simpIndicesInside := SSE_LDST.list
));

# ==============================================================================================
# 2-way double precision real
#
Class(SSE_2x64f, SIMD_Intel, rec(
    info := "SSE2 2 x 64-bit double",

    countrec := rec(
        ops := [
            [ add, sub, chslo_2x64f, chshi_2x64f, addsub_2x64f, hadd_2x64f], 
	    [ mul ],
            [ vunpacklo_2x64f, vunpackhi_2x64f, vshuffle_2x64f, vushuffle_2x64f ],
            [ vload1sd_2x64f,  vload_1l_2x64f,  vload_1h_2x64f, vloadu_2x64f, 
	      vstore_1l_2x64f, vstore_1h_2x64f, vstoreu_2x64f ],
            [ deref ],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),
    
    includes     := self >> ["<include/omega64.h>", "<include/mm_malloc.h>"] :: self.intelCommonIncludes(), 
    active       := true,
    isFixedPoint := false,
    isFloat      := true,

    v     := 2,
    t     := TVect(TReal, 2),
    ctype := "double",
    instr := [vunpacklo_2x64f, vunpackhi_2x64f, vshuffle_2x64f],
    bits  := 64,

    splopts  := rec(precision := "double"),
    dupload  := (y, x) -> assign(y, vdup(x, 2)),
    duploadn := (y, x, n) -> assign(y, vushuffle_2x64f(x, [n,n])),
    hadd     := (x1,x2) -> hadd_2x64f(x1,x2),

    # load full vectors using subvectors of length 1 or 2
    svload:= [ [ (y,x,opts) -> assign(y, vload1sd_2x64f(x[1].toPtr(TReal))),
                 (y,x,opts) -> let(u := var.fresh_t("U", TVectDouble(2)),
                      decl([u], chain(assign(u, vload1sd_2x64f(x[1].toPtr(TReal))),
                                      assign(y, vload_1h_2x64f(u, x[2].toPtr(TReal)))))) ],
               [ (y,x,opts) -> assign(y, nth(x[1].toPtr(TVect(TReal, 2)), 0)) ]
	     ],

    # store full vector using subvectors of length 1 or 2
    svstore := [ [ (y,x,opts) -> vstore_1l_2x64f(y[1].toPtr(TReal), x),
                   (y,x,opts) -> chain(vstore_1l_2x64f(y[1].toPtr(TReal), x),
                                       vstore_1h_2x64f(y[2].toPtr(TReal), x)) ],
                 [ (y,x,opts) -> assign(nth(y[1].toPtr(TVect(TReal, 2)), 0), x) ]
	       ],

    # keep the n lower scalars and zero the other ones
    optional_mask :=  (c, n, opts) -> When(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and not(n=2),
        let(f:="0xFFFFFFFF", z:="0x0", bin_and(c,tcast(TVect(TReal, 2), vhex(List([1..4],x->When(x/2<=n,f,z)))))),
        c),

    # load contiguous with unaligned loads
    loadc := (self, sv, opts) >> (
	(y,x) -> assign(y, self.optional_mask(vloadu_2x64f(x.toPtr(TReal)), sv, opts))),

    # load contiguous + known alignment -> using 2 aligned load to be smarter
    loadc_align := (self, sv, align, opts) >> ((y,x,addr) -> let(
        v1 := nth(nth(x,add(addr,-align)).toPtr(TVect(TReal, 2)),0),
        v2 := nth(nth(x,add(addr,2-align)).toPtr(TVect(TReal, 2)), 0),
        m := x -> self.optional_mask(x, sv, opts),

        Cond(align=0,  assign(y, m(v1)),
             sv=1,     assign(y, m(vushuffle_2x64f(v1, [2, 2]))),
             sv=2,     assign(y, m(vshuffle_2x64f(v1, v2, [2, 3]))),
             Error("bad parameters")))),

    # store contiguous unaligned
    storec := [ (y,x) -> vstore_1l_2x64f(y.toPtr(TReal), x),
                (y,x) -> vstoreu_2x64f  (y.toPtr(TReal), x) ],

    reverse := (y,x) -> assign(vref(y,0,2), vushuffle_2x64f(vref(x,0,2), [2,1])),

    mul_cx := (self, opts) >> Cond(
                    # SSE and MMX can't do double precision
                    opts.vector.SIMD in ["MMX", "SSE"],
                    Error("SSE2 required for double precision"),
                    # SSE2 only
                    opts.vector.SIMD = "SSE2",
                    (y,x,c) -> let(u1 := var.fresh_t("U", TVectDouble(2)), u2 := var.fresh_t("U", TVectDouble(2)),
                        u3 := var.fresh_t("U", TVectDouble(2)), u4 := var.fresh_t("U", TVectDouble(2)),
                        decl([u1, u2, u3, u4], chain(
                            assign(u1, mul(x, vushuffle_2x64f(c, [1,1]))),
                            assign(u2, chshi_2x64f(x)),
                            assign(u3, mul(u2, vushuffle_2x64f(c, [2,2]))),
                            assign(u4, vushuffle_2x64f(u3, [2,1])),
                            assign(y, add(u1, u4))))),
                    # SSE3 or higher
                    (y, x, c) -> let(u1 := var.fresh_t("U", TVectDouble(2)),
                                 u2 := var.fresh_t("U", TVectDouble(2)),
                                 u3 := var.fresh_t("U", TVectDouble(2)),
                        decl([u1, u2, u3], chain(
                            assign(u1, mul(x, vushuffle_2x64f(c, [1,1]))),
                            assign(u2, vushuffle_2x64f(x, [2,1])),
                            assign(u3, mul(u2, vushuffle_2x64f(c, [2,2]))),
                            assign(y, addsub_2x64f(u1, u3)))))
            ),

   mul_cx_conj := (self, opts) >> Cond(
                    # SSE and MMX can't do double precision
                    opts.vector.SIMD in ["MMX", "SSE"],
                    Error("SSE2 required for double precision"),
                    # SSE2 only
                    (y,x,c) -> let(u1 := var.fresh_t("U", TVectDouble(2)), u2 := var.fresh_t("U", TVectDouble(2)),
                        u3 := var.fresh_t("U", TVectDouble(2)), u4 := var.fresh_t("U", TVectDouble(2)),
                        decl([u1, u2, u3, u4], chain(
                            assign(u1, mul(x, vushuffle_2x64f(c, [1,1]))),
                            assign(u2, chslo_2x64f(x)),
                            assign(u3, mul(u2, vushuffle_2x64f(c, [2,2]))),
                            assign(u4, vushuffle_2x64f(u3, [2,1])),
                            assign(y, add(u1, u4)))))
		    ),

    bin_shl1 := (y,x,opts) -> assign(y, vec_shl(x, 1)),
    bin_shl2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],1))),
    bin_shr1 := (y,x,opts) -> assign(y, vec_shr(x, 1)),
    bin_shr2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],1))),
    # support for VO1dsJ(n, v)
    bin_shrev := (y,x,opts) -> assign(y, vshuffle_2x64f(x[1], x[2], [1,2])),
    swap_cx := (y, x, opts) -> assign(y, vushuffle_2x64f(x, [2,1])),
    RCVIxJ2 := (y, x, opts) -> assign(y, vushuffle_2x64f(x, [2,1])) # ???? maybe shuf3 rule is invalid

));


#==============================================================================================
#
Class(SSE_2x64i, SIMD_Intel, rec(
    info := "SSE2 2 x 64-bit integer",

    includes     := self >> ["<include/omega64i.h>"] :: self.intelCommonIncludes(), 
    active       := true,
    isFixedPoint := true,
    isFloat      := false,
    saturatedArithmetic := false,

    v     := 2,
    t     := TVect(TReal, 2),
    ctype := "__int64",
    instr := [vunpacklo_2x64i, vunpackhi_2x64i, vshuffle_2x64i],
    bits  := 64,
    fracbits := 62,

    splopts := rec(precision := "double"),
#    dupload := (y, x) -> assign(y, vushuffle_2x64i(vload1sd_2x64i(x.toPtr(TReal)), [1,1])),
    reverse := (y,x) -> assign(vref(y,0,2), vushuffle_2x64f(vref(x,0,2), [2,1])),
    bin_shl1 := (y,x,opts) -> assign(y, vec_shl(x, 1)),
    bin_shl2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],1))),
    bin_shr1 := (y,x,opts) -> assign(y, vec_shr(x, 1)),
    bin_shr2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],1))),
));

#==============================================================================================
Class(SSE_2x32f, SIMD_Intel, rec(
    info := "SSE 2 x 32-bit float",
    countrec := rec(
        ops := [
            [add, sub, addsub_4x32f, hadd_4x32f], [mul],
            [vunpacklo_4x32f, vunpackhi_4x32f, vshuffle_4x32f, vushuffle_4x32f],
            [vload1_4x32f,  vloadu_4x32f,  vloadu2_4x32f,
             vstore1_4x32f, vstoreu_4x32f, vstoreu2_4x32f, vstoremsk_4x32f],
            [deref, vload_2l_4x32f, vload_2h_4x32f, vstore_2l_4x32f, vstore_2h_4x32f],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),

    includes     := self >> ["<include/omega32.h>"] :: self.intelCommonIncludes(), 
    active       := true,
    isFixedPoint := false,
    isFloat      := true,

    v      := 2,
    t      := TVect(TReal, 2),
    ctype  := "float",
    instr  := [vunpacklo_2x32f, vunpackhi_2x32f, vshuffle_2x32f],
    bits   := 32,

    splopts  := rec(precision := "single"),
    dupload  := (y, x) -> assign(y, vdup(x, 2)),
    duploadn := (y, x, n) -> assign(y, vushuffle_2x32f(x, [n,n])),
    hadd     := (x1,x2) -> hadd_2x32f(x1,x2),

    # NOTE: Hacks. YSV: these are still alive as of svn 9206.
    #
    requireLoad := true,
    loadop := l -> vload_2x32f(vzero_2x32f(), l),
    requireStore := true,
    storeop := (l, v) -> vstore_2x32f(l.loc, v),
    needScalarVarFix := true,
    scalarVar := () -> var.fresh_t("P", TVectDouble(4)),
    ##

    svload:= [ [     # load full vectors using subvectors of length 1 or 2
                (y,x,opts) -> assign(y, vload1sd_2x32f(x[1].toPtr(TReal))),
                (y,x,opts) -> let(u := var.fresh_t("U", TVectDouble(2)),
                    decl([u], chain(assign(u, vload1sd_2x32f(x[1].toPtr(TReal))),
                                    assign(y, vload_1h_2x32f(u, x[2].toPtr(TReal))))))
                ],
                [(y,x,opts) -> assign(y, nth(x[1].toPtr(TVect(TReal, 2)), 0)) ]
        ],
    svstore := [[    # store full vector using subvectors of length 1 or 2
                (y,x,opts) -> vstore_1l_2x32f(y[1].toPtr(TReal), x),
                (y,x,opts) -> chain(vstore_1l_2x32f(y[1].toPtr(TReal), x),
                               vstore_1h_2x32f(y[2].toPtr(TReal), x))
                ],
                [
                (y,x,opts) -> assign(nth(y[1].toPtr(TVect(TReal, 2)), 0), x)
                ]],

    # keep the n lower scalars and zero the other ones
    optional_mask :=  (self, c, n, opts) >> Cond(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and n<>2,
	bin_and(c, tcast(self.t, vhex(List([1..4], x -> When(x/2<=n, "0xFFFFFFFF", "0x0"))))),
        c),

    # load contiguous with unaligned loads
    loadc := (self, sv, opts) >> (
	(y,x) -> assign(y, self.optional_mask(vloadu_2x32f(x.toPtr(TReal)), sv, opts))),

    # load contiguous + known alignment -> using 2 aligned load to be smarter
    loadc_align := (self, sv, align, opts)>>
    ((y,x,addr) -> let(
        v1 := nth(nth(x,add(addr,-align)).toPtr(TVect(TReal, 2)),0),
        v2 := nth(nth(x,add(addr,2-align)).toPtr(TVect(TReal, 2)), 0),
        m := x-> self.optional_mask(x, sv, opts),
        Cond(align=0,
                 assign(y, m(v1)),
	     sv=1,
                 assign(y, m(vushuffle_2x32f(v1, [2, 2]))),
             sv=2,
                 assign(y, m(vshuffle_2x32f(v1, v2, [2, 3]))),
	     # else
		 Error("bad parameters")))),

    # store contiguous unaligned
    storec := [    
        (y,x) -> vstore_1l_2x32f(y.toPtr(TReal), x),
        (y,x) -> vstoreu_2x32f(y.toPtr(TReal), x)
    ],

    reverse := (y,x) -> assign(vref(y,0,2), vushuffle_2x32f(vref(x,0,2), [2,1])),

    mul_cx := (self, opts) >> Cond(
        # SSE and MMX can't do double precision
        opts.vector.SIMD in ["MMX", "SSE"],
            Error("SSE2 required for double precision"),

        # SSE2 only
        opts.vector.SIMD = "SSE2",
            (y,x,c) -> let(
		u := var.fresh_t("U", self.t), w := var.fresh_t("U", self.t),
                decl([u, w], chain(
                     assign(u,             x  * vushuffle_2x32f(c, [1,1])),
                     assign(w, chshi_2x32f(x) * vushuffle_2x32f(c, [2,2])),
                     assign(y, u + vushuffle_2x32f(w, [2,1]))))),
        # SSE3 or higher
            (y, x, c) -> let(
		u := var.fresh_t("U", self.t), w := var.fresh_t("U", self.t),
                decl([u, w], chain(
                     assign(u,                       x   * vushuffle_2x32f(c, [1,1])),
                     assign(w, vushuffle_2x32f(x, [2,1]) * vushuffle_2x32f(c, [2,2])),
                     assign(y, addsub_2x32f(u, w)))))
    ),

    bin_shl1 := (y,x,opts) -> assign(y, vec_shl(x, 8)),
    bin_shl2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],1))),
    bin_shr1 := (y,x,opts) -> assign(y, vec_shr(x, 1)),
    bin_shr2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],1))),
    # support for VO1dsJ(n, v)
    bin_shrev := (y,x,opts) -> assign(y, vshuffle_2x32f(x[1], x[2], [1,2]))
));

#==============================================================================================

Class(SSE_4x32f, SIMD_Intel, rec(
    info := "SSE 4 x 32-bit float",

    # experimental -- parametized ISA, <el_t> is the type of each slot in the vector
    __call__ := (self, el_t) >> WithBases(self, rec(
        t            := TVect(el_t, 4), 
	isSigned     := el_t.isSigned(),
	isFloat      := IsRealT(el_t),
	isFixedPoint := IsFixedPtT(el_t),
	splopts      := CopyFields(self.splopts, rec(XType := TPtr(el_t), YType := TPtr(el_t))),
	svload       := self.svload_init(TVect(el_t, 4)),
	svstore      := self.svstore_init(TVect(el_t, 4)),
	storec       := self.storec_init(TVect(el_t, 4)),
	operations   := ISAOps,
	print        := self >> Print(self.__name__, "(", self.t.t, ")"),
	id           := self >> self.__name__ :: "_" :: el_t.strId(),
    )),

    countrec := rec(
        ops := [
            [ add, sub, addsub_4x32f, hadd_4x32f, chshi_4x32f, chslo_4x32f ], 
	    [ mul ],
            [ vunpacklo_4x32f, vunpackhi_4x32f, vshuffle_4x32f, vushuffle_4x32f ],
            [ vload1_4x32f,    vload_2l_4x32f,  vload_2h_4x32f,  vloadu_4x32f,  vloadu2_4x32f,
              vstore1_4x32f,   vstore_2l_4x32f, vstore_2h_4x32f, vstoreu_4x32f, vstoreu2_4x32f, 
	      vstoremsk_4x32f, vinsert_4x32f,   vextract_4x32f ],
            [ deref ],
            Value,      # Value without [] is a keyword in countOps !!
            [vcvt_4x32_i2f]
        ],
        printstrings := ["[adds]", "[mults]", "[vperms]", "[svldst]", "[vldst]", "[vval]", "[vcvt]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),

    includes     := self >> ["<include/omega32.h>"] :: self.intelCommonIncludes(), 
    active       := true,
    isFixedPoint := false,
    isFloat      := true,

    v      := 4,
    t      := TVect(TReal, 4),
    ctype  := "float",
    instr  := [vunpacklo_4x32f, vunpackhi_4x32f, vshuffle_4x32f, vushuffle_4x32f],
    bits   := 32,

    splopts  := rec(precision := "single"),

    #dupload := (self, y, x) >> self.duploadn(y, vload1_4x32f(x.toPtr(self.t.t)), 1),
    dupload := (self, y, x) >> assign(y, vdup(x, self.v)),
    duploadn := (y, x, n) -> assign(y, vushuffle_4x32f(x, [n,n,n,n])),

    hadd     := (x1,x2) -> hadd_4x32f(x1,x2),

    svload_init := (vt) -> [
        # load using subvectors of length 1
	[
            (y,x,opts) -> assign(y, vload1_4x32f(x[1].toPtr(vt.t))),

            (y,x,opts)->When(_hasSSE4_1(opts),
                let(u := var.fresh_t("U", vt),
                    decl([u], chain(
                            assign(u, vload1_4x32f(x[1].toPtr(vt.t))),
                            assign(y, vinsert_4x32f(u, deref(x[2].toPtr(T_Int(32))), 2))
                            ))),
                let(u1 := var.fresh_t("U", vt), u2 := var.fresh_t("U", vt),
                    decl([u1, u2], chain(
                            assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                            assign(u2, vload1_4x32f(x[2].toPtr(vt.t))),
                            assign(y, vunpacklo_4x32f(u1, u2))
                            )))
                ),

            (y,x,opts)->When(_hasSSE4_1(opts),
                let(u := var.fresh_t("U", vt),
                    decl([u], chain(
                            assign(u, vload1_4x32f(x[1].toPtr(vt.t))),
                            assign(u, vinsert_4x32f(u, deref(x[2].toPtr(T_Int(32))), 2)),
                            assign(y, vinsert_4x32f(u, deref(x[3].toPtr(T_Int(32))), 3))
                            ))),
                let(u1 := var.fresh_t("U", vt), u2 := var.fresh_t("U", vt),
                    u3 := var.fresh_t("U", vt), u4 := var.fresh_t("U", vt),
                    decl([u1, u2, u3, u4], chain(
                            assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                            assign(u2, vload1_4x32f(x[2].toPtr(vt.t))),
                            assign(u3, vshuffle_4x32f(u1, u2, [1,2,1,2])),
                            assign(u4, vload1_4x32f(x[3].toPtr(vt.t))),
                            assign(y, vshuffle_4x32f(u3, u4, [1,3,1,3]))
                            )))
                ),

            (y,x,opts)->When(_hasSSE4_1(opts),
                let(u := var.fresh_t("U", vt),
                    decl([u], chain(
                            assign(u, vload1_4x32f(x[1].toPtr(vt.t))),
                            assign(u, vinsert_4x32f(u, deref(x[2].toPtr(T_Int(32))), 2)),
                            assign(u, vinsert_4x32f(u, deref(x[3].toPtr(T_Int(32))), 3)),
                            assign(y, vinsert_4x32f(u, deref(x[4].toPtr(T_Int(32))), 4))
                            ))),
                let(u1 := var.fresh_t("U", vt), u2 := var.fresh_t("U", vt),
                    u3 := var.fresh_t("U", vt), u4 := var.fresh_t("U", vt),
                    u5 := var.fresh_t("U", vt), u6 := var.fresh_t("U", vt),
                    decl([u1, u2, u3, u4, u5, u6], chain(
                            assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                            assign(u2, vload1_4x32f(x[2].toPtr(vt.t))),
                            assign(u3, vshuffle_4x32f(u1, u2, [1,2,1,2])),
                            assign(u4, vload1_4x32f(x[3].toPtr(vt.t))),
                            assign(u5, vload1_4x32f(x[4].toPtr(vt.t))),
                            assign(u6, vshuffle_4x32f(u4, u5, [1,2,1,2])),
                            assign(y,  vshuffle_4x32f(u3, u6, [1,3,1,3]))
                            )))
                ),
	],
        # load using subvectors of length 2
        [    
            (y,x,opts) -> assign(y, vload_2l_4x32f(vzero_4x32f(), x[1].toPtr(TVect(vt.t,2)))),
            (y,x,opts) -> let(u := var.fresh_t("U", vt),
                decl(u, chain(
                        assign(u, vload_2l_4x32f(vzero_4x32f(), x[1].toPtr(TVect(vt.t,2)))),
                        assign(y, vload_2h_4x32f(u, x[2].toPtr(TVect(vt.t,2)))))
                    ))
        ]],

    svload := ~.svload_init(~.t),

    # keep the n lower scalars and zero the other ones
    optional_mask :=  (c, n, opts) -> Cond(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and n <> 4,
	bin_and(c, vhex(List([1..4], x -> When(x<=n, "0xFFFFFFFF", "0x0")))),
        c),

    # load contiguous with unaligned loads
    loadc := (self, sv, opts) >> ((y,x) -> assign(y, self.optional_mask(vloadu_4x32f(x.toPtr(self.t.t)), sv, opts))),

    #load contiguous + known alignment -> using 2 aligned load to be smarter
    loadc_align := (self, sv, align, opts)>>
    ((y,x,addr) -> let(
        v1 := nth(nth(x, addr-align)  .toPtr(self.t), 0),
        v2 := nth(nth(x, addr-align+4).toPtr(self.t), 0),
        m := x -> self.optional_mask(x, sv, opts),
        Cond(align=0,
            assign(y, m(v1)),

            _hasSSSE3(opts),
            assign(y, m(alignr_4x32f(v2, v1, align*4))),

            assign(y, m(bin_or(
                        vec_shr(v1, align),
                        vec_shl(v2, (4-align))
                        )))))),

    svstore_init := (vt) -> [
        [
            (y,x,opts) -> vstore1_4x32f(y[1].toPtr(vt.t), x),
            (y,x,opts)->When(_hasSSE4_1(opts),
                chain(
                    vstore1_4x32f(y[1].toPtr(vt.t), x),
                    vextract_4x32f(y[2].toPtr(T_Int(32)), x, 2)
                    ),
                let(u1 := var.fresh_t("U", vt),
                    decl([u1], chain(
                            vstore1_4x32f(y[1].toPtr(vt.t), x),
                            assign(u1, vushuffle_4x32f(x, [2,3,4,1])),
                            vstore1_4x32f(y[2].toPtr(vt.t), u1)
                            )))
                ),
            (y,x,opts)->When(_hasSSE4_1(opts),
                chain(
                    vstore1_4x32f(y[1].toPtr(vt.t), x),
                    vextract_4x32f(y[2].toPtr(T_Int(32)), x, 2),
                    vextract_4x32f(y[3].toPtr(T_Int(32)), x, 3)
                    ),
                let(u1 := var.fresh_t("U", vt), u2 := var.fresh_t("U", vt),
                    decl([u1, u2], chain(
                            vstore1_4x32f(y[1].toPtr(vt.t), x),
                            assign(u1, vushuffle_4x32f(x, [2,3,4,1])),
                            vstore1_4x32f(y[2].toPtr(vt.t), u1),
                            assign(u2, vushuffle_4x32f(u1, [2,3,4,1])),
                            vstore1_4x32f(y[3].toPtr(vt.t), u2)
                            )))
                ),
            (y,x,opts)->When(_hasSSE4_1(opts),
                chain(
                    vstore1_4x32f(y[1].toPtr(vt.t), x),
                    vextract_4x32f(y[2].toPtr(T_Int(32)), x, 2),
                    vextract_4x32f(y[3].toPtr(T_Int(32)), x, 3),
                    vextract_4x32f(y[4].toPtr(T_Int(32)), x, 4)
                    ),
                let(u1 := var.fresh_t("U", vt), u2 := var.fresh_t("U", vt),
                    u3 := var.fresh_t("U", vt),
                    decl([u1, u2, u3], chain(
                            vstore1_4x32f(y[1].toPtr(vt.t), x),
                            assign(u1, vushuffle_4x32f(x, [2,3,4,1])),
                            vstore1_4x32f(y[2].toPtr(vt.t), u1),
                            assign(u2, vushuffle_4x32f(u1, [2,3,4,1])),
                            vstore1_4x32f(y[3].toPtr(vt.t), u2),
                            assign(u3, vushuffle_4x32f(u2, [2,3,4,1])),
                            vstore1_4x32f(y[4].toPtr(vt.t), u3)
                            )))
                )
        ],
        [
            (y,x,opts) -> vstore_2l_4x32f(y[1].toPtr(TVect(vt.t,2)), x),
            (y,x,opts) -> chain(vstore_2l_4x32f(y[1].toPtr(TVect(vt.t,2)), x),
                                vstore_2h_4x32f(y[2].toPtr(TVect(vt.t,2)), x))
        ]
    ],

    svstore := ~.svstore_init(~.t),

    # Store contiguous unaligned
    storec_init := (vt) -> [    
        (y,x) -> vstore1_4x32f(y.toPtr(vt.t), x),
        (y,x) -> vstoreu2_4x32f(y.toPtr(TVect(T_Int(64),2)), x),
        (y,x) -> vstoremsk_4x32f(y.toPtr(vt.t), x, ["0xFFFFFFFF", "0xFFFFFFFF", "0xFFFFFFFF", "0x0"]),
        (y,x) -> vstoreu_4x32f(y.toPtr(vt.t), x)
    ],

    storec := ~.storec_init(~.t),

    reverse := (y,x) -> assign(vref(y,0,4), vushuffle_4x32f(vref(x,0,4), [4,3,2,1])),

    # support for VS and VS.transpose()
    bin_shl1 := (y,x,opts) -> assign(y, vec_shl(x, 1)),
#   bin_shl2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],3), vec_shl(x[2],1))),
    bin_shl2 := (self,y,x,opts) >> When(
	_hasSSSE3(opts),
            assign(y, alignr_4x32f(x[2], x[1], 12)),
	# else
        let(u := self.freshU(),
            chain(
                assign(u, vshuffle_4x32f(x[1], x[2], [3,4,1,2])),
                assign(y, vshuffle_4x32f(u,    x[2], [2,3,2,3]))))
    ),

    bin_shr1 := (y,x,opts) -> assign(y, vec_shr(x, 1)), 

    bin_shr2 := (self,y,x,opts) >> Cond(
	_hasSSSE3(opts),
            assign(y, alignr_4x32f(x[2], x[1], 4)),
	# else, no SSSE3
            let(u := self.freshU(), chain(                           # x = [a,b,c,d] [e,f,g,h] 
                assign(u, vshuffle_4x32f(x[1], x[2], [3,4,1,2])),    # u = [c,d,e,f]
                assign(y, vshuffle_4x32f(x[1], u,    [2,3,2,3]))))), # y = [b,c,d,e]
	# another way
	#   assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],3)))

    # support for VO1dsJ(n, v)
    bin_shrev := (self, y, x, opts) >> let(
	u := self.freshU(),                                     # data = [a b c d] [e f g h]
        chain(                                                  # x = [e,f,g,h] [a,b,c,d] 
            assign(u, vshuffle_4x32f(x[1], x[2], [1,2,3,4])),   # u = [e f c d]
            assign(y, vshuffle_4x32f(u,    x[2], [1,4,3,2])))), # y = [e d c b]

    # returns function (y,x,c) -> <code for y=x*c>
    mul_cx := (self, opts) >> Cond(
        # MMX can't do single precision
        opts.vector.SIMD in ["MMX"], Error("SSE required for single precision"),
        # SSE only
        opts.vector.SIMD in ["SSE", "SSE2"],
            (y,x,c) -> let(
		u := self.freshU(),  v := self.freshU(),
                decl([u, v], chain(
                    assign(u,                             x * vushuffle_4x32f(c, [1,1,3,3])),
                    assign(v, vushuffle_4x32f(x, [2,1,4,3]) * vushuffle_4x32f(c, [2,2,4,4])),
                    assign(y, u + chslo_4x32f(v))))),
        # SSE3 or higher
            (y,x,c) -> let(
		u := self.freshU(),  v := self.freshU(),
                decl([u, v], chain(
                    assign(u,                             x * vushuffle_4x32f(c, [1,1,3,3])),
                    assign(v, vushuffle_4x32f(x, [2,1,4,3]) * vushuffle_4x32f(c, [2,2,4,4])),
                    assign(y, addsub_4x32f(u, v)))))
    ),

    # returns function (y,x,c) -> <code for y=x*conj(c)>
    mul_cx_conj := (self, opts) >> Cond(
        # MMX can't do single precision
        opts.vector.SIMD in ["MMX"], Error("SSE required for single precision"),
        # SSE 
            (y,x,c) -> let(
		u := self.freshU(),  v := self.freshU(),
                decl([u, v], chain(
                    assign(u,             x  * vushuffle_4x32f(c, [1,1,3,3])),
                    assign(v, chslo_4x32f(x) * vushuffle_4x32f(c, [2,2,4,4])),
                    assign(v, vushuffle_4x32f(v, [2,1,4,3])),
                    assign(y, u + v))))
    ),

    swap_cx := (y, x, opts) -> assign(y, vushuffle_4x32f(x, [2,1,4,3])),
    RCVIxJ2 := (y, x, opts) -> assign(y, vushuffle_4x32f(x, [3,4,1,2])),

    freshU  := self >> var.fresh_t("U", self.t),

    hmin := (self, y, x, opts) >> let( u := self.freshU(),
       decl( [u], chain( 
           assign(u, min(x, vushuffle_4x32f(x, [2,1,4,3]))),
           assign(y, min(u, vushuffle_4x32f(u, [3,4,1,2]))) 
       ))),
    
    rotate_left := (self, shift) >> Cond(
            shift mod 4=0,
                ((y, x) -> assign(vtref(self.t, y, 0), vtref(self.t, x, 0))),
            shift mod 4=1,                         
                ((y, x) -> assign(vtref(self.t, y, 0), vushuffle_4x32f(vtref(self.t, x, 0), [4,1,2,3]))),
            shift mod 4=2,                                                              
                ((y, x) -> assign(vtref(self.t, y, 0), vushuffle_4x32f(vtref(self.t, x, 0), [3,4,1,2]))),
            shift mod 4=3,                                                              
                ((y, x) -> assign(vtref(self.t, y, 0), vushuffle_4x32f(vtref(self.t, x, 0), [2,3,4,1]))),
            #else                                  
                ((y, x) -> assign(vtref(self.t, y, 0),
                    bin_or( vec_shl(vtref(self.t, x, 0), imod(shift,4)),
                            vec_shr(vtref(self.t, x, 0), 4-imod(shift,4)))))),
));

#==============================================================================================

Class(SSE_4x32i, SIMD_Intel, rec(
    active := true,
    isFixedPoint := true,
    isFloat := false,
    saturatedArithmetic := false,
    info := "SSE 4 x 32-bit integer",
    v := 4,
    t := TVect(TReal, 4),
    instr := [vunpacklo_4x32i, vunpackhi_4x32i, vshuffle_4x32i, vushuffle_4x32i],
    bits := 32,
    fracbits := 30,
    includes := self >> ["<include/omega32i.h>"] :: self.intelCommonIncludes(), 
    splopts := DataTypes.i32re,

#    dupload := (y, x) -> assign(y, vushuffle_4x32i(vload1_4x32i(x.toPtr(TReal)), [1,1,1,1])),
    reverse := (y,x) -> assign(vref(y,0,4), vushuffle_4x32i(vref(x,0,4), [4,3,2,1])),
    # support for VS and VS.transpose()
    bin_shl1 := (y,x,opts) -> assign(y, vec_shl(x, 1)),
    bin_shl2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],3), vec_shl(x[2],1))),
    bin_shr1 := (y,x,opts) -> assign(y, vec_shr(x, 1)),
    bin_shr2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],3))),
    interleavedmask := (b,idx,x1,x2) -> assign(nth(tcast(TPtr(TUChar), b),idx),interleavedmask_4x32i(x1,x2)),
    vrshift := "_mm_srli_epi32",
    vlshift := "_mm_slli_epi32",
    vmul := "_mm_mulhi_epi32"
));

#==============================================================================================

Class(SSE_8x16i, SIMD_Intel, rec(
    info := "SSE2 8 x 16-bit int",
    
    # experimental -- parametized ISA, <el_t> is the type of each 8-bit slot in the vector
    __call__ := (self, el_t) >> WithBases(self, rec(
        t            := TVect(el_t, 8), 
	isSigned     := el_t.isSigned(),
	isFloat      := IsRealT(el_t),
	isFixedPoint := IsFixedPtT(el_t),
	splopts      := CopyFields(self.splopts, rec(XType := TPtr(el_t), YType := TPtr(el_t))),
	svload       := self.svload_init(TVect(el_t, 8)),
	svstore      := self.svstore_init(TVect(el_t, 8)),
	storec       := self.storec_init(TVect(el_t, 8)),
        operations   := ISAOps,
	print        := self >> Print(self.__name__, "(", self.t.t, ")"),
	id           := self >> self.__name__ :: "_" :: el_t.strId(),
    )),

    countrec := rec(
        ops := [
            [ add, sub, chs_8x16i ], 
	    [ fpmul ],
            [ vunpacklo_8x16i,   vunpackhi_8x16i,  vunpacklo2_8x16i, vunpackhi2_8x16i,
              vunpacklo4_8x16i,  vunpackhi4_8x16i, vushuffle2_8x16i, vushufflelo_8x16i, 
	      vushufflehi_8x16i, vushuffle_8x16i,  vshuffle2_8x16i,  vshuffle4_8x16i, 
	      alignr_8x16i ],
            [ vload1_8x16i, vload2_8x16i, vload4_8x16i, vloadu_8x16i,
              vextract1_8x16i, vextract2_8x16i, vstoreu_8x16i, vstore4_8x16i, vstoremsk_8x16i ],
            [ deref ],
            Value
        ],
        printstrings := ["[adds]", "[mults]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]*2
    ),

    includes     := self >> ["<include/omega16i.h>"] :: self.intelCommonIncludes(), 
    active       := true,
    isFixedPoint := true,
    isFloat      := false,
    saturatedArithmetic := false,

    v     := 8,
    t     := TVect(TReal, 8),
    ctype := "short int",
    instr := [vunpacklo_8x16i,   vunpackhi_8x16i,   vunpacklo2_8x16i, vunpackhi2_8x16i,
              vunpacklo4_8x16i,  vunpackhi4_8x16i,  vushuffle2_8x16i, vushufflelo_8x16i, 
	      vushufflehi_8x16i, vshuffle2_8x16i ], #vshuffle4_8x16i
    bits     := 16,
    fracbits := 14,
    vrshift  := "_mm_srai_epi16", # "_mm_srli_epi16",
    vlshift  := "_mm_slli_epi16",

    splopts := rec(customDataType := "short_fp14"),

    dupload := (self, y, x) >> assign(y, vdup(x, self.v)),

    svload_init := (vt) -> [
        # ------------------------------------
        # Load using subvectors of length 1
	[ 
	  (y,x,opts) -> assign(y, vload1_8x16i(vt.zero(), x[1], 0)),
          (y,x,opts) -> let(
	      u  := var.fresh_t("U", vt),
	      cc := List([1..2], i -> assign(u, vload1_8x16i(u, x[i], i-1))),
              decl([u], chain(assign(u, vt.zero()), cc, assign(y, u)))),
          (y,x,opts) -> let(
	      u  := var.fresh_t("U", vt),
	      cc := List([1..3], i -> assign(u, vload1_8x16i(u, x[i], i-1))),
              decl([u], chain(assign(u, vt.zero()), cc, assign(y, u)))),
          (y,x,opts) -> let(
	      u  := var.fresh_t("U", vt),
	      cc := List([1..4], i -> assign(u, vload1_8x16i(u, x[i], i-1))),
              decl([u], chain(assign(u, vt.zero()), cc, assign(y, u)))),
          (y,x,opts) -> let(
	      u  := var.fresh_t("U", vt),
	      cc := List([1..5], i -> assign(u, vload1_8x16i(u, x[i], i-1))),
              decl([u], chain(assign(u, vt.zero()), cc, assign(y, u)))),
          (y,x,opts) -> let(
	      u  := var.fresh_t("U", vt),
	      cc := List([1..6], i -> assign(u, vload1_8x16i(u, x[i], i-1))),
              decl([u], chain(assign(u, vt.zero()), cc, assign(y, u)))),
          (y,x,opts) -> let(
	      u  := var.fresh_t("U", vt),
	      cc := List([1..7], i -> assign(u, vload1_8x16i(u, x[i], i-1))),
              decl([u], chain(assign(u, vt.zero()), cc, assign(y, u)))),
          (y,x,opts) -> let(
	      u  := var.fresh_t("U", vt),
	      cc := List([1..8], i -> assign(u, vload1_8x16i(u, x[i], i-1))),
              decl([u], chain(assign(u, vt.zero()), cc, assign(y, u)))),                 
        ],

        # ------------------------------------
        # Load using subvectors of length 2
        [ (y,x,opts) -> let(
	      u := var.fresh_t("U", vt), 
              decl([u], chain(
                      assign(u, vload2_8x16i(nth(x[1].toPtr(T_Int(32)), 0))),
                      assign(u, vunpacklo2_8x16i(u, vt.zero())),
                      assign(y, vunpacklo4_8x16i(u, vt.zero()))))),
          (y,x,opts) -> let(
	      u := var.fresh_t("U", vt), v := var.fresh_t("U", vt),
              decl([u, v], chain(
                      assign(u, vload2_8x16i(nth(x[1].toPtr(T_Int(32)), 0))), 
		      assign(v, vload2_8x16i(nth(x[2].toPtr(T_Int(32)), 0))),
                      assign(u, vunpacklo2_8x16i(u, v)),
                      assign(y, vunpacklo4_8x16i(u, vt.zero()))))),
          (y,x,opts) -> let(
	      u := var.fresh_t("U", vt), v := var.fresh_t("U", vt), 
	      w := var.fresh_t("U", vt),
              decl([u, v, w], chain(
                      assign(u, vload2_8x16i(nth(x[1].toPtr(T_Int(32)), 0))), 
		      assign(v, vload2_8x16i(nth(x[2].toPtr(T_Int(32)), 0))),
                      assign(u, vunpacklo2_8x16i(u, v)),
                      assign(w, vload2_8x16i(nth(x[3].toPtr(T_Int(32)), 0))),
                      assign(w, vunpacklo2_8x16i(w, vt.zero())),
                      assign(y, vunpacklo4_8x16i(u, w))))),
          (y,x,opts) -> let(
	      u := var.fresh_t("U", vt),  v := var.fresh_t("U", vt),
	      uu := var.fresh_t("U", vt), vv := var.fresh_t("U", vt),
              decl([u, v, uu, vv], chain(
                      assign(u, vload2_8x16i(nth(x[1].toPtr(T_Int(32)), 0))),
		      assign(v, vload2_8x16i(nth(x[2].toPtr(T_Int(32)), 0))),
                      assign(u, vunpacklo2_8x16i(u, v)),
                      assign(uu, vload2_8x16i(nth(x[3].toPtr(T_Int(32)), 0))), 
		      assign(vv, vload2_8x16i(nth(x[4].toPtr(T_Int(32)), 0))),
                      assign(uu, vunpacklo2_8x16i(uu, vv)),
                      assign(y,  vunpacklo4_8x16i(u, uu)))))
        ]
    ],

    svload := ~.svload_init(~.t),

    svstore_init := (vt) -> [
        # ------------------------------------
        # Store subvectors of length 1
	#
	[ (y,x,opts) -> chain(List([1..1], i -> assign(y[i], vextract1_8x16i(x, i-1)))),
          (y,x,opts) -> chain(List([1..2], i -> assign(y[i], vextract1_8x16i(x, i-1)))),
          (y,x,opts) -> chain(List([1..3], i -> assign(y[i], vextract1_8x16i(x, i-1)))),
          (y,x,opts) -> chain(List([1..4], i -> assign(y[i], vextract1_8x16i(x, i-1)))),
          (y,x,opts) -> chain(List([1..5], i -> assign(y[i], vextract1_8x16i(x, i-1)))),
          (y,x,opts) -> chain(List([1..6], i -> assign(y[i], vextract1_8x16i(x, i-1)))),
          (y,x,opts) -> chain(List([1..7], i -> assign(y[i], vextract1_8x16i(x, i-1)))),
          (y,x,opts) -> chain(List([1..8], i -> assign(y[i], vextract1_8x16i(x, i-1)))),
        ],
        # ------------------------------------
        # Store subvectors of length 2
	#
        [ (y,x,opts) -> assign(nth(y[1].toPtr(T_Int(32)),0), vextract2_8x16i(x)),

          (y,x,opts) -> let(
	      u := var.fresh_t("U", vt),
              decl([u], chain(
                      assign(nth(y[1].toPtr(T_Int(32)),0), vextract2_8x16i(x)),
                      assign(u, vushuffle2_8x16i(x, [2,3,4,1])), 
		      assign(nth(y[2].toPtr(T_Int(32)),0), vextract2_8x16i(u))
                      ))),
          (y,x,opts) -> let(
	      u1 := var.fresh_t("U", vt), u2 := var.fresh_t("U", vt),
              decl([u1, u2], chain(
                      assign(nth(y[1].toPtr(T_Int(32)),0), vextract2_8x16i(x)),
                      assign(u1, vushuffle2_8x16i(x,  [2,3,4,1])), assign(nth(y[2].toPtr(T_Int(32)),0), vextract2_8x16i(u1)),
                      assign(u2, vushuffle2_8x16i(u1, [2,3,4,1])), assign(nth(y[3].toPtr(T_Int(32)),0), vextract2_8x16i(u2))
                      ))),
          (y,x,opts) -> let(
	      u1 := var.fresh_t("U", vt), u2 := var.fresh_t("U", vt), u3 := var.fresh_t("U", vt),
              decl([u1, u2, u3], chain(
                      assign(nth(y[1].toPtr(T_Int(32)),0), vextract2_8x16i(x)),
                      assign(u1, vushuffle2_8x16i(x,  [2,3,4,1])), assign(nth(y[2].toPtr(T_Int(32)),0), vextract2_8x16i(u1)),
                      assign(u2, vushuffle2_8x16i(u1, [2,3,4,1])), assign(nth(y[3].toPtr(T_Int(32)),0), vextract2_8x16i(u2)),
                      assign(u3, vushuffle2_8x16i(u2, [2,3,4,1])), assign(nth(y[4].toPtr(T_Int(32)),0), vextract2_8x16i(u3))
                      )))
        ]
    ],

    svstore := ~.svstore_init(~.t),

    # keep the n lower scalars and zero the other ones
    optional_mask :=  (c, n, opts) -> Cond(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and n<>8, 
        bin_and(c, vhex(List([1..8], x -> When(x<=n, "0xFFFF", "0x0")))),
        c),

    # Load contiguous with unaligned loads
    loadc := (self, sv, opts) >> (
	(y,x) -> assign(y, self.optional_mask(vloadu_8x16i(x.toPtr(self.t.t)), sv, opts))),

    # Load contiguous + known alignment -> using 2 aligned load to be smarter
    loadc_align := (self, sv, align, opts) >>
    ((y,x,addr) -> let(
        v1 := nth(nth(x, addr-align  ).toPtr(self.t), 0),
        v2 := nth(nth(x, addr-align+8).toPtr(self.t), 0),
        mask := x -> self.optional_mask(x, sv, opts),
        Cond(align=0,
                 assign(y, mask(v1)),
	     _hasSSSE3(opts),
                 assign(y, mask(alignr_8x16i(v2, v1, align*2))),
	     # else
                 assign(y, mask(bin_or(vec_shr(v1, align), vec_shl(v2, 8-align))))
	))),

    # Store contiguous unaligned
    storec_init := (vt) -> [    
        (y,x) -> assign(y, vextract1_8x16i(x, 0)),
        (y,x) -> assign(nth(y.toPtr(T_Int(32)),0), vextract2_8x16i(x)),
        (y,x) -> vstoremsk_8x16i(y.toPtr(vt.t), x, ["0xFFFF", "0xFFFF", "0xFFFF", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstore4_8x16i(y.toPtr(vt), x),
        (y,x) -> vstoremsk_8x16i(y.toPtr(vt.t), x, ["0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_8x16i(y.toPtr(vt.t), x, ["0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0x0", "0x0"]),
        (y,x) -> vstoremsk_8x16i(y.toPtr(vt.t), x, ["0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0xFFFF", "0x0"]),
        (y,x) -> vstoreu_8x16i(y.toPtr(vt), x)
    ],

    storec := ~.storec_init(~.t),

    reverse := (y,x) -> assign(vref(y,0,8), 
	vushufflehi_8x16i(vushufflelo_8x16i(vushuffle2_8x16i(vref(x,0,8), [3,4,1,2]), [4,3,2,1]), [4,3,2,1])),

    # support for VS and VS.transpose()
    bin_shl1 := (self,y,x,opts) >> assign(y, vec_shl(x, 1)),
    bin_shl2 := (self,y,x,opts) >> When(_hasSSSE3(opts),
        assign(y, alignr_8x16i(x[2], x[1], 14)),
        assign(y, bin_or(vec_shr(x[1], 7), vec_shl(x[2], 1)))
    ),

    bin_shr1 := (self,y,x,opts) >> assign(y, vec_shr(x, 1)),
    bin_shr2 := (self,y,x,opts) >> When(_hasSSSE3(opts),
        assign(y, alignr_8x16i(x[2], x[1], 2)),
        assign(y, bin_or(vec_shr(x[1], 1), vec_shl(x[2], 7)))
    ),

    interleavedmask := (b,idx,x1,x2) -> assign(nth(tcast(TPtr(TSym("short int")), b),idx),interleavedmask_8x16i(x1,x2)),

    # support for VO1dsJ(n, v)
    bin_shrev := (y,x,opts) -> let(
	u := var.fresh_t("U", TVectDouble(8)),
        chain(
            opts.vector.isa.bin_shr2(u, [x[2], x[1]], opts),
            assign(y, vushufflehi_8x16i(vushufflelo_8x16i(vushuffle2_8x16i(u, [3,4,1,2]), [4,3,2,1]), [4,3,2,1])))
    ),

    mul_cx := (self, opts) >> Cond(
        # MMX and SSE can't do 8x16i
        opts.vector.SIMD in ["MMX", "SSE"],
            Error("SSE2 required for 8-way 16-bit integer"),

        opts.vector.SIMD in ["SSE2", "SSE3"],
            (y,x,c) -> let(
		u1 := var.fresh_t("U", self.t), u2 := var.fresh_t("U", self.t),
                u3 := var.fresh_t("U", self.t), u4 := var.fresh_t("U", self.t),
                decl([u1, u2, u3, u4], chain(
                        assign(u1, x * vushufflehi_8x16i(vushufflelo_8x16i(c, [1,1,3,3]), [1,1,3,3])),
                        assign(u2, x * self.t.value([1,-1,1,-1,1,-1,1,-1])),
                        assign(u3, u2 * vushufflehi_8x16i(vushufflelo_8x16i(c, [2,2,4,4]), [2,2,4,4])),
                        assign(u4, vushufflehi_8x16i(vushufflelo_8x16i(u3, [2,1,4,3]), [2,1,4,3])),
                        assign(y, add(u1, u4))))),
        # SSSE3 and higher
            (y,x,c) -> let(
		u1 := var.fresh_t("U", self.t), u2 := var.fresh_t("U", self.t),
                u3 := var.fresh_t("U", self.t), u4 := var.fresh_t("U", self.t),
                decl([u1, u2, u3, u4], chain(
                        assign(u1, x * vushuffle_8x16i(c, TVect(T_Int(8), 16).value([0,1,0,1, 4,5,4,5, 8,9,8,9, 12,13,12,13]))),
                        assign(u2, chs_8x16i(x, TVect(T_Int(16), 8).value([1,-1,1,-1,1,-1,1,-1]))),
                        assign(u3, u2 * vushuffle_8x16i(c, TVect(T_Int(8), 16).value([2,3,2,3, 6,7,6,7, 10,11,10,11, 14,15,14,15]))),
                        assign(u4, vushuffle_8x16i(u3, TVect(T_Int(8), 16).value([2,3,0,1, 6,7,4,5, 10,11,8,9, 14,15,12,13]))),
                        assign(y,  u1 + u4))))
    ),

    swap_cx := (y, x, opts) -> assign(y, vushuffle_8x16i(x, TVect(T_Int(8), 16).value([2,1,4,3, 6,5,8,7, 10,9,12,11, 14,13,16,15]))),
    RCVIxJ2 := (y, x, opts) -> assign(y, vushuffle_8x16i(x, TVect(T_Int(8), 16).value([4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11]))),

    # computes the horizontal minimum of the vector and splats it
    hmin := (self,y,x,opts) >> Cond(
        # MMX and SSE can't do 8x16i
        opts.vector.SIMD in ["MMX", "SSE"],
            Error("SSE2 required for 8-way 16-bit integer"),

        opts.vector.SIMD in ["SSE2", "SSE3", "SSSE3"], let(
	    z := var.fresh_t("m", self.t),
	    decl([z], chain(
	        assign(z, Cond(self.t.isSigned(), x, add(x, self.t.value(-128)))),
	        assign(z, min(z, vushufflelo_8x16i(vushufflehi_8x16i(z, [2,1,4,3]), [2,1,4,3]))),
	        assign(z, min(z, vushuffle_4x32i(z, [2,1,4,3]))),
	        assign(z, min(z, vushuffle_4x32i(z, [3,4,1,2]))),
	        assign(y, Cond(self.t.isSigned(), z, add(z, self.t.value(-128))))))),
	# else SSE4 - there is hmin instruction, need to put it here 
	let( z := var.fresh_t("m", self.t),
	    decl([z], chain(
	        assign(z, min(x, vushufflelo_8x16i(vushufflehi_8x16i(x, [2,1,4,3]), [2,1,4,3]))),
	        assign(z, min(z, vushuffle_4x32i(z, [2,1,4,3]))),
	        assign(y, min(z, vushuffle_4x32i(z, [3,4,1,2]))))))
    ),

));

#==============================================================================================

_svload_16x8i_sv1 := (cnt) -> (
	(y,x,opts) -> let(
	      u := var.fresh_t("U", y.t),
	      i := Ind(idiv(cnt, 2)),
	      c := [assign(u, vzero_8x16i())]
	           :: List( [1..IntDouble(cnt/2.0)], i->assign(u, vload1_8x16i(u, bin_or(x[2*i-1],  vec_shl(x[2*i],  1)), i-1)) )
	           :: When( cnt mod 2 = 0, [], [assign(u, vload1_8x16i(u, x[cnt]), idiv(cnt, 2))] ),
              decl([u], chain(c, assign(y, u)))
        )
);

_svstore_16x8i_sv1 := (cnt) -> (
	(y,x,opts) -> let(
	      u := var.fresh_t("U", T_UInt(16)),
	      c := ConcatList( [1..IntDouble(cnt/2.0)], i -> 
	               [ assign(u, vextract1_8x16i(x, i-1)),
	                 assign(y[2*i-1], u),
	                 assign(y[2*i  ], vec_shr(u, 1)) ] )
	           :: When( cnt mod 2 = 0, [], 
	               [ assign(u, vextract1_8x16i(x, idiv(cnt,2))),
	                 assign(y[cnt], u) ]),
              decl([u], chain(c))
        )
);

Class(SSE_16x8i, SIMD_Intel, rec(
    info := "SSE2 16 x 8-bit int",

    # experimental -- parametized ISA, <el_t> is the type of each 8-bit slot in the vector
    __call__ := (self, el_t) >> WithBases(self, rec(
        t            := TVect(el_t, 16), 
	isSigned     := el_t.isSigned(),
	isFloat      := IsRealT(el_t),
	isFixedPoint := IsFixedPtT(el_t),
	splopts      := CopyFields(self.splopts, rec(XType := TPtr(el_t), YType := TPtr(el_t))),
	svload       := self.svload_init(TVect(el_t, 16)),
	svstore      := self.svstore_init(TVect(el_t, 16)),
	storec       := self.storec_init(TVect(el_t, 16)),
        operations   := ISAOps,
	print        := self >> Print(self.__name__, "(", self.t.t, ")"),
	id           := self >> self.__name__ :: "_" :: el_t.strId(),
    )),

    includes     := self >> ["<include/omega8i.h>"] :: self.intelCommonIncludes(), 
    active       := true,
    isFixedPoint := true,
    isFloat      := false,

    isSigned            := true,
    saturatedArithmetic := false,

    v     := 16,
    t     := TVect(TReal, 16),
    ctype := "__int8",
    instr := [vunpacklo_16x8i,  vunpackhi_16x8i,    vunpacklo2_16x8i, vunpackhi2_16x8i,
              vunpacklo4_16x8i, vunpackhi4_16x8i,   vunpacklo8_16x8i, vunpackhi8_16x8i,
              vushuffle4_16x8i, vushufflelo2_16x8i, vushufflehi2_16x8i, 
	      vshuffle4_16x8i,  vshuffle8_16x8i],
    bits     := 8,
    fracbits := 6,
    vrshift := "_mm_srli_epi8",
    vlshift := "_mm_slli_epi8",
    splopts := rec(customDataType := "char_fp4"),

    # NOTE: missing .loadCont, .storeCont, .svstore

    # keep the n lower scalars and zero the other ones
    optional_mask :=  (c, n, opts) -> When(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and not(n=16),
        let(f:="0xFF", z:="0x0", bin_and(c,vhex(List([1..16],x->When(x<=n,f,z))))),
        c),

    loadCont := (self, n, y, yofs, x, xofs, xofs_align, opts) >> let(
	a := _unwrap(xofs_align),
	nn := _unwrap(n),
	yy := vtref(self.t, y, yofs),
	assign(yy, self.optional_mask(vloadu_16x8i(x + xofs), n, opts))),

#    ((y,x,addr) -> let(
#        v1 := nth(nth(x, addr-align  ).toPtr(TVectDouble(8)), 0),
#        v2 := nth(nth(x, addr-align+8).toPtr(TVectDouble(8)), 0),
#        mask := x -> self.optional_mask(x, sv, opts),
#        Cond(align=0,
#                 assign(y, mask(v1)),
#	     _hasSSSE3(opts),
#                 assign(y, mask(alignr_8x16i(v2, v1, align*2))),
#	     # else
#                 assign(y, mask(bin_or(vec_shr(v1, align),
#                                       vec_shl(v2, (8-align))))))
#	)),

    # NOTE: rewrite using a method, and add missing.
    svload_init := (vt) -> [
	[ _svload_16x8i_sv1(1),
	  _svload_16x8i_sv1(2),
	  _svload_16x8i_sv1(3),
	  _svload_16x8i_sv1(4),
	  _svload_16x8i_sv1(5),
	  _svload_16x8i_sv1(6),
	  _svload_16x8i_sv1(7),
	  _svload_16x8i_sv1(8),
	  _svload_16x8i_sv1(9),
	  _svload_16x8i_sv1(10),
	  _svload_16x8i_sv1(11),
	  _svload_16x8i_sv1(12),
	  _svload_16x8i_sv1(13),
	  _svload_16x8i_sv1(14),
	  _svload_16x8i_sv1(15),
	  _svload_16x8i_sv1(16)
        ]
    ],

    svload := ~.svload_init(~.t),

    svstore_init := (vt) -> [
	[ _svstore_16x8i_sv1(1),
	  _svstore_16x8i_sv1(2),
	  _svstore_16x8i_sv1(3),
	  _svstore_16x8i_sv1(4),
	  _svstore_16x8i_sv1(5),
	  _svstore_16x8i_sv1(6),
	  _svstore_16x8i_sv1(7),
	  _svstore_16x8i_sv1(8),
	  _svstore_16x8i_sv1(9),
	  _svstore_16x8i_sv1(10),
	  _svstore_16x8i_sv1(11),
	  _svstore_16x8i_sv1(12),
	  _svstore_16x8i_sv1(13),
	  _svstore_16x8i_sv1(14),
	  _svstore_16x8i_sv1(15),
	  _svstore_16x8i_sv1(16)
        ]
    ],

    svstore := ~.svstore_init(~.t),

    bin_shl1 := (y,x,opts) -> assign(y, vec_shl(x, 1)),
    bin_shl2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],15), vec_shl(x[2],1))),
    bin_shr1 := (y,x,opts) -> assign(y, vec_shr(x, 1)),
    bin_shr2 := (y,x,opts) -> assign(y, bin_or(vec_shr(x[1],1), vec_shl(x[2],15))),

    bin_shr  := (self, y, x, shift, opts) >> assign(y, bin_and( tcast(self.t, bin_shr(tcast(TVect(T_Int(16), 8), x), shift)), 
	                                                        TVect(T_UInt(8), 16).value(idiv(255, 2^shift)))),

    interleavedmask := (b,idx,x1,x2) -> chain(
        assign(nth(tcast(TPtr(TSym("short int")), b), 2*idx),   interleavedmasklo_16x8i(x1,x2)),
        assign(nth(tcast(TPtr(TSym("short int")), b), 2*idx+1), interleavedmaskhi_16x8i(x1,x2))),

    average := (x1,x2) -> average_16x8i(x1,x2),

    #splats the first slot of the vector across the full vector
    dupload := (self, y, x) >> assign(y, vdup(x, self.v)),
    # computes the horizontal minimum of the vector and splats it
    hmin := (self,y,x,opts) >> let(
        sh_t := TVect(T_Int(64), 2),
	z := var.fresh_t("m", self.t),
	decl([z], chain(
	    assign(z, min(vec_shr(x, 8), x)),
	    assign(z, min(vec_shr(z, 4), z)),
            assign(z, min(vec_shr(z, 2), z)),
            assign(z, min(vec_shr(z, 1), z)),
	    assign(z, vunpacklo_16x8i(z, z)),
	    assign(z, vushufflelo_8x16i(z, [1,1,1,1])),
	    assign(y, vunpacklo_2x64i(z, z))))
    ),

    # keep the n lower scalars and zero the other ones    
    optional_mask :=  (c, n, opts) -> When(IsBound(opts.trueSVSemantics) and opts.trueSVSemantics and not(n=16),
                          let(f:="0xFF", z:="0x0", bin_and(c,vhex(List([1..16],x->When(x<=n,f,z))))),
                          c),
    # load contiguous with unaligned loads
    loadc := (self, sv, opts) >> ((y,x) -> assign(y, self.optional_mask(vloadu_16x8i(x.toPtr(self.t.t)), sv, opts))),

    # Store contiguous unaligned
    storec_init := (vt) -> [    
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        #(y,x) -> assign(y, vextract2_16x8i(x, 0)),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        #(y,x) -> assign(nth(y.toPtr(TVect(vt.t, 4)),0), vextract4_16x8i(x)),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        #(y,x) -> vstore8_16x8i(y.toPtr(TVect(vt.t, 8)), x),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0", "0x0"]),
        (y,x) -> vstoremsk_16x8i(y.toPtr(vt.t), x, ["0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0xFF", "0x0"]),
        (y,x) -> vstoreu_16x8i(y.toPtr(vt), x)
    ],

    storec := ~.storec_init(~.t),
));

SIMD_ISA_DB.addISA(SSE_2x64f);
SIMD_ISA_DB.addISA(SSE_2x64i);
SIMD_ISA_DB.addISA(SSE_2x32f);
SIMD_ISA_DB.addISA(SSE_4x32f);
SIMD_ISA_DB.addISA(SSE_4x32i);
SIMD_ISA_DB.addISA(SSE_8x16i);
SIMD_ISA_DB.addISA(SSE_16x8i);






