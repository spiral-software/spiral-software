
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_AVXINTRIN := () ->  ["<immintrin.h>"];
_MM_MALLOC := () -> When(not LocalConfig.osinfo.isDarwin(), ["<include/mm_malloc.h>"],[]);

Class(AVX_Intel, SIMD_ISA, rec(
    unparser := AVXUnparser,
    arch := "Intel_AVX",
    info := "Intel AVX architecture definition base",
    declareConstants := false,

    file := "avx",
    #arrayDataModifier := "static __declspec(align(32))",
    #arrayBufModifier  := "static __declspec(align(32))",
	alignmentBytes := 32,

    autolib := rec(
        includes := () -> _NMMINTRIN() :: _SMMINTRIN() :: _TMMINTRIN() :: _PMMINTRIN() 
                       :: _EMMINTRIN() :: _XMMINTRIN() :: _AVXINTRIN(), 
        timerIncludes := () -> ["<include/sp_rdtsc.h>"],
        cFlags := "-mavx",
    ),

    simpIndicesInside := AVX_LDST.getList(),
));

#==============================================================================================
# 4-way double precision real

Class(AVX_4x64f, AVX_Intel, rec(
    info     := "AVX 4 x 64-bit double",
    countrec := rec(
        ops := [
            [ add, sub, addsub_4x64f, chslo_2x64f, chshi_2x64f, addsub_2x64f, hadd_2x64f ], 
	    [ mul ], 
	    [ fmaddsub_4x64f],
            [ vunpacklo_4x64f,  vunpackhi_4x64f,  vshuffle_4x64f,  vperm2_4x64f,    vpermf128_4x64f, 
              vuunpacklo_4x64f, vuunpackhi_4x64f, vushuffle_4x64f, vuperm2_4x64f,   vupermf128_4x64f,
              vunpacklo_2x64f,  vunpackhi_2x64f,  vshuffle_2x64f,  vushuffle_2x64f, vperm_4x64f,
              vinsert_2l_4x64f, vextract_2l_4x64f, vblend_4x64f],
            [ vloadmask_4x64f,  vstoremask_4x64f, vload1sd_2x64f,  vload_1l_2x64f,  vload_1h_2x64f, 
	      vloadu_2x64f,     vstore_1l_2x64f,  vstore_1h_2x64f, vstoreu_2x64f ],
            [ deref, vloadu_4x64f, vstoreu_4x64f ],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[fmas]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]+2*opcount[3]
    ),

    includes     := () -> ["<include/omega64.h>"] :: _AVXINTRIN(),
    active       := true,
    isFixedPoint := false,
    isFloat      := true,

    v       := 4,
    t       := TVect(T_Real(64), 4),
    ctype   := "double",
    instr   := [ vunpacklo_4x64f, vunpackhi_4x64f, vshuffle_4x64f, vperm2_4x64f, vpermf128_4x64f, vperm_4x64f, vblend_4x64f ],
               #vuunpacklo_4x64f, vuunpackhi_4x64f, vushuffle_4x64f, vuperm2_4x64f, vupermf128_4x64f
    bits    := 64,
    splopts := rec(precision := "double"),

    freshU  := self >> var.fresh_t("U", self.t),

    dupload := (y, x) -> assign(y, vdup(x, 4)),
    mul_cx := (self, opts) >>
        ((y, x, c) -> let( u1 := self.freshU(), u2 := self.freshU(), u3 := self.freshU(),
            decl([u1, u2, u3], chain(
                assign(u1, mul(x, vunpacklo_4x64f(c, c))),
                assign(u2, vshuffle_4x64f(x, x, [2,1,2,1])),
                assign(u3, mul(u2, vunpackhi_4x64f(c, c))),
                assign(y, addsub_4x64f(u1, u3))))
        )),

    mul_cx_conj := (self, opts) >> 
        ((y,x,c) -> let(
		u1 := self.freshU(), u2 := self.freshU(), v := self.freshU(),
                decl([u1, u2, v], chain(
                    assign(u1,             mul(x, vunpackhi_4x64f(c, c))),
                    assign(u2, mul(vshuffle_4x64f(x, x, [2,1,2,1]), vunpacklo_4x64f(c, c))),
                    assign(v, addsub_4x64f(u2, u1)),
                    assign(y, vshuffle_4x64f(v, v, [2,1,2,1]))
                )))),

    svload_init := (vt) -> [
        # load using subvectors of length 1
	#
	[
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), 
            decl([u1], chain(
                assign(u1, vload1sd_2x64f(x[1].toPtr(vt.t))),
                assign(y, vinsert_2l_4x64f(vt.zero(), u1, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), u2 := var.fresh_t("U", TVect(vt.t, 2)),
            decl([u1, u2], chain(
                assign(u1, vload1sd_2x64f(x[1].toPtr(vt.t))),
                assign(u2, vload_1h_2x64f(u1, x[2].toPtr(vt.t))),
                assign(y, vinsert_2l_4x64f(vt.zero(), u2, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), u2 := var.fresh_t("U", TVect(vt.t, 2)),
                        u3 := var.fresh_t("U", TVect(vt.t, 2)), v1 := var.fresh_t("U", vt),
            decl([u1, u2, u3, v1], chain(
                assign(u1, vload1sd_2x64f(x[1].toPtr(vt.t))),
                assign(u2, vload_1h_2x64f(u1, x[2].toPtr(vt.t))),
                assign(u3, vload1sd_2x64f(x[3].toPtr(vt.t))),
                assign(v1, vinsert_2l_4x64f(vt.zero(), u2, [0])),
                assign(y, vinsert_2l_4x64f(v1, u3, [1]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), u2 := var.fresh_t("U", TVect(vt.t, 2)),
                        u3 := var.fresh_t("U", TVect(vt.t, 2)), u4 := var.fresh_t("U", TVect(vt.t, 2)),
                        v1 := var.fresh_t("U", vt),
            decl([u1, u2, u3, v1], chain(
                assign(u1, vload1sd_2x64f(x[1].toPtr(vt.t))),
                assign(u2, vload_1h_2x64f(u1, x[2].toPtr(vt.t))),
                assign(u3, vload1sd_2x64f(x[3].toPtr(vt.t))),
                assign(u4, vload_1h_2x64f(u3, x[4].toPtr(vt.t))),
                assign(v1, vinsert_2l_4x64f(vt.zero(), u2, [0])),
                assign(y, vinsert_2l_4x64f(v1, u4, [1]))
            )))
        ],
        # load using subvectors of length 2
	#
        [
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)),
            decl([u1], chain(
                assign(u1, vloadu_2x64f(x[1].toPtr(vt.t))),
                assign(y, vinsert_2l_4x64f(vt.zero(), u1, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), u2 := var.fresh_t("U", TVect(vt.t, 2)),
                        v1 := var.fresh_t("U", vt),
            decl([u1, u2, v1], chain(
                assign(u1, vloadu_2x64f(x[1].toPtr(vt.t))),
                assign(u2, vloadu_2x64f(x[2].toPtr(vt.t))),
                assign(v1, vinsert_2l_4x64f(vt.zero(), u1, [0])),
                assign(y, vinsert_2l_4x64f(v1, u2, [1]))
            )))
        ]
    ],
    svstore_init := (vt) -> [
        # store using subvectors of length 1
        [
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)),
            decl([u1], chain(
                assign(u1, vextract_2l_4x64f(x, [0])),
                vstore_1l_2x64f(y[1].toPtr(vt.t), u1)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)),
            decl([u1], chain(
                assign(u1, vextract_2l_4x64f(x, [0])),
                vstore_1l_2x64f(y[1].toPtr(vt.t), u1),
                vstore_1h_2x64f(y[2].toPtr(vt.t), u1)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), u2 := var.fresh_t("U", TVect(vt.t, 2)),
            decl([u1, u2], chain(
                assign(u1, vextract_2l_4x64f(x, [0])),
                vstore_1l_2x64f(y[1].toPtr(vt.t), u1),
                vstore_1h_2x64f(y[2].toPtr(vt.t), u1),
                assign(u2, vextract_2l_4x64f(x, [1])),
                vstore_1l_2x64f(y[3].toPtr(vt.t), u2)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), u2 := var.fresh_t("U", TVect(vt.t, 2)),
            decl([u1, u2], chain(
                assign(u1, vextract_2l_4x64f(x, [0])),
                vstore_1l_2x64f(y[1].toPtr(vt.t), u1),
                vstore_1h_2x64f(y[2].toPtr(vt.t), u1),
                assign(u2, vextract_2l_4x64f(x, [1])),
                vstore_1l_2x64f(y[3].toPtr(vt.t), u2),
                vstore_1h_2x64f(y[4].toPtr(vt.t), u2)
            )))
        ],
        # store using subvectors of length 2
        [
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)),
            decl([u1], chain(
                assign(u1, vextract_2l_4x64f(x, [0])),
                vstoreu_2x64f(y[1].toPtr(vt.t), u1)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), u2 := var.fresh_t("U", TVect(vt.t, 2)),
            decl([u1, u2], chain(
                assign(u1, vextract_2l_4x64f(x, [0])),
                vstoreu_2x64f(y[1].toPtr(vt.t), u1),
                assign(u2, vextract_2l_4x64f(x, [1])),
                vstoreu_2x64f(y[2].toPtr(vt.t), u2)
            )))
        ]],

    svload  := ~.svload_init(~.t),
    svstore := ~.svstore_init(~.t),

    loadc := (self, sv, opts) >> Cond( 
        sv = 4, 
            ((y,x) -> assign(y, vloadu_4x64f(x.toPtr(self.t.t)))),
        sv = 2, 
            ((y,x) -> assign(y, vinsert_2l_4x64f(self.t.zero(), vloadu_2x64f(x.toPtr(self.t.t)), [0]))),
        # else
	Checked( sv=1,
	    ((y,x) -> assign(y, vinsert_2l_4x64f(self.t.zero(), vload1sd_2x64f(x.toPtr(self.t.t)), [0]))))),

    loadc_align := (self, sv, align, opts) >> Cond(
        align = 0 and sv = 8,
            ((y,x,addr) -> assign(y, deref(nth(x,addr).toPtr(self.t)))),
        align mod 2 = 0 and sv = 2,
            ((y,x,addr) -> assign(y, vinsert_2l_4x64f(self.t.zero(), deref(nth(x,addr).toPtr(TVect(self.t.t, 2))), [0]))),
        sv = 4,
            ((y,x,addr) -> assign(y, vloadu_4x64f(nth(x, addr).toPtr(self.t.t)))),
        sv = 2, 
            ((y,x,addr) -> assign(y, vinsert_2l_4x64f(self.t.zero(), vloadu_2x64f(nth(x, addr).toPtr(self.t.t)), [0]))),
        # else
        Checked( sv=1,
	    ((y,x,addr) -> assign(y, vinsert_2l_4x64f(self.t.zero(), vload1sd_2x64f(nth(x, addr).toPtr(self.t.t)), [0]))))),

    storec_init := (vt) -> [
        ((y,x) -> vstore_1l_2x64f(y.toPtr(vt.t), vextract_2l_4x64f(x, [0]))),
        ((y,x) -> vstoreu_2x64f(y.toPtr(vt.t), vextract_2l_4x64f(x, [0]))), 
        ((y,x) -> vstoremask_4x64f(y.toPtr(vt.t), x, vstoremask_4x64f.mask(3))),
        ((y,x) -> vstoreu_4x64f(y.toPtr(vt.t), x))
    ],

    storec := ~.storec_init(~.t),

    swap_cx := (y, x, opts) -> assign(y, vushuffle_4x64f(x, [2,1,2,1])),
    RCVIxJ2 := (y, x, opts) -> assign(y, vupermf128_4x64f(x, [2,1])),

    bin_shrev := (self, y, x, opts) >> let(
        u1 := self.freshU(), u2 := self.freshU(),
        decl( [u1, u2], chain(
            assign(u1, vupermf128_4x64f(x[2], [2,1])),
            assign(u2, vblend_4x64f(x[1], x[2], [1,1,2,2])),
            assign(y,  vblend_4x64f(u1, u2, [2,1,2,1]))))),
));


#==============================================================================================
Class(AVX_8x32f, AVX_Intel, rec(
    info     := "AVX 8 x 32-bit float",
    countrec := rec(
        ops := [
            [ add, sub, addsub_8x32f, addsub_4x32f, hadd_4x32f ],
	    [ mul ], 
	    [ fmaddsub_8x32f ],
            [ vunpacklo_8x32f,  vunpackhi_8x32f,   vpermf128_8x32f,  vperm2_8x32f,  vshuffle_8x32f, 
              vinsert_4l_8x32f, vextract_4l_8x32f, vperm_8x32f,      vpermv_8x32f,
              vunpacklo_4x32f,  vunpackhi_4x32f,   vshuffle_4x32f,   vushuffle_4x32f,
              vuunpacklo_8x32f, vuunpackhi_8x32f,  vupermf128_8x32f, vushuffle_8x32f, 
              vldup_8x32f, vhdup_8x32f, vblend_8x32f ],
            [ vloadmask_8x32f, vstoremask_8x32f,
              vload1_4x32f,    vload_2l_4x32f,  vload_2h_4x32f,  vloadu_4x32f,  vloadu2_4x32f,
              vstore1_4x32f,   vstore_2l_4x32f, vstore_2h_4x32f, vstoreu_4x32f, vstoreu2_4x32f,  
              vstoremsk_4x32f ],
            [ deref, vloadu_8x32f, vstoreu_8x32f ],
            Value   # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[fmas]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]+2*opcount[3]
    ),

    includes     := () -> ["<include/omega32.h>"] :: _AVXINTRIN(),
    active       := true,
    isFixedPoint := false,
    isFloat      := true,

    v       := 8,
    t       := TVect(T_Real(32), 8), 
    ctype   := "float",
    instr   := [ vunpacklo_8x32f,  vunpackhi_8x32f,  vpermf128_8x32f,  vshuffle_8x32f,  vperm_8x32f,
                 vuunpacklo_8x32f, vuunpackhi_8x32f, vupermf128_8x32f, vushuffle_8x32f, vblend_8x32f,
                 vcxtr_8x32f],

    bits    := 32,
    splopts := rec(precision := "single"),
    
    freshU  := self >> var.fresh_t("U", self.t),

    dupload := (y, x) -> assign(y, vdup(x, 8)),

    mul_cx := (self, opts) >>
        ((y, x, c) -> let( u1 := self.freshU(), u2 := self.freshU(), u3 := self.freshU(),
            decl([u1, u2, u3], chain(
                assign(u1, mul(x, vldup_8x32f(c))),
                assign(u2, vshuffle_8x32f(x, x, [2,1,4,3])),
                assign(u3, mul(u2, vhdup_8x32f(c))),
                assign(y,  addsub_8x32f(u1, u3))))
        )),

    mul_cx_conj := (self, opts) >> 
        ((y,x,c) -> let( u1 := self.freshU(), u2 := self.freshU(), v := self.freshU(),
            decl([u1, u2, v], chain(
                assign(u1, mul(x, vhdup_8x32f(c))),
                assign(u2, mul(vshuffle_8x32f(x, x, [2,1,4,3]), vldup_8x32f(c))),
                assign(v, addsub_8x32f(u2, u1)),
                assign(y, vshuffle_8x32f(v, v, [2,1,4,3]))))
        )),

    svload_init := (vt) -> [
        # Load using subvectors of length 1.
	#
	[      
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), 
            decl([u1], chain(
                assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                assign(y,  vinsert_4l_8x32f(vt.zero(), u1, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                assign(u2, vinsert_4x32f(u1, deref(x[2].toPtr(T_Int(32))), 2)),
                assign(y,  vinsert_4l_8x32f(vt.zero(), u2, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
                     u3 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2, u3], chain(
                assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                assign(u2, vinsert_4x32f(u1, deref(x[2].toPtr(T_Int(32))), 2)),
                assign(u3, vinsert_4x32f(u2, deref(x[3].toPtr(T_Int(32))), 3)),
                assign(y,  vinsert_4l_8x32f(vt.zero(), u3, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
                     u3 := var.fresh_t("U", TVect(vt.t, 4)), u4 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2, u3, u4], chain(
                assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                assign(u2, vinsert_4x32f(u1, deref(x[2].toPtr(T_Int(32))), 2)),
                assign(u3, vinsert_4x32f(u2, deref(x[3].toPtr(T_Int(32))), 3)),
                assign(u4, vinsert_4x32f(u3, deref(x[4].toPtr(T_Int(32))), 4)),
                assign(y,  vinsert_4l_8x32f(vt.zero(), u4, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
                     u3 := var.fresh_t("U", TVect(vt.t, 4)), u4 := var.fresh_t("U", TVect(vt.t, 4)),
                     u5 := var.fresh_t("U", TVect(vt.t, 4)),
                     v1 := var.fresh_t("U", vt),
            decl([u1, u2, u3, u4, u5, v1], chain(
                assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                assign(u2, vinsert_4x32f(u1, deref(x[2].toPtr(T_Int(32))), 2)),
                assign(u3, vinsert_4x32f(u2, deref(x[3].toPtr(T_Int(32))), 3)),
                assign(u4, vinsert_4x32f(u3, deref(x[4].toPtr(T_Int(32))), 4)),
                assign(v1, vinsert_4l_8x32f(vt.zero(), u4, [0])),
                assign(u5, vload1_4x32f(x[5].toPtr(vt.t))),
                assign(y,  vinsert_4l_8x32f(v1, u5, [1]))
            ))),

        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
                     u3 := var.fresh_t("U", TVect(vt.t, 4)), u4 := var.fresh_t("U", TVect(vt.t, 4)),
                     u5 := var.fresh_t("U", TVect(vt.t, 4)), u6 := var.fresh_t("U", TVect(vt.t, 4)),
                     v1 := var.fresh_t("U", vt),
            decl([u1, u2, u3, u4, u5, u6, v1], chain(
                assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                assign(u2, vinsert_4x32f(u1, deref(x[2].toPtr(T_Int(32))), 2)),
                assign(u3, vinsert_4x32f(u2, deref(x[3].toPtr(T_Int(32))), 3)),
                assign(u4, vinsert_4x32f(u3, deref(x[4].toPtr(T_Int(32))), 4)),
                assign(v1, vinsert_4l_8x32f(vt.zero(), u4, [0])),
                assign(u5, vload1_4x32f(x[5].toPtr(vt.t))),
                assign(u6, vinsert_4x32f(u5, deref(x[6].toPtr(T_Int(32))), 2)),
                assign(y,  vinsert_4l_8x32f(v1, u6, [1]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
                     u3 := var.fresh_t("U", TVect(vt.t, 4)), u4 := var.fresh_t("U", TVect(vt.t, 4)),
                     u5 := var.fresh_t("U", TVect(vt.t, 4)), u6 := var.fresh_t("U", TVect(vt.t, 4)),
                     u7 := var.fresh_t("U", TVect(vt.t, 4)),
                     v1 := var.fresh_t("U", vt),
            decl([u1, u2, u3, u4, u5, u6, u7, v1], chain(
                assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                assign(u2, vinsert_4x32f(u1, deref(x[2].toPtr(T_Int(32))), 2)),
                assign(u3, vinsert_4x32f(u2, deref(x[3].toPtr(T_Int(32))), 3)),
                assign(u4, vinsert_4x32f(u3, deref(x[4].toPtr(T_Int(32))), 4)),
                assign(v1, vinsert_4l_8x32f(vt.zero(), u4, [0])),
                assign(u5, vload1_4x32f(x[5].toPtr(vt.t))),
                assign(u6, vinsert_4x32f(u5, deref(x[6].toPtr(T_Int(32))), 2)),
                assign(u7, vinsert_4x32f(u6, deref(x[7].toPtr(T_Int(32))), 3)),
                assign(y,  vinsert_4l_8x32f(v1, u7, [1]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
                     u3 := var.fresh_t("U", TVect(vt.t, 4)), u4 := var.fresh_t("U", TVect(vt.t, 4)),
                     u5 := var.fresh_t("U", TVect(vt.t, 4)), u6 := var.fresh_t("U", TVect(vt.t, 4)),
                     u7 := var.fresh_t("U", TVect(vt.t, 4)), u8 := var.fresh_t("U", TVect(vt.t, 4)),
                     v1 := var.fresh_t("U", vt),
            decl([u1, u2, u3, u4, u5, u6, u7, u8, v1], chain(
                assign(u1, vload1_4x32f(x[1].toPtr(vt.t))),
                assign(u2, vinsert_4x32f(u1, deref(x[2].toPtr(T_Int(32))), 2)),
                assign(u3, vinsert_4x32f(u2, deref(x[3].toPtr(T_Int(32))), 3)),
                assign(u4, vinsert_4x32f(u3, deref(x[4].toPtr(T_Int(32))), 4)),
                assign(v1, vinsert_4l_8x32f(vt.zero(), u4, [0])),
                assign(u5, vload1_4x32f(x[5].toPtr(vt.t))),
                assign(u6, vinsert_4x32f(u5, deref(x[6].toPtr(T_Int(32))), 2)),
                assign(u7, vinsert_4x32f(u6, deref(x[7].toPtr(T_Int(32))), 3)),
                assign(u8, vinsert_4x32f(u7, deref(x[8].toPtr(T_Int(32))), 4)),
                assign(y,  vinsert_4l_8x32f(v1, u8, [1]))
            )))
        ],
        # Load using subvectors of length 2
	#
        [
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1], chain(
                assign(u1, vload_2l_4x32f(vzero_4x32f(), x[1].toPtr(TVect(vt.t, 2)))),
                assign(y,  vinsert_4l_8x32f(vt.zero(), u1, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vload_2l_4x32f(vzero_4x32f(), x[1].toPtr(TVect(vt.t, 2)))),
                assign(u2, vload_2h_4x32f(u1, x[2].toPtr(TVect(vt.t, 2)))),
                assign(y,  vinsert_4l_8x32f(vt.zero(), u2, [0]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
                     u3 := var.fresh_t("U", TVect(vt.t, 4)),
                     z1 := var.fresh_t("Z", TVect(vt.t, 4)),
                     v1 := var.fresh_t("U", vt),
            decl([u1, u2, u3, v1], chain(
                assign(z1, vzero_4x32f()),
                assign(u1, vload_2l_4x32f(z1, x[1].toPtr(TVect(vt.t, 2)))),
                assign(u2, vload_2h_4x32f(u1, x[2].toPtr(TVect(vt.t, 2)))),
                assign(v1, vinsert_4l_8x32f(vt.zero(), u2, [0])),
                assign(u3, vload_2l_4x32f(z1, x[3].toPtr(TVect(vt.t, 2)))),
                assign(y,  vinsert_4l_8x32f(v1, u3, [1]))
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
                     u3 := var.fresh_t("U", TVect(vt.t, 4)), u4 := var.fresh_t("U", TVect(vt.t, 4)),
                     z1 := var.fresh_t("Z", TVect(vt.t, 4)),
                     v1 := var.fresh_t("U", vt),
            decl([u1, u2, u3, u4, v1], chain(
                assign(z1, vzero_4x32f()),
                assign(u1, vload_2l_4x32f(z1, x[1].toPtr(TVect(vt.t, 2)))),
                assign(u2, vload_2h_4x32f(u1, x[2].toPtr(TVect(vt.t, 2)))),
                assign(v1, vinsert_4l_8x32f(vt.zero(), u2, [0])),
                assign(u3, vload_2l_4x32f(z1, x[3].toPtr(TVect(vt.t, 2)))),
                assign(u4, vload_2h_4x32f(u3, x[4].toPtr(TVect(vt.t, 2)))),
                assign(y,  vinsert_4l_8x32f(v1, u4, [1]))
            ))),
        ],
        # Load using subvectors of length 4.
	#
        [
        (y,x,opts) -> 
            assign(y, vinsert_4l_8x32f(vt.zero(), vloadu_4x32f(x[1].toPtr(vt.t)), [0])),
            
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), 
            decl([u1], chain(
                assign(u1, vinsert_4l_8x32f(vt.zero(), vloadu_4x32f(x[1].toPtr(vt.t)), [0])),
                assign(y,  vinsert_4l_8x32f(u1,        vloadu_4x32f(x[2].toPtr(vt.t)), [1]))
            )))
        ]
    ],

    svstore_init := (vt) -> [
        # Store using subvectors of length 1.
	# 
        [
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore1_4x32f(y[1].toPtr(vt.t), u1)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore1_4x32f(y[1].toPtr(vt.t), u1),
                vextract_4x32f(y[2].toPtr(T_Int(32)), u1, 2)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore1_4x32f(y[1].toPtr(vt.t), u1),
                vextract_4x32f(y[2].toPtr(T_Int(32)), u1, 2),
                vextract_4x32f(y[3].toPtr(T_Int(32)), u1, 3)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore1_4x32f(y[1].toPtr(vt.t), u1),
                vextract_4x32f(y[2].toPtr(T_Int(32)), u1, 2),
                vextract_4x32f(y[3].toPtr(T_Int(32)), u1, 3),
                vextract_4x32f(y[4].toPtr(T_Int(32)), u1, 4)
            ))),
    
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore1_4x32f(y[1].toPtr(vt.t), u1),
                vextract_4x32f(y[2].toPtr(T_Int(32)), u1, 2),
                vextract_4x32f(y[3].toPtr(T_Int(32)), u1, 3),
                vextract_4x32f(y[4].toPtr(T_Int(32)), u1, 4),
                assign(u2, vextract_4l_8x32f(x, [1])),
                vstore1_4x32f(y[5].toPtr(vt.t), u2)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore1_4x32f(y[1].toPtr(vt.t), u1),
                vextract_4x32f(y[2].toPtr(T_Int(32)), u1, 2),
                vextract_4x32f(y[3].toPtr(T_Int(32)), u1, 3),
                vextract_4x32f(y[4].toPtr(T_Int(32)), u1, 4),
                assign(u2, vextract_4l_8x32f(x, [1])),
                vstore1_4x32f(y[5].toPtr(vt.t), u2),
                vextract_4x32f(y[6].toPtr(T_Int(32)), u2, 2)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore1_4x32f(y[1].toPtr(vt.t), u1),
                vextract_4x32f(y[2].toPtr(T_Int(32)), u1, 2),
                vextract_4x32f(y[3].toPtr(T_Int(32)), u1, 3),
                vextract_4x32f(y[4].toPtr(T_Int(32)), u1, 4),
                assign(u2, vextract_4l_8x32f(x, [1])),
                vstore1_4x32f(y[5].toPtr(vt.t), u2),
                vextract_4x32f(y[6].toPtr(T_Int(32)), u2, 2),
                vextract_4x32f(y[7].toPtr(T_Int(32)), u2, 3)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore1_4x32f(y[1].toPtr(vt.t), u1),
                vextract_4x32f(y[2].toPtr(T_Int(32)), u1, 2),
                vextract_4x32f(y[3].toPtr(T_Int(32)), u1, 3),
                vextract_4x32f(y[4].toPtr(T_Int(32)), u1, 4),
                assign(u2, vextract_4l_8x32f(x, [1])),
                vstore1_4x32f(y[5].toPtr(vt.t), u2),
                vextract_4x32f(y[6].toPtr(T_Int(32)), u2, 2),
                vextract_4x32f(y[7].toPtr(T_Int(32)), u2, 3),
                vextract_4x32f(y[8].toPtr(T_Int(32)), u2, 4)
            )))
        ],
        # Store using subvectors of length 2.
	#
        [
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore_2l_4x32f(y[1].toPtr(TVect(vt.t, 2)), u1)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore_2l_4x32f(y[1].toPtr(TVect(vt.t, 2)), u1),
                vstore_2h_4x32f(y[2].toPtr(TVect(vt.t, 2)), u1)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore_2l_4x32f(y[1].toPtr(TVect(vt.t, 2)), u1),
                vstore_2h_4x32f(y[2].toPtr(TVect(vt.t, 2)), u1),
                assign(u2, vextract_4l_8x32f(x, [1])),
                vstore_2l_4x32f(y[3].toPtr(TVect(vt.t, 2)), u2)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstore_2l_4x32f(y[1].toPtr(TVect(vt.t, 2)), u1),
                vstore_2h_4x32f(y[2].toPtr(TVect(vt.t, 2)), u1),
                assign(u2, vextract_4l_8x32f(x, [1])),
                vstore_2l_4x32f(y[3].toPtr(TVect(vt.t, 2)), u2),
                vstore_2h_4x32f(y[4].toPtr(TVect(vt.t, 2)), u2)
            )))
        ],
        # Store using subvectors of length 4.
	#
        [
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstoreu_4x32f(y[1].toPtr(vt.t), u1)
            ))),
        (y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 4)), u2 := var.fresh_t("U", TVect(vt.t, 4)),
            decl([u1, u2], chain(
                assign(u1, vextract_4l_8x32f(x, [0])),
                vstoreu_4x32f(y[1].toPtr(vt.t), u1),
                assign(u2, vextract_4l_8x32f(x, [1])),
                vstoreu_4x32f(y[2].toPtr(vt.t), u2)
            )))
        ]
    ],

    svload  := ~.svload_init(~.t),
    svstore := ~.svstore_init(~.t),

    loadc := (self, sv, opts) >> Cond( 
        sv = 8, 
            ((y,x) -> assign(y, vloadu_8x32f(x.toPtr(self.t.t)))),
        sv = 4,
            ((y,x) -> assign(y, vinsert_4l_8x32f(self.t.zero(), vloadu_4x32f(x.toPtr(self.t.t)), [0]))),
        sv = 2,
            ((y,x) -> assign(y, vinsert_4l_8x32f(self.t.zero(), vload_2l_4x32f(vzero_4x32f(), x.toPtr(TVect(self.t.t, 2))), [0]))),
        # else
            ((y,x) -> assign(y, vinsert_4l_8x32f(self.t.zero(), vload1_4x32f(x.toPtr(self.t.t)), [0])))),

    loadc_align := (self, sv, align, opts) >> Cond(
        align = 0 and sv = 8,
            ((y,x,addr) -> assign(y, deref(nth(x,addr).toPtr(self.t)))),
        align mod 4 = 0 and sv = 4,
            ((y,x,addr) -> assign(y, vinsert_4l_8x32f(self.t.zero(), deref(nth(x,addr).toPtr(TVect(self.t.t, 4))), [0]))),
        sv = 8,
            ((y,x,addr) -> assign(y, vloadu_8x32f(nth(x,addr).toPtr(self.t.t)))),
        sv = 4,
            ((y,x,addr) -> assign(y, vinsert_4l_8x32f(self.t.zero(), vloadu_4x32f(nth(x,addr).toPtr(self.t.t)), [0]))),
        sv = 2,
            ((y,x,addr) -> assign(y, vinsert_4l_8x32f(self.t.zero(), vload_2l_4x32f(vzero_4x32f(), nth(x,addr).toPtr(TVect(self.t.t, 2))), [0]))),
        # else
            ((y,x,addr) -> assign(y, vinsert_4l_8x32f(self.t.zero(), vload1_4x32f(nth(x,addr).toPtr(self.t.t)), [0])))),

    storec_init := (vt) -> [
        ((y,x) -> vstore1_4x32f(y.toPtr(vt.t), vextract_4l_8x32f(x, [0]))),
        ((y,x) -> vstore_2l_4x32f(y.toPtr(TVect(vt.t, 2)), vextract_4l_8x32f(x, [0]))),
        ((y,x) -> vstoremask_8x32f(y.toPtr(vt.t), x, vstoremask_8x32f.mask(3))),
        ((y,x) -> vstoreu_4x32f(y.toPtr(vt.t), vextract_4l_8x32f(x, [0]))),
        ((y,x) -> vstoremask_8x32f(y.toPtr(vt.t), x, vstoremask_8x32f.mask(5))),
        ((y,x) -> vstoremask_8x32f(y.toPtr(vt.t), x, vstoremask_8x32f.mask(6))),
        ((y,x) -> vstoremask_8x32f(y.toPtr(vt.t), x, vstoremask_8x32f.mask(7))),
        ((y,x) -> vstoreu_8x32f(y.toPtr(vt.t), x))
    ],
    
    storec  := ~.storec_init(~.t),

    swap_cx := (y, x, opts) -> assign(y, vshuffle_8x32f(x, x, [2,1,4,3])),
    RCVIxJ2 := (y, x, opts) -> assign(y, vperm_8x32f(x, [3,4,1,2])),

    # VJxI implements VJxI(m,v) which is J(m) /tensor I(v/m)
    VJxI := (m, y, x, opts) -> Cond( m=1, assign(y, x),         
                                     m=2, assign(y, vupermf128_8x32f(x, [2,1])),
                                     m=4, assign(y, vperm_8x32f(vupermf128_8x32f(x, [2,1]), [3,4,1,2])),
                                     m=8, assign(y, vperm_8x32f(vupermf128_8x32f(x, [2,1]), [4,3,2,1])),
                                     Error("unexpected m")),

    bin_shrev := (self, y, x, opts) >> let(
        u1 := self.freshU(), u2 := self.freshU(), u3 := self.freshU(),
        decl( [u1, u2, u3], chain(
            assign(u1, vupermf128_8x32f(x[2], [2,1])),                # [a,b,c,d,e,f,g,h] -> [e,f,g,h,a,b,c,d]
            assign(u2, vblend_8x32f(x[1], x[2], [1,1,1,1,2,2,2,2])),  # -> [x11,x12,x13,x14,e,f,g,h]
            assign(u3, vushuffle_8x32f(u1, [1,4,3,2])),               # [e,f,g,h,a,b,c,d] -> [e,h,g,f,a,d,c,b]
            assign(y,  vblend_8x32f(u3, u2, [2,1,1,1,2,1,1,1]))))),   # -> [x11,h,g,f,e,d,c,b]

));

#==============================================================================================

SIMD_ISA_DB.addISA(AVX_4x64f);
SIMD_ISA_DB.addISA(AVX_8x32f);
