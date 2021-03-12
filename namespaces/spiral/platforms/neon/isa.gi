
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
# NEON 64-bit and 128-bit SIMD ISA Definitions
#
# legend: v = ok, ! = missing, u = added, untested
    # Required:
    #  v active
    #  v bin_shl1  bin_shl2  bin_shr1  bin_shr2  bin_shrev
    #  v bits, ctype, t, v
    #  v countrec
    #  v includes
    #  v info
    #  v instr
    #  v isFixedPoint, isFloat
    #  v loadCont (uses sv loads, ARM does not have masked loads)
    #  v mul_cx, mul_cx_conj
    #  u reverse
    #  v splopts
    #  v storeCont (uses sv stores, ARM does not have masked stores)
    #  v svload
    #  v svstore
    #  v RCVIxJ2
    #  v dupload, duploadn
    #  ! hadd
    #  v swap_cx
    #  v vzero

    # Fixed point:
    #  fracbits
    #  saturatedArithmetic

    # Viterbi:
    #  ! interleavedmask, hmin, average (?), isSigned

Class(SIMD_NEON, SIMD_ISA, rec(   
    file := "neon",
    commonIncludes := self >> [],
    active       := true,
    isFixedPoint := false,
    isFloat      := true,
    ctype        := "float32_t",
    stype        := "__attribute__ ((aligned(16))) float32_t",
    bits         := 32,
    useDeref     := false,
    splopts      := rec(precision := "single"),
    arrayDataModifier := "__attribute__ ((aligned(16)))",
    arrayBufModifier  := "static __attribute__ ((aligned(16)))",
    declareConstants := true,
    threeOps         := false,
    fma		     := true,
    vzero := self >> self.t.zero(),
    unparser := NEONUnparser,
    compileStrategy := self >> BaseIndicesCS
      :: When(self.fma, [DoFMA], [])
      :: [ MarkDefUse, #
           (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
           MarkDefUse, #
           (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))) ]
      :: When(self.threeOps, [DoThreeOp], [])
      :: [ Compile.declareVars,
           (c, opts) -> opts.vector.isa.fixProblems(c, opts),
           HashConsts ],

    loadCont := (self, n, y, yofs, x, xofs, xofs_align, opts) >> let(
	nn := _unwrap(n), 
	yy := vtref(self.t, y, yofs),
	assign(yy, deref(nth(x, xofs).toPtr(self.t)))),

    storeCont := (self, n, y, yofs, yofs_align, x, xofs, opts) >> let(
	nn := _unwrap(n),
	xx := vtref(self.t, x, xofs),
	assign(deref(nth(y, yofs).toPtr(self.t)), xx)),

    storeContAcc := (self, n, y, yofs, yofs_align, x, xofs, opts) >> let(
	a  := _unwrap(yofs_align),
	nn := _unwrap(n),
	xx := vtref(self.t, x, xofs),
	t  := TempVec(TArray(xx.t.t, xx.t.size)),
	decl([t], chain(
	    self.loadCont(nn, t, 0, y, yofs, yofs_align, opts), 
	    assign(vtref(self.t, t, 0), vtref(self.t, t, 0) + xx), 
	    self.storeCont(nn, y, yofs, yofs_align, t, 0, opts)))),
));

Declare(NEON_HALF);

Class(NEON_HALF, SIMD_NEON, rec(
    info := "NEON 2 x floating point",
    v     := 2,
    t     := TVect(T_Real(32), 2),
    freshU  := self >> var.fresh_t("u", TVect(self.t, 2)),
    freshT  := self >> var.fresh_t("t", self.t),
    instr   := [vunpacklo_half,  vunpackhi_half, vrev_half ],

    includes := self >> self.commonIncludes() :: 
        ["<arm_neon.h>", "<include/omega32_neon.h>", "<include/mm_malloc.h>"], 

    dupload := (y, x) -> assign(y, vdup(x, 2)),
    duploadn := (y, x, n) -> assign(y, vdup_lane_half(x, n)),

    reverse := (y,x) -> assign(vref(y,0,2), vrev_half(vref(x,0,2))),
    RCVIxJ2 := (y,x,opts) -> assign(y, vrev_half(x)),
   
    # this is shift "right" in Spiral notation (and ARM too)
    # [a b] -> [0 a]
    bin_shl1 := (self, y,x,opts) >> assign(y, vext_half(self.t.zero(), x, 1)),
    # this is shift "left" in Spiral notation (and ARM too)
    # [a b] -> [b 0]
    bin_shr1 := (self, y,x,opts) >> assign(y, vext_half(x, self.t.zero(), 1)),

    # this is shift "right" in Spiral notation (and ARM too)
    # [a b], [c,d] -> [b c]
    bin_shl2 := (self, y,x,opts) >> assign(y, vext_half(x[1], x[2], 1)),
    # this is shift "left" in Spiral notation (and ARM too)
    # [a b], [c,d] -> [b c]
    bin_shr2 := (self, y,x,opts) >> assign(y, vext_half(x[1], x[2], 1)),

    # [a b], [c,d] -> [b c]
    bin_shrev := (self, y,x,opts) >> assign(y, vext_half(x[1], x[2], 1)),

    countrec := rec(
        ops := [
           [ add, sub ],
	   [ mul, vmulcx_half ],
           [ fma, fms, nfma ],
           [ vunpacklo_half, vunpackhi_half, vrev_half, vswapcx_half, vtrnq_half, vextract_half,
	     vdup_lane_half ],
           [ vload1_half ],
           [ deref, nth, ],
           Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]","[mults]","[fmas]","[vperms]","[svldst]","[vldst]","[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),

   loadCont := (self, n, y, yofs, x, xofs, xofs_align, opts) >> let(
	nn := _unwrap(n), 
	yy := vtref(self.t, y, yofs),
	p0 := nth(x, xofs).toPtr(self.t.t),
	Cond(
	    nn=1, assign(yy, vload1_half(p0, self.vzero(), 1)), 
	    nn=2, assign(yy, deref(nth(x, xofs).toPtr(self.t)))
	)
    ),

    storeCont := (self, n, y, yofs, yofs_align, x, xofs, opts) >> let(
	nn := _unwrap(n),
	xx := vtref(self.t, x, xofs),
	p0 := nth(y, yofs).toPtr(self.t.t),
	Cond(
	    nn=1, vstore1_half(p0, xx, 1),
	    nn=2, assign(deref(nth(y, yofs).toPtr(self.t)), xx)
	)
    ),

    svload := [
        # load using subvectors of length 1
	[   (y,x,opts) -> assign(y, vload1_half(x[1].toPtr(NEON_HALF.t.t),
		                                NEON_HALF.vzero(), 1)),

            (y,x,opts) -> let(u := var.fresh_t("T", NEON_HALF.t),
                decl([u], chain(
                        assign(u, vload1_half(x[1].toPtr(NEON_HALF.t.t), 
				              NEON_HALF.vzero(), 1)),
                        assign(y, vload1_half(x[2].toPtr(NEON_HALF.t.t), u, 2))
                )))
	],
        # load using subvectors of length 2
        [(y,x,opts) -> assign(y, nth(x[1].toPtr(TVect(TReal, 2)), 0)) ]
    ],

    svstore := [
        [  (y,x,opts) -> vstore1_half(y[1].toPtr(NEON_HALF.t.t), x, 1),
           (y,x,opts) -> chain(
               vstore1_half(y[1].toPtr(NEON_HALF.t.t), x, 1),
               vstore1_half(y[2].toPtr(NEON_HALF.t.t), x, 2)
           )
        ],
        [  (y,x,opts) -> assign(nth(y[1].toPtr(NEON_HALF.t), 0), x) ]
    ],

    mul_cx := (self, opts) >> ((y,x,c) -> let(
	u := self.freshU(),  v1 := self.freshT(), v2 := self.freshT(), r1 := self.freshT(), 
	decl([u, v1, v2, r1], chain(
		assign(r1, self.t.value([-1.0,1.0])),
         	assign(u, vtrnq_half(c, c)),
		assign(v1, x * vextract_half(u, [0])),
		assign(v2, vswapcx_half(x) * vextract_half(u, [1])),
		assign(y, fma(v1, v2, r1))))
    )),

    mul_cx_conj := (self, opts) >> ((y,x,c) -> let(
	u := self.freshU(),  v1 := self.freshT(), v2 := self.freshT(), r1 := self.freshT(), 
	decl([u, v1, v2, r1], chain(
		assign(r1, self.t.value([1.0, -1.0])),
         	assign(u, vtrnq_half(c, c)),
		assign(v1, x * vextract_half(u, [0])),
		assign(v2, vswapcx_half(x) * vextract_half(u, [1])),
		assign(y, fma(v1, v2, r1))))
    )),

    swap_cx := (y, x, opts) -> assign(y, vswapcx_half(x)),
));

Class(NEON, SIMD_NEON, rec(
    info := "NEON 4 x floating point",

    countrec := rec(
        ops := [
            [ add, sub ],
	    [ mul, vmulcx_neon ],
            [ fma, fms, nfma ],
            [ vpacklo_neon,      vpackhi_neon,       vunpacklo_neon,    vunpackhi_neon,
              vunpacklolo2_neon, vunpacklohi2_neon,  vunpackhilo2_neon, vunpackhihi2_neon,
              vtransposelo_neon, vtransposehi_neon,  vrev_neon,         vswapcx_neon, 
              vuzpq_32f,         vzipq_32f,          vtrnq_32f,         vextract_neon_4x32f,
	      vdup_lane_neon ],
            [ vload1_neon, vload_half_neon, vcombine_neon ],
            [ deref, nth, ],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]","[mults]","[fmas]","[vperms]","[svldst]","[vldst]","[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),

    v     := 4,
    t     := TVect(T_Real(32), 4),
    freshU := self >> var.fresh_t("u", TVect(self.t, 2)),
    freshT := self >> var.fresh_t("t", self.t),

    instr := [vpacklo_neon,    vpackhi_neon, 
              vunpacklo_neon,  vunpackhi_neon, 
              vunpacklolo2_neon,   vunpacklohi2_neon,
              vunpackhilo2_neon,   vunpackhihi2_neon,
              vtransposelo_neon, vtransposehi_neon,
	      vrev_neon ],

    includes := self >> self.commonIncludes() :: 
        ["<arm_neon.h>", "<include/omega32_neon.h>", "<include/mm_malloc.h>"], 

    dupload  := (y, x) -> assign(y, vdup(x, 4)),
    duploadn := (y, x, n) -> assign(y, vdup_lane_neon(x, n)),

    reverse := (y,x) -> assign(vref(y,0,4), vrev_neon(vext_neon(vref(x,0,4), vref(x,0,4), 2))),
    RCVIxJ2 := (y,x,opts) -> assign(y, vext_neon(x, x, 2)),

    # this is shift "right" in Spiral notation (and ARM too)
    # [a b c d] -> [0 a b c]
    bin_shl1 := (self, y,x,opts) >> assign(y, vext_neon(self.t.zero(), x, 1)),
    # this is shift "left" in Spiral notation (and ARM too)
    # [a b c d] -> [b c d 0]
    bin_shr1 := (self, y,x,opts) >> assign(y, vext_neon(x, self.t.zero(), 3)),

    # this is shift "right" in Spiral notation (and ARM too)
    # [a b c d], [e f g h] -> [d e f g]
    bin_shl2 := (self, y,x,opts) >> assign(y, vext_neon(x[1], x[2], 1)),
    # this is shift "left" in Spiral notation (and ARM too)
    # [a b c d], [e f g h] -> [b c d e]
    bin_shr2 := (self, y,x,opts) >> assign(y, vext_neon(x[1], x[2], 3)),

    # [a b c d] [e f g h] -> [e d c b]
    # support for VO1dsJ(n, v)
    bin_shrev := (self, y, x, opts) >> let(
	u := var.fresh_t("T", self.t), 
	decl([u], chain(
		assign(u, vext_neon(x[1], x[2], 3)), # [ b c d e]
		assign(u, vext_neon(u, u, 2)),       # [ d e b c]
		assign(y, vrev_neon(u))))            # [ e d c b]
    ),

    loadCont := (self, n, y, yofs, x, xofs, xofs_align, opts) >> let(
	nn := _unwrap(n), 
	yy := vtref(self.t, y, yofs),
	p0 := nth(x, xofs).toPtr(self.t.t),
	p2 := nth(x, xofs+2).toPtr(self.t.t),
	Cond(
	    nn=1, 
	        assign(yy, vload1_neon(p0, self.vzero(), 1)), 
	    nn=2,
                assign(yy, vcombine_neon(
			       vload_half_neon(p0), 
			       NEON_HALF.vzero())),
	    nn=3,
                assign(yy, vcombine_neon(
			       vload_half_neon(p0), 
			       vload1_half(p2, NEON_HALF.vzero(), 1))),
	    nn=4,
	        assign(yy, deref(nth(x, xofs).toPtr(self.t)))
	)
    ),

    storeCont := (self, n, y, yofs, yofs_align, x, xofs, opts) >> let(
	nn := _unwrap(n),
	xx := vtref(self.t, x, xofs),
	p0 := nth(y, yofs).toPtr(self.t.t),
	p2 := nth(y, yofs+2).toPtr(self.t.t),
	Cond(
	    nn=1, 
		vstore1_neon(p0, xx, 1),
	    nn=2,
		vstore2lo_neon(p0, xx),
	    nn=3, chain(
		vstore2lo_neon(p0, xx),
		vstore1_neon(p2, xx, 3)),
	    nn=4,
	        assign(deref(nth(y, yofs).toPtr(self.t)), xx)
	)
    ),

    svload := [
        # load using subvectors of length 1
	[
            (y,x,opts) -> assign(y, vload1_neon(x[1].toPtr(NEON.t.t), NEON.vzero(), 1)),

            (y,x,opts) -> let(u := NEON.freshT(), 
                    decl([u], chain(
                            assign(u, vload1_neon(x[1].toPtr(NEON.t.t), NEON.vzero(), 1)),
                            assign(y, vload1_neon(x[2].toPtr(NEON.t.t), u,            2))))),
            (y,x,opts) -> let(u := NEON.freshT(), 
                    decl([u], chain(
                            assign(u, vload1_neon(x[1].toPtr(NEON.t.t), NEON.vzero(), 1)),
                            assign(u, vload1_neon(x[2].toPtr(NEON.t.t), u,            2)),
                            assign(y, vload1_neon(x[3].toPtr(NEON.t.t), u,            3))))),
            (y,x,opts) -> let(u := NEON.freshT(), 
                    decl([u], chain(
                            assign(u, vload1_neon(x[1].toPtr(NEON.t.t), NEON.vzero(), 1)),
                            assign(u, vload1_neon(x[2].toPtr(NEON.t.t), u,            2)),
                            assign(u, vload1_neon(x[3].toPtr(NEON.t.t), u,            3)),
                            assign(y, vload1_neon(x[4].toPtr(NEON.t.t), u,            4)))))
	],
        # load using subvectors of length 2
        [
            (y,x,opts) -> let(
		u := var.fresh_t("T", NEON_HALF.t),
                decl([u], chain(
                        assign(u, vload_half_neon(x[1].toPtr(NEON_HALF.t.t))),
                        assign(y, vcombine_neon(u, NEON_HALF.vzero()))))),
            (y,x,opts) -> let(
		u1 := var.fresh_t("T", NEON_HALF.t),
		u2 := var.fresh_t("T", NEON_HALF.t),
                decl([u1, u2] , chain(
                        assign(u1, vload_half_neon(x[1].toPtr(NEON_HALF.t.t))),
                        assign(u2, vload_half_neon(x[2].toPtr(NEON_HALF.t.t))),
			assign(y, vcombine_neon(u1, u2)))))
        ]],


    svstore := [ 
        # store using subvectors of length 1
        [  (y,x,opts) -> vstore1_neon(y[1].toPtr(NEON.t.t), x, 1),
           (y,x,opts) -> chain(
               vstore1_neon(y[1].toPtr(NEON.t.t), x, 1),
               vstore1_neon(y[2].toPtr(NEON.t.t), x, 2)),
           (y,x,opts) -> chain(
               vstore1_neon(y[1].toPtr(NEON.t.t), x, 1),
               vstore1_neon(y[2].toPtr(NEON.t.t), x, 2),
               vstore1_neon(y[3].toPtr(NEON.t.t), x, 3)),
           (y,x,opts) -> chain(
               vstore1_neon(y[1].toPtr(NEON.t.t), x, 1),
               vstore1_neon(y[2].toPtr(NEON.t.t), x, 2),
               vstore1_neon(y[3].toPtr(NEON.t.t), x, 3),
               vstore1_neon(y[4].toPtr(NEON.t.t), x, 4))
        ],
        # store using subvectors of length 2
        [  (y,x,opts) -> vstore2lo_neon(y[1].toPtr(NEON.t.t), x),
           (y,x,opts) -> chain(vstore2lo_neon(y[1].toPtr(NEON.t.t), x),
                               vstore2hi_neon(y[2].toPtr(NEON.t.t), x))
        ]
    ],

    mul_cx := (self, opts) >> ((y,x,c) -> let(
	    u := self.freshU(),  v1 := self.freshT(), v2 := self.freshT(), r1 := self.freshT(), 
	    decl([u, v1, v2, r1], chain(
		    assign(r1, self.t.value([-1.0,1.0,-1.0,1.0])),
         	    assign(u, vtrnq_32f(c, c)),
		    assign(v1, x * vextract_neon_4x32f(u, [0])),
		    assign(v2, vswapcx_neon(x) * vextract_neon_4x32f(u, [1])),
		    assign(y, fma(v1, v2, r1))))
    )),

    mul_cx_conj := (self, opts) >> ((y,x,c) -> let(
	    u := self.freshU(),  v1 := self.freshT(), v2 := self.freshT(), r1 := self.freshT(), 
	    decl([u, v1, v2, r1], chain(
		    assign(r1, self.t.value([1.0, -1.0, 1.0, -1.0])),
         	    assign(u, vtrnq_32f(c, c)),
		    assign(v1, x * vextract_neon_4x32f(u, [0])),
		    assign(v2, vswapcx_neon(x) * vextract_neon_4x32f(u, [1])),
		    assign(y, fma(v1, v2, r1))))
    )),

    swap_cx := (y, x, opts) -> assign(y, vswapcx_neon(x)),
));

SIMD_ISA_DB.addISA(NEON);
SIMD_ISA_DB.addISA(NEON_HALF);

