
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(SKLR_Vx1i, SIMD_ISA, rec(

    includes     := () -> ["<stdlib.h>"] :: _MM_MALLOC(), 
    active       := true,
    ctype        := "char",
    instr        := [],
    bits         := 1,
    isFloat      := false,
    isFixedPoint := false,
    splopts      := rec(),
    alignment    := 8,
    
    autolib := rec(
        includes := () -> ["<include/sp_bits.h>"],
        includesTimer := () -> [],
    ),

    _op_load_u  := abstract(),
    _op_bcast   := abstract(),
    _op_store_u := abstract(),
    
    dupload  := (self, y, x) >> Checked(ObjId(x)=nth, # NOTE: is there a better way?
	let(base := x.loc,
	    ofs  := x.idx,
	    v    := self.v,
	    xvec := Cond(IsUnalignedPtrT(base.t), self._op_load_u(base, idiv(ofs,v)*v, v),
		         vtref(self.t, base, idiv(ofs, v))),
	    assign(y, self._op_bcast(xvec, imod(ofs, v))))),
	        
    svload := [ [ ], # load using subvecs of len 1
                [ ], # load using subvecs of len 2
                [ ], # load using subvecs of len 4
    ],

    svstore := [ [ ], # store using subvecs of len 1
                 [ ], # store using subvecs of len 2
                 [ ], # store using subvecs of len 4
    ],

    # keep the n lower scalars and zero the other ones
    mask_l := (self, c, n) >> Cond( n = self.v, c,
        bin_and(c, self.val(Replicate(n, 1) :: Replicate(self.v - n, 0)))),
    mask_h := (self, c, n) >> Cond( n = self.v, c,
        bin_and(c, self.val(Replicate(n, 0) :: Replicate(self.v - n, 1)))),


    loadCont := (self, n, y, yofs, x, xofs, xofs_align, opts) >> let(
	a  := _unwrap(xofs_align),
	nn := _unwrap(n), 
	yy := vtref(self.t, y, yofs),
	m  := x -> self.mask_l(x, nn),
	Cond(a = 0 and not IsUnalignedPtrT(x.t), 
	         assign(yy, m(vtref(self.t, x, xofs/self.v))),

	     # known alignment, sv is small, so that we only need 1 aligned load + 1 shift + mask
	     ((IsInt(a) and (nn <= self.v - a)) or nn=1) and not IsUnalignedPtrT(x.t), 
		 let(v1 := vtref(self.t, x, idiv(xofs, self.v)), 
		     assign(yy, m(bin_shr(v1, a)))),

	     # known alignment, sv covers 2 vectors, 2 aligned loads + 2 shifts + mask
	     # NB: no masking is needed because shifts will do the job
	     IsInt(a) and not IsUnalignedPtrT(x.t),
		 let(v1 := vtref(self.t, x, (xofs - a)/self.v), 
		     v2 := vtref(self.t, x, (xofs - a)/self.v + 1),
		     assign(yy, m(bin_or(bin_shr(v1, a), bin_shl(v2, self.v - a))))),
             # else, unknown alignment, use unaligned load
             assign(yy, self._op_load_u(x, xofs, nn))
        )
    ),

    storeCont := (self, n, y, yofs, yofs_align, x, xofs, opts) >> let(
	a  := _unwrap(yofs_align),
	nn := _unwrap(n), 
	xx := vtref(self.t, x, xofs),
	yy := vtref(self.t, y, yofs/self.v),
	Cond(nn = self.v and a = 0 and not IsUnalignedPtrT(y.t), 
	        assign(yy, xx),
	     #else 
	        self._op_store_u(y, yofs, xx, nn))),

    rotate_left := (self, shift) >> ((y, x) -> assign(vtref(self.t, y, 0), rCyclicShift(vtref(self.t, x, 0), shift, self.v))),
    
    kswap := (self, y, x, k, mask) >> let( u := var.fresh_t("U", self.t),
                                         chain(assign( u, bin_and(bin_xor(x, bin_shr(x, 2^(k-1))), self.t.value(mask))),
                                               assign( y, bin_xor(bin_xor(x, u), bin_shl(u, 2^(k-1)))))),

    kexch := (self, y1, y2, x1, x2, mask) >> let( u := var.fresh_t("U", self.t),
                                         chain(assign(  u, bin_and(bin_xor(x1, x2), self.t.value(mask))),
                                               assign( y1, bin_xor(x1, u)),
                                               assign( y2, bin_xor(x2, u)))),
));

Class(SKLR_16x1i, SKLR_Vx1i, rec(
    # countrec below is invalid
    countrec := rec( 
        ops := [
            [add, sub], 
	    [mul],
            [sklr_bcast_16x1i], # shuffles 
            [sklr_loadu_16x1i, sklr_storeu_16x1i],
            [deref],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        type := "TVect",
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),

    info     := "Scalar 16 x 1-bit",
    v        := 16,
    t        := BitVector(16),

    v_ones   := BitVector(16).one(), 
    v_zeros  := BitVector(16).zero(),
    val      := bits -> BitVector(16).value(bits),

    _op_load_u  := (self, ptr, offs, elts)      >> sklr_loadu_16x1i(ptr, offs, elts),
    _op_bcast   := (self, loc, elt_num)         >> sklr_bcast_16x1i(loc, elt_num),
    _op_store_u := (self, ptr, offs, src, elts) >> sklr_storeu_16x1i(ptr, offs, src, elts),
));

Class(SKLR_32x1i, SKLR_Vx1i, rec(
    # countrec below is invalid
    countrec := rec( 
        ops := [
            [add, sub], 
	    [mul],
            [sklr_bcast_32x1i], # shuffles 
            [sklr_loadu_32x1i, sklr_storeu_32x1i],
            [deref],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        type := "TVect",
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),

    info     := "Scalar 32 x 1-bit",
    v        := 32,
    t        := BitVector(32),

    v_ones   := BitVector(32).one(), 
    v_zeros  := BitVector(32).zero(),
    val      := bits -> BitVector(32).value(bits),

    _op_load_u  := (self, ptr, offs, elts)      >> sklr_loadu_32x1i(ptr, offs, elts),
    _op_bcast   := (self, loc, elt_num)         >> sklr_bcast_32x1i(loc, elt_num),
    _op_store_u := (self, ptr, offs, src, elts) >> sklr_storeu_32x1i(ptr, offs, src, elts),
    
));


Class(SKLR_64x1i, SKLR_Vx1i, rec(
    # countrec below is invalid
    countrec := rec( 
        ops := [
            [add, sub], 
	    [mul],
            [sklr_bcast_64x1i], # shuffles 
            [sklr_loadu_64x1i, sklr_storeu_64x1i],
            [deref],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        type := "TVect",
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),

    info     := "Scalar 64 x 1-bit",
    v        := 64,
    t        := BitVector(64),

    v_ones   := BitVector(64).one(), 
    v_zeros  := BitVector(64).zero(),
    val      := bits -> BitVector(64).value(bits),

    _op_load_u  := (self, ptr, offs, elts)      >> sklr_loadu_64x1i(ptr, offs, elts),
    _op_bcast   := (self, loc, elt_num)         >> sklr_bcast_64x1i(loc, elt_num),
    _op_store_u := (self, ptr, offs, src, elts) >> sklr_storeu_64x1i(ptr, offs, src, elts),
));



RewriteRules(RulesStrengthReduce, rec(
    aligned_loadu_32x1 := Rule( @(1, sklr_loadu_32x1i, x -> IsInt(_unwrap(x.args[2] mod 32)) and not IsUnalignedPtrT(x.args[1])),
                             e -> let(
                                     offs := imod(e.args[2], 32),
                                     xx0  := vtref(e.t, e.args[1], idiv(e.args[2], 32)),
                                     xx1  := vtref(e.t, e.args[1], idiv(e.args[2], 32)+1),
                                     nn   := _unwrap(e.args[3]),
                                     bin_and( When(offs + nn <= 32,
                                             bin_shr(xx0, offs),
                                             bin_or(bin_shr(xx0, offs), bin_shl(xx1, 32 - offs))),
                                         e.t.value(Replicate(nn, 1) :: Replicate(32 - nn, 0)))
                                     )),
    aligned_loadu_64x1 := Rule( @(1, sklr_loadu_64x1i, x -> IsInt(_unwrap(x.args[2] mod 64)) and not IsUnalignedPtrT(x.args[1])),
                             e -> let(
                                     offs := imod(e.args[2], 64),
                                     xx0  := vtref(e.t, e.args[1], idiv(e.args[2], 64)),
                                     xx1  := vtref(e.t, e.args[1], idiv(e.args[2], 64)+1),
                                     nn   := _unwrap(e.args[3]),
                                     bin_and( When(offs + nn <= 64,
                                             bin_shr(xx0, offs),
                                             bin_or(bin_shr(xx0, offs), bin_shl(xx1, 64 - offs))),
                                         e.t.value(Replicate(nn, 1) :: Replicate(64 - nn, 0)))
                                     )),

));

Class(SKLR_32x1i_to_SSE_16x8i, ISA_Bridge, rec(
    isa_from    := SKLR_32x1i,
    isa_to      := SSE_16x8i(T_Int(8)),

    code        := (self, y, x, opts) >> let(
                    xx := (offs) -> vtref(self.isa_from.t, x, offs),
                    yy := (offs) -> vtref(self.isa_to.t,   y, offs),
                    a  := var.fresh_t("U", T_UInt(32)),
                    b0 := var.fresh_t("U", T_UInt(32)),
                    b1 := var.fresh_t("U", T_UInt(32)),
                    b2 := var.fresh_t("U", T_UInt(32)),
                    b3 := var.fresh_t("U", T_UInt(32)),
                    mask := T_UInt(32).value(1 + 256 + 65536 + 16777216),
                    decl([a,b0,b1,b2,b3], chain(
                        assign( a, tcast(a.t, xx(0)) ),
                        assign( b0, bin_and(            a, mask)),
                        assign( b1, bin_and(bin_shr(a, 1), mask)),
                        assign( b2, bin_and(bin_shr(a, 2), mask)),
                        assign( b3, bin_and(bin_shr(a, 3), mask)),
                        assign( yy(0), tcast(self.isa_to.t, vpack(b0, b1, b2, b3))),
                        assign( b0, bin_and(bin_shr(a, 4), mask)),
                        assign( b1, bin_and(bin_shr(a, 5), mask)),
                        assign( b2, bin_and(bin_shr(a, 6), mask)),
                        assign( b3, bin_and(bin_shr(a, 7), mask)),
                        assign( yy(1), tcast(self.isa_to.t, vpack(b0, b1, b2, b3)))
                    ))),

    toAMat := self >> L(self.isa_from.v, 8).toAMat(),
    toSpl  := self >> Cvt(self)*TL(self.isa_from.v, div(self.isa_from.v, 8)).withTags([AVecReg(self.isa_from)])
));

Class(SKLR_64x1i_to_SSE_16x8i, SKLR_32x1i_to_SSE_16x8i, rec(
    isa_from := SKLR_64x1i,
    isa_to   := SSE_16x8i(T_Int(8)),

    code     := (self, y, x, opts) >> let(
                    xx := (offs) -> vtref(self.isa_from.t, x, offs),
                    yy := (offs) -> vtref(self.isa_to.t,   y, offs),
                    a  := var.fresh_t("U", T_UInt(64)),
                    b0 := var.fresh_t("U", T_UInt(64)),
                    b1 := var.fresh_t("U", T_UInt(64)),
                    mask := T_UInt(64).value(1 + 2^8 + 2^16 + 2^24 + 2^32 + 2^40 + 2^48 + 2^56),
                    decl([a,b0,b1], chain(
                        assign( a, tcast(a.t, xx(0)) ),
                        assign( b0, bin_and(            a, mask)),
                        assign( b1, bin_and(bin_shr(a, 1), mask)),
                        assign( yy(0), tcast(self.isa_to.t, vpack(b0, b1))),
                        assign( b0, bin_and(bin_shr(a, 2), mask)),
                        assign( b1, bin_and(bin_shr(a, 3), mask)),
                        assign( yy(1), tcast(self.isa_to.t, vpack(b0, b1))),
                        assign( b0, bin_and(bin_shr(a, 4), mask)),
                        assign( b1, bin_and(bin_shr(a, 5), mask)),
                        assign( yy(2), tcast(self.isa_to.t, vpack(b0, b1))),
                        assign( b0, bin_and(bin_shr(a, 6), mask)),
                        assign( b1, bin_and(bin_shr(a, 7), mask)),
                        assign( yy(3), tcast(self.isa_to.t, vpack(b0, b1)))
                    )))
));

Class(SKLR_32x1i_to_SSE_4x32f_f32, SKLR_32x1i_to_SSE_16x8i, rec(
    isa_from := SKLR_32x1i,
    isa_to   := SSE_4x32f(T_Real(32)),

    code     := (self, y, x, opts) >> let(
                    xx := (offs) -> vtref(self.isa_from.t, x, offs),
                    yy := (offs) -> vtref(self.isa_to.t,   y, offs),
                    ti := TVect(T_Int(32), 4),
                    tf := TVect(T_Real(32), 4),
                    a  := var.fresh_t("U", T_UInt(32)),
                    b  := var.fresh_t("U", ti),
                    decl( [a, b], chain( 
                        assign( a, tcast(a.t, xx(0)) ),
                        assign( b, vpack(a, bin_shr(a, 8), bin_shr(a, 16), bin_shr(a, 24)) ),
                        assign(yy(0), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b, 31), 31) ))),
                        assign(yy(1), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b, 30), 31) ))),
                        assign(yy(2), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b, 29), 31) ))),
                        assign(yy(3), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b, 28), 31) ))),
                        assign(yy(4), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b, 27), 31) ))),
                        assign(yy(5), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b, 26), 31) ))),
                        assign(yy(6), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b, 25), 31) ))),
                        assign(yy(7), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b, 24), 31) )))
                    ))
                )
));

Class(SKLR_64x1i_to_SSE_4x32f_f32, SKLR_32x1i_to_SSE_16x8i, rec(
    isa_from := SKLR_64x1i,
    isa_to   := SSE_4x32f(T_Real(32)),

    code     := (self, y, x, opts) >> let(
                    xx := (offs) -> vtref(self.isa_from.t, x, offs),
                    yy := (offs) -> vtref(self.isa_to.t,   y, offs),
                    ti := TVect(T_Int(32), 4),
                    tf := TVect(T_Real(32), 4),
                    a  := var.fresh_t("U", T_UInt(64)),
                    b0 := var.fresh_t("U", ti),
                    b1 := var.fresh_t("U", ti),
                    shift := (t, n) -> tcast(T_Int(32), bin_shr(t, n)),
                    decl( [a, b0, b1], chain( 
                        assign( a, tcast(a.t, xx(0)) ),
                        assign( b0, vpack(shift(a,  0), shift(a,  8), shift(a, 16), shift(a, 24)) ),
                        assign( b1, vpack(shift(a, 32), shift(a, 40), shift(a, 48), shift(a, 56)) ),
                        assign(yy( 0), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b0, 31), 31) ))),
                        assign(yy( 1), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b1, 31), 31) ))),
                        assign(yy( 2), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b0, 30), 31) ))),
                        assign(yy( 3), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b1, 30), 31) ))),
                        assign(yy( 4), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b0, 29), 31) ))),
                        assign(yy( 5), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b1, 29), 31) ))),
                        assign(yy( 6), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b0, 28), 31) ))),
                        assign(yy( 7), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b1, 28), 31) ))),
                        assign(yy( 8), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b0, 27), 31) ))),
                        assign(yy( 9), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b1, 27), 31) ))),
                        assign(yy(10), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b0, 26), 31) ))),
                        assign(yy(11), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b1, 26), 31) ))),
                        assign(yy(12), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b0, 25), 31) ))),
                        assign(yy(13), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b1, 25), 31) ))),
                        assign(yy(14), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b0, 24), 31) ))),
                        assign(yy(15), tcast(tf, bin_and( tf.one(), arith_shr(bin_shl(b1, 24), 31) )))
                    ))
                )
));

# NOTE: assumption that most significant bit is set for non zero numbers:
#
# SSE_16x8i_i8_to_SKLR_32x1i can be implemented as 
#    neg(vmovemask_16x8i(eq(self.isa_from.t.zero(), xx(2*i))))
# and later simplified if xx(2*i) comes from comparision, yet we cannot match this situation
# because it's unlikely that we will have comparision propagated into this expression.
# Another way is to make T_Bool and simplify expression above by looking at data type.

ISA_Bridge.add(Class(CVT_SKLR_16x1i_SSE_16x8i, ISA_Bridge_I, rec(
    isa_from    := SSE_16x8i(T_Int(8)),
    isa_to      := SKLR_16x1i,
    props       := ["saturation"],
    code := (self, y, x, opts) >> assign(self._y(y,0), vmovemask_16x8i(self._x(x,0))),
)));

ISA_Bridge.add(Class(CVT_SKLR_16x1i_SSE_16x8ui, CVT_SKLR_16x1i_SSE_16x8i, rec(
    isa_from := SSE_16x8i(T_UInt(8)),
)));

Class(SKLR_32f_to_SKLR_32x1i, ISA_Bridge_I, rec(
    isa_from    := SKLR(T_Real(32)),
    isa_to      := SKLR_32x1i,
    granularity := self >> self.isa_to.v,

    code := (self, y, x, opts) >> let(
                j  := Ind(self.isa_to.v),
                xt := self.isa_from.t,
                yt := T_UInt(self.isa_to.v),
                yy := (offs) -> vtref(self.isa_to.t, y, offs),
                a  := var.fresh_t("U", yt),
                decl([a], chain(
                    assign(a, a.t.zero()),
                    loop(j, j.range, 
                        assign(a, bin_or(a, cond(eq(nth(x, j), xt.zero()), yt.zero(), bin_shl(yt.one(), j))))),
                    assign(yy(0), a)
                ))
            ),
));

Class(SKLR_32f_to_SKLR_64x1i, SKLR_32f_to_SKLR_32x1i, rec(
    isa_from := SKLR(T_Real(32)),
    isa_to   := SKLR_64x1i
));


Class(SKLR_64x1i_to_SKLR_32f, ISA_Bridge_I, rec(
    isa_from    := SKLR_64x1i,
    isa_to      := SKLR(T_Real(32)),
    granularity := self >> self.isa_from.v,

    code := (self, y, x, opts) >> let(
                j  := Ind(self.isa_from.v),
                xx := (offs) -> vtref(self.isa_from.t, x, offs),
                ti := T_UInt(self.isa_from.v),
                tf := self.isa_to.t,
                a  := var.fresh_t("U", ti),
                decl( [a], chain( 
                    assign( a, tcast(a.t, xx(0)) ),
                    loop(j, j.range, 
                        assign( nth(y, j), tcvt( tf, bin_and(bin_shr(a, j), a.t.one())))
                    ).unroll()
                ))
            ),
));


Class(SKLR_32x1i_to_SKLR_32f, SKLR_64x1i_to_SKLR_32f, rec(
    isa_from := SKLR_32x1i,
    isa_to   := SKLR(T_Real(32))
));

Class(SKLR_64x1i_to_SKLR_8i, SKLR_64x1i_to_SKLR_32f, rec(
    isa_from := SKLR_64x1i,
    isa_to   := SKLR(T_Int(8))
));

Class(SKLR_32x1i_to_SKLR_8i, SKLR_64x1i_to_SKLR_32f, rec(
    isa_from := SKLR_32x1i,
    isa_to   := SKLR(T_Int(8))
));




ISA_Bridge.add(Class(CVT_SKLR_32x1i_SKLR_16x1i, ISA_Bridge_I, rec(
    isa_from    := SKLR_16x1i,
    isa_to      := SKLR_32x1i,
    code := (self, y, x, opts) >> 
        assign(self._y(y,0), bin_or(tcvt(T_UInt(32), self._x(x,0)), bin_shl(tcvt(T_UInt(32), self._x(x,1)), 16)) ),
)));

ISA_Bridge.add(Class(CVT_SKLR_64x1i_SKLR_16x1i, ISA_Bridge_I, rec(
    isa_from    := SKLR_16x1i,
    isa_to      := SKLR_64x1i,
    code := (self, y, x, opts) >> 
        assign(self._y(y,0), bin_or(
                     tcvt(T_UInt(64), self._x(x,0)),
             bin_shl(tcvt(T_UInt(64), self._x(x,1)), 16),
             bin_shl(tcvt(T_UInt(64), self._x(x,2)), 32),
             bin_shl(tcvt(T_UInt(64), self._x(x,3)), 48)
        )),
)));






















