
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
# SSE conversion bridges
#

# simple cases - signed<->unsigned 

Class(_cvt_int_uint_saturated, ISA_Bridge_I, rec(
    props := ["saturation"],
    code  := (self, y, x, opts) >> 
        assign( self._y(y,0), tcast(self.isa_to.t, bin_and(self._x(x,0), mask_gt(self._x(x,0), self.isa_from.t.zero()))) )
));

ISA_Bridge.add(Class(CVT_SSE_4x32ui_4x32i_wrap, ISA_Bridge_tcast, rec(
    isa_from := SSE_4x32f(T_Int(32)),   isa_to := SSE_4x32f(T_UInt(32)), 
)));

ISA_Bridge.add(Class(CVT_SSE_4x32ui_4x32i_sat, _cvt_int_uint_saturated, rec(
    isa_from := SSE_4x32f(T_Int(32)),   isa_to := SSE_4x32f(T_UInt(32))
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_4x32ui_wrap, ISA_Bridge_tcast, rec(
    isa_from := SSE_4x32f(T_UInt(32)),  isa_to := SSE_4x32f(T_Int(32)),
)));

ISA_Bridge.add(Class(CVT_SSE_8x16ui_8x16i_wrap, ISA_Bridge_tcast, rec(
    isa_from := SSE_8x16i(T_Int(16)),   isa_to := SSE_8x16i(T_UInt(16)),
)));

ISA_Bridge.add(Class(CVT_SSE_8x16ui_8x16i_sat, _cvt_int_uint_saturated, rec(
    isa_from := SSE_8x16i(T_Int(16)),   isa_to := SSE_8x16i(T_UInt(16))
)));

ISA_Bridge.add(Class(CVT_SSE_8x16i_8x16ui_wrap, ISA_Bridge_tcast, rec(
    isa_from := SSE_8x16i(T_UInt(16)),  isa_to := SSE_8x16i(T_Int(16)),
)));

ISA_Bridge.add(Class(CVT_SSE_16x8ui_16x8i_wrap, ISA_Bridge_tcast, rec(
    isa_from := SSE_16x8i(T_Int(8)),    isa_to := SSE_16x8i(T_UInt(8)),
)));

ISA_Bridge.add(Class(CVT_SSE_16x8ui_16x8i_sat, _cvt_int_uint_saturated, rec(
    isa_from := SSE_16x8i(T_Int(8)),    isa_to := SSE_16x8i(T_UInt(8)),
)));

ISA_Bridge.add(Class(CVT_SSE_16x8i_16x8ui_wrap, ISA_Bridge_tcast, rec(
    isa_from := SSE_16x8i(T_UInt(8)),   isa_to := SSE_16x8i(T_Int(8)),
)));

# more complex cases

ISA_Bridge.add(Class(CVT_SSE_4x32i_4x32f_trunc, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Real(32)),
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := ["trunc"],
    code := (self, y, x, opts) >> assign( self._y(y,0), vcvtt_4x32_f2i(self._x(x,0)) ),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_4x32f_round, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Real(32)),
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := ["round"],
    code := (self, y, x, opts) >> assign( self._y(y,0), vcvt_4x32_f2i(self._x(x,0)) ),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32f_4x32f_clip32i, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Real(32)),
    isa_to      := SSE_4x32f(T_Real(32)),
    props       := ["saturation"],
    range       := (self) >> RangeT(self.clip.min, self.clip.max, T_Real(32).range().eps),
    clip        := rec( min := T_Int(32).range().min, max := T_Int(32).range().max ),
    code := (self, y, x, opts) >> assign( self._y(y,0), min(max(self._x(x,0), self.isa_from.t.value(self.clip.min)), self.isa_from.t.value(self.clip.max)) ),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32f_4x32f_clip32ui, CVT_SSE_4x32f_4x32f_clip32i, rec(
    clip        := rec( min := T_UInt(32).range().min, max := T_UInt(32).range().max ),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32f_4x32f_clip16i, CVT_SSE_4x32f_4x32f_clip32i, rec(
    clip        := rec( min := T_Int(16).range().min,  max := T_Int(16).range().max ),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32f_4x32f_clip16ui, CVT_SSE_4x32f_4x32f_clip32i, rec(
    clip        := rec( min := T_UInt(16).range().min, max := T_UInt(16).range().max ),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32f_4x32f_clip8i, CVT_SSE_4x32f_4x32f_clip32i, rec(
    clip        := rec( min := T_Int(8).range().min,  max := T_Int(8).range().max ),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32f_4x32f_clip8ui, CVT_SSE_4x32f_4x32f_clip32i, rec(
    clip        := rec( min := T_UInt(8).range().min, max := T_UInt(8).range().max ),
)));

ISA_Bridge.add(Class(CVT_SSE_8x16i_4x32i_sat, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Int(32)),
    isa_to      := SSE_8x16i(T_Int(16)),
    props       := ["saturation"],
    code := (self, y, x, opts) >> assign( self._y(y,0), vpacks_4x32i(self._x(x,0), self._x(x,1)) )
)));

# saturated SSE_4x32i to SSE_8x16ui 
ISA_Bridge.add(Class(CVT_SSE_8x16ui_4x32i_sat, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Int(32)),
    isa_to      := SSE_8x16i(T_UInt(16)),
    props       := ["saturation"],
    code := (self, y, x, opts) >> let(
                c1 := self.isa_from.t.value(32768),
                c2 := self.isa_to.t.value(32768),
                u1 := var.fresh_t("U", self.isa_from.t),
                u2 := var.fresh_t("U", self.isa_from.t),
                m1 := var.fresh_t("U", self.isa_from.t),
                m2 := var.fresh_t("U", self.isa_from.t),
                decl([u1, u2, m1, m2], chain(
                    assign( u1, self._x(x,0) ),
                    assign( u2, self._x(x,1) ),
                    assign( m1, sub(bin_and(u1, mask_lt(u1.t.zero(), u1)), c1)),
                    assign( m2, sub(bin_and(u2, mask_lt(u1.t.zero(), u2)), c1)),
                    assign( self._y(y,0), add(tcast(self.isa_to.t, vpacks_4x32i(m1, m2)), c2) )
                ))
            ),
)));

# saturated SSE_4x32ui to SSE_8x16ui 
ISA_Bridge.add(Class(CVT_SSE_8x16ui_4x32ui_sat, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_UInt(32)),
    isa_to      := SSE_8x16i(T_UInt(16)),
    props       := ["saturation"],
    code := (self, y, x, opts) >> let(
                c1 := self.isa_from.t.value(32768),
                c2 := self.isa_to.t.value(32768),
                m1 := var.fresh_t("U", self.isa_from.t),
                m2 := var.fresh_t("U", self.isa_from.t),
                decl([m1, m2], chain(
                    assign( m1, sub(self._x(x,0), c1)),
                    assign( m2, sub(self._x(x,1), c1)),
                    assign( self._y(y,0), add(tcast(self.isa_to.t, vpacks_4x32i(m1, m2)), c2) )
                ))
            ),
)));

# SSE_4x32i to SSE_8x16ui without saturation by throwing out high word. 
ISA_Bridge.add(Class(CVT_SSE_8x16ui_4x32i_wrap, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Int(32)),
    isa_to      := SSE_8x16i(T_UInt(16)),
    props       := ["wraparound"],
    code := (self, y, x, opts) >> let(
                u1 := var.fresh_t("U", self.isa_to.t),
                u2 := var.fresh_t("U", self.isa_to.t),
                m1 := var.fresh_t("U", self.isa_to.t),
                m2 := var.fresh_t("U", self.isa_to.t),
                m3 := var.fresh_t("U", self.isa_to.t),
                m4 := var.fresh_t("U", self.isa_to.t),
                decl([u1, u2, m1, m2, m3, m4], chain(
                    assign( u1, tcast(u1.t, self._x(x,0)) ),
                    assign( u2, tcast(u2.t, self._x(x,1)) ),
                    assign( m1, vunpacklo_8x16i(u1, u2) ),
                    assign( m2, vunpackhi_8x16i(u1, u2) ),
                    assign( m3, vunpacklo_8x16i(m1, m2) ),
                    assign( m4, vunpackhi_8x16i(m1, m2) ),
                    assign( self._y(y,0), vunpacklo_8x16i(m3, m4) )
                ))
            ),
)));

ISA_Bridge.add(Class(CVT_SSE_16x8i_8x16i_sat, ISA_Bridge_I, rec(
    isa_from    := SSE_8x16i(T_Int(16)),
    isa_to      := SSE_16x8i(T_Int(8)),
    props       := ["saturation"],
    code := (self, y, x, opts) >> assign( self._y(y,0), vpacks_8x16i(self._x(x,0), self._x(x,1)) )
)));

ISA_Bridge.add(Class(CVT_SSE_16x8ui_8x16i_sat, ISA_Bridge_I, rec(
    isa_from    := SSE_8x16i(T_Int(16)),
    isa_to      := SSE_16x8i(T_UInt(8)),
    props       := ["saturation"],
    code := (self, y, x, opts) >> assign( self._y(y,0), vpackus_8x16i(self._x(x,0), self._x(x,1)) )
)));


ISA_Bridge.add(Class(CVT_SSE_16x8ui_8x16ui_sat_as, ISA_Bridge_I, rec(
    isa_from    := SSE_8x16i(T_UInt(16)),
    isa_to      := SSE_16x8i(T_UInt(8)),
    props       := ["saturation"],
    code := (self, y, x, opts) >> let(
                t  := TVect(T_Int(16), 8),
                a1 := var.fresh_t("U", t),
                a2 := var.fresh_t("U", t),
                m1 := var.fresh_t("U", self.isa_to.t),
                m2 := var.fresh_t("U", self.isa_to.t),
                decl([a1,a2,m1,m2], chain(
                    assign( a1, arith_shr(tcast(t, self._x(x,0)), 15) ),
                    assign( a2, arith_shr(tcast(t, self._x(x,1)), 15) ),
                    assign( m1, vpacks_8x16i(a1, a2) ),
                    assign( m2, vpackus_8x16i(self._x(x,0), self._x(x,1)) ),
                    assign( self._y(y,0), bin_or(m1,m2) )
                ))
            ),
)));

ISA_Bridge.add(Class(CVT_SSE_16x8ui_8x16ui_sat_gt, ISA_Bridge_I, rec(
    isa_from    := SSE_8x16i(T_UInt(16)),
    isa_to      := SSE_16x8i(T_UInt(8)),
    props       := ["saturation"],
    code := (self, y, x, opts) >> let(
                t  := TVect(T_Int(16), 8),
                a1 := var.fresh_t("U", t),
                a2 := var.fresh_t("U", t),
                m1 := var.fresh_t("U", self.isa_to.t),
                m2 := var.fresh_t("U", self.isa_to.t),
                decl([a1,a2,m1,m2], chain(
                    assign( a1, gt(t.zero(), tcast(t, self._x(x,0))) ),
                    assign( a2, gt(t.zero(), tcast(t, self._x(x,1))) ),
                    assign( m1, vpacks_8x16i(a1, a2) ),
                    assign( m2, vpackus_8x16i(self._x(x,0), self._x(x,1)) ),
                    assign( self._y(y,0), bin_or(m1,m2) )
                ))
            ),
)));

ISA_Bridge.add(Class(CVT_SSE_8x16i_16x8i_gt, ISA_Bridge_I, rec(
    isa_from    := SSE_16x8i(T_Int(8)),
    isa_to      := SSE_8x16i(T_Int(16)),
    props       := [],
    code := (self, y, x, opts) >> let(
                sign := var.fresh_t("U", self.isa_from.t),
                decl( [sign], chain(
                    assign(sign, gt(self.isa_from.t.zero(), self._x(x,0))),
                    assign(self._y(y,0), tcast( self.isa_to.t, vunpacklo_16x8i(self._x(x,0), sign)) ),
                    assign(self._y(y,1), tcast( self.isa_to.t, vunpackhi_16x8i(self._x(x,0), sign)) )
                ))
            ),
)));

ISA_Bridge.add(Class(CVT_SSE_8x16i_16x8i_as, ISA_Bridge_I, rec(
    isa_from    := SSE_16x8i(T_Int(8)),
    isa_to      := SSE_8x16i(T_Int(16)),
    props       := [],
    code := (self, y, x, opts) >> chain(
                    assign(self._y(y,0), arith_shr(tcast( self.isa_to.t, vunpacklo_16x8i(self._x(x,0), self._x(x,0))), 8) ),
                    assign(self._y(y,1), arith_shr(tcast( self.isa_to.t, vunpackhi_16x8i(self._x(x,0), self._x(x,0))), 8) )
                ),
)));

ISA_Bridge.add(Class(CVT_SSE_8x16i_16x8ui, ISA_Bridge_I, rec(
    isa_from    := SSE_16x8i(T_UInt(8)),
    isa_to      := SSE_8x16i(T_Int(16)),
    props       := [],
    code := (self, y, x, opts) >> chain(
                    assign(self._y(y,0), vcastizxlo(self._x(x,0)) ),
                    assign(self._y(y,1), vcastizxhi(self._x(x,0)) )
                )
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_8x16i_gt, ISA_Bridge_I, rec(
    isa_from    := SSE_8x16i(T_Int(16)),
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := [],
    code := (self, y, x, opts) >> let(
                sign := var.fresh_t("U", self.isa_from.t),
                decl( [sign], chain(
                    assign(sign, gt(self.isa_from.t.zero(), self._x(x,0))),
                    assign(self._y(y,0), tcast( self.isa_to.t, vunpacklo_8x16i(self._x(x,0), sign)) ),
                    assign(self._y(y,1), tcast( self.isa_to.t, vunpackhi_8x16i(self._x(x,0), sign)) )
                ))
            ),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_8x16i_as, ISA_Bridge_I, rec(
    isa_from    := SSE_8x16i(T_Int(16)),
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := [],
    code := (self, y, x, opts) >> chain(
                    assign(self._y(y,0), arith_shr(tcast( self.isa_to.t, vunpacklo_8x16i(self._x(x,0), self._x(x,0))), 16) ),
                    assign(self._y(y,1), arith_shr(tcast( self.isa_to.t, vunpackhi_8x16i(self._x(x,0), self._x(x,0))), 16) )
                )
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_8x16ui, ISA_Bridge_I, rec(
    isa_from    := SSE_8x16i(T_UInt(16)),
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := [],
    code := (self, y, x, opts) >> chain(
                    assign(self._y(y,0), tcast( self.isa_to.t, vunpacklo_8x16i(self._x(x,0), self.isa_from.t.zero())) ),
                    assign(self._y(y,1), tcast( self.isa_to.t, vunpackhi_8x16i(self._x(x,0), self.isa_from.t.zero())) )
                )
)));

ISA_Bridge.add(Class(CVT_SSE_4x32f_4x32i, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Int(32)),
    isa_to      := SSE_4x32f(T_Real(32)),
    props       := [],
    code := (self, y, x, opts) >> assign(self._y(y,0), vcvt_4x32_i2f(self._x(x,0)))
)));

