
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
# AVX conversion bridges
#


ISA_Bridge.add(Class(CVT_SSE_4x32f_AVX_8x32f, ISA_Bridge_I, rec(
    isa_from    := AVX_8x32f,
    isa_to      := SSE_4x32f(T_Real(32)),
    props       := [],
    code := (self, y, x, opts) >> let( t := var.fresh_t("c", self.isa_from.t), 
        decl( [t], chain(
            assign(t, self._x(x,0)),
            assign(self._y(y,0), vextract_4l_8x32f(t, [0])), 
            assign(self._y(y,1), vextract_4l_8x32f(t, [1]))
        )))
)));

ISA_Bridge.add(Class(CVT_AVX_8x32f_AVX_4x64f, ISA_Bridge_I, rec(
    isa_from    := AVX_4x64f,
    isa_to      := AVX_8x32f,
    props       := [],
    code := (self, y, x, opts) >>
        assign(self._y(y,0), vpermf128_8x32f(vcvt_8x32f_4x64f(self._x(x,0)), vcvt_8x32f_4x64f(self._x(x,1)), [1,3]))
)));

ISA_Bridge.add(Class(CVT_AVX_4x64f_SSE_4x32f, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Real(32)),
    isa_to      := AVX_4x64f,
    props       := [],
    code := (self, y, x, opts) >>
        assign(self._y(y,0), vcvt_4x64f_4x32f(self._x(x,0)))
)));

ISA_Bridge.add(Class(CVT_AVX_8x32f_SSE_4x32i, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Int(32)),
    isa_to      := AVX_8x32f,
    props       := [],
    code := (self, y, x, opts) >>
        assign(self._y(y,0), vcvt_8x32f_8x32i(vinsert_4l_8x32f(vinsert_4l_8x32f(self.isa_to.t.zero(), self._x(x,0), [0]), self._x(x,1), [1])))
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_AVX_8x32f_round, ISA_Bridge_I, rec(
    isa_from    := AVX_8x32f,
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := ["round"],
    code := (self, y, x, opts) >> let( t := var.fresh_t("c", TVect(T_Int(32), 8)),
        decl([t], chain(
            assign(t, vcvt_8x32i_8x32f(self._x(x,0))),
            assign(self._y(y,0), vextract_4l_8x32f(t, [0])), 
            assign(self._y(y,1), vextract_4l_8x32f(t, [1]))
        )))
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_AVX_8x32f_trunc, ISA_Bridge_I, rec(
    isa_from    := AVX_8x32f,
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := ["trunc"],
    code := (self, y, x, opts) >> let( t := var.fresh_t("c", TVect(T_Int(32), 8)),
        decl([t], chain(
            assign(t, vcvtt_8x32i_8x32f(self._x(x,0))),
            assign(self._y(y,0), vextract_4l_8x32f(t, [0])), 
            assign(self._y(y,1), vextract_4l_8x32f(t, [1]))
        )))
)));

ISA_Bridge.add(Class(CVT_AVX_4x64f_SSE_4x32i, ISA_Bridge_I, rec(
    isa_from    := SSE_4x32f(T_Int(32)),
    isa_to      := AVX_4x64f,
    props       := [],
    code := (self, y, x, opts) >>
        assign(self._y(y,0), vcvt_4x64f_4x32i(self._x(x,0)))
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_AVX_4x64f_round, ISA_Bridge_I, rec(
    isa_from    := AVX_4x64f,
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := ["round"],
    code := (self, y, x, opts) >>
        assign(self._y(y,0), vcvt_4x32i_4x64f(self._x(x,0))),
)));

ISA_Bridge.add(Class(CVT_SSE_4x32i_AVX_4x64f_trunc, ISA_Bridge_I, rec(
    isa_from    := AVX_4x64f,
    isa_to      := SSE_4x32f(T_Int(32)),
    props       := ["trunc"],
    code := (self, y, x, opts) >> 
        assign(self._y(y,0), vcvtt_4x32i_4x64f(self._x(x,0))),
)));

ISA_Bridge.add(Class(CVT_AVX_8x32f_8x32f_clip32i, ISA_Bridge_I, rec(
    isa_from    := AVX_8x32f,
    isa_to      := AVX_8x32f,
    props       := ["saturation"],
    range       := (self) >> RangeT(self.clip.min, self.clip.max, T_Real(32).range().eps),
    clip        := rec( min := T_Int(32).range().min, max := T_Int(32).range().max ),
    code := (self, y, x, opts) >> assign( self._y(y,0), min(max(self._x(x,0), self.isa_from.t.value(self.clip.min)), self.isa_from.t.value(self.clip.max)) ),
)));

ISA_Bridge.add(Class(CVT_AVX_8x32f_8x32f_clip32ui, CVT_AVX_8x32f_8x32f_clip32i, rec(
    clip        := rec( min := T_UInt(32).range().min, max := T_UInt(32).range().max ),
)));

ISA_Bridge.add(Class(CVT_AVX_8x32f_8x32f_clip16i, CVT_AVX_8x32f_8x32f_clip32i, rec(
    clip        := rec( min := T_Int(16).range().min,  max := T_Int(16).range().max ),
)));

ISA_Bridge.add(Class(CVT_AVX_8x32f_8x32f_clip16ui, CVT_AVX_8x32f_8x32f_clip32i, rec(
    clip        := rec( min := T_UInt(16).range().min, max := T_UInt(16).range().max ),
)));

ISA_Bridge.add(Class(CVT_AVX_8x32f_8x32f_clip8i, CVT_AVX_8x32f_8x32f_clip32i, rec(
    clip        := rec( min := T_Int(8).range().min,  max := T_Int(8).range().max ),
)));

ISA_Bridge.add(Class(CVT_AVX_8x32f_8x32f_clip8ui, CVT_AVX_8x32f_8x32f_clip32i, rec(
    clip        := rec( min := T_UInt(8).range().min, max := T_UInt(8).range().max ),
)));

ISA_Bridge.add(Class(CVT_AVX_4x64f_4x64f_clip32i, ISA_Bridge_I, rec(
    isa_from    := AVX_4x64f,
    isa_to      := AVX_4x64f,
    props       := ["saturation"],
    range       := (self) >> RangeT(self.clip.min, self.clip.max, T_Real(32).range().eps),
    clip        := rec( min := T_Int(32).range().min, max := T_Int(32).range().max ),
    code := (self, y, x, opts) >> assign( self._y(y,0), min(max(self._x(x,0), self.isa_from.t.value(self.clip.min)), self.isa_from.t.value(self.clip.max)) ),
)));

ISA_Bridge.add(Class(CVT_AVX_4x64f_4x64f_clip32ui, CVT_AVX_4x64f_4x64f_clip32i, rec(
    clip        := rec( min := T_UInt(32).range().min, max := T_UInt(32).range().max ),
)));

