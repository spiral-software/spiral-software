
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_SKLR_ISA_TYPES := [
    [ T_Int(8),    "char"      ],
    [ T_Int(16),   "short"     ],
    [ T_Int(32),   "int"       ],
    [ T_Int(64),   "long long" ],
    [ T_UInt(8),   "unsigned char"      ],
    [ T_UInt(16),  "unsigned short"     ],
    [ T_UInt(32),  "unsigned int"       ],
    [ T_UInt(64),  "unsigned long long" ],
    [ T_Real(32),  "float"  ],
    [ T_Real(64),  "double" ],
    [ TInt,        "long"   ],
    [ TUInt,       "unsigned long"],
    [ TReal,       "double" ]
];


Class(SKLR, ISA, rec(
    __call__ := (self, el_t) >> WithBases(self, rec(
        t            := el_t, 
	isSigned     := el_t.isSigned(),
	isFloat      := IsRealT(el_t),
	isFixedPoint := IsFixedPtT(el_t),
	operations   := ISAOps,
	print        := self >> Cond( self.t in List(_SKLR_ISA_TYPES, e -> e[1]), 
	                                Print(self.id()), Print(self.__name__, "(", self.t, ")")),
	id           := self >> self.__name__ :: "_" :: el_t.strId(),
        ctype        := let( t := PositionProperty(_SKLR_ISA_TYPES, e -> e[1]=el_t),
                             Cond(t=false, "", _SKLR_ISA_TYPES[t][2])),
    )),

    isScalarISA  := true,

    v         := 1,
    getTags   := self >> [],
    getTagsCx := self >> [],
    toScalar  := self >> self,
    
    # this must be depricated:
    saturatedArithmetic := false,

));

IsSKLR_ISA := (x) -> IsRec(x) and IsBound(x.isScalarISA) and x.isScalarISA;

# register short variable names for SKLR(t) where t in  _SKLR_ISA_TYPES
# SKLR_i8, SKLR_ui8, ..., SKLR_f32, SKLR_f64, SKLR_f, SKLR_i
DoForAll( _SKLR_ISA_TYPES, t -> Assign(SKLR(t[1]).id(), SKLR(t[1])) );

# ISA.toScalar() returns scalar ISA of appropriate data type

SIMD_ISA.toScalar := self >> SKLR(self.gt());

ISA_Bridge.add(Class(CVT_SKLR_i16_i8,    ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_Int(8)),  isa_to := SKLR(T_Int(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_i16_ui8,   ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(8)), isa_to := SKLR(T_Int(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui16_ui8,  ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(8)), isa_to := SKLR(T_UInt(16)) )));

ISA_Bridge.add(Class(CVT_SKLR_i32_i16,   ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_Int(16)),  isa_to := SKLR(T_Int(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_i32_ui16,  ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(16)), isa_to := SKLR(T_Int(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_ui16, ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(16)), isa_to := SKLR(T_UInt(32)) )));

ISA_Bridge.add(Class(CVT_SKLR_i64_i32,   ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_Int(32)),  isa_to := SKLR(T_Int(64)) )));
ISA_Bridge.add(Class(CVT_SKLR_i64_ui32,  ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(32)), isa_to := SKLR(T_Int(64)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui64_ui32, ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(32)), isa_to := SKLR(T_UInt(64)) )));

ISA_Bridge.add(Class(CVT_SKLR_f32_i32,   ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_Int(32)),  isa_to := SKLR(T_Real(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_f32_ui32,  ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(32)), isa_to := SKLR(T_Real(32)) )));

ISA_Bridge.add(Class(CVT_SKLR_f32_i64,   ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_Int(64)),  isa_to := SKLR(T_Real(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_f32_ui64,  ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(64)), isa_to := SKLR(T_Real(32)) )));

ISA_Bridge.add(Class(CVT_SKLR_f64_i32,   ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_Int(32)),  isa_to := SKLR(T_Real(64)) )));
ISA_Bridge.add(Class(CVT_SKLR_f64_ui32,  ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(32)), isa_to := SKLR(T_Real(64)) )));

ISA_Bridge.add(Class(CVT_SKLR_f64_i64,   ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_Int(64)),  isa_to := SKLR(T_Real(64)) )));
ISA_Bridge.add(Class(CVT_SKLR_f64_ui64,  ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_UInt(64)), isa_to := SKLR(T_Real(64)) )));

ISA_Bridge.add(Class(CVT_SKLR_f64_f32,   ISA_Bridge_tcast, rec( props := [], isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_Real(64)) )));

# helper class: integer to larger unsigned integer with saturation on zero
Class(ISA_Bridge_IxIntToUInt, ISA_Bridge_I, rec(
    props    := ["saturation"],
    code     := (self, y, x, opts) >> assign( self._y(y,0), tcast(self.isa_to.t, max(self.isa_from.t.zero(), self._x(x,0))) )
));

ISA_Bridge.add(Class(CVT_SKLR_ui16_i8_wrap,  ISA_Bridge_wrap,        rec( isa_from := SKLR(T_Int(8)),  isa_to := SKLR(T_UInt(16)))));
ISA_Bridge.add(Class(CVT_SKLR_ui16_i8_sat,   ISA_Bridge_IxIntToUInt, rec( isa_from := SKLR(T_Int(8)),  isa_to := SKLR(T_UInt(16)))));
ISA_Bridge.add(Class(CVT_SKLR_ui32_i16_wrap, ISA_Bridge_wrap,        rec( isa_from := SKLR(T_Int(16)), isa_to := SKLR(T_UInt(32)))));
ISA_Bridge.add(Class(CVT_SKLR_ui32_i16_sat,  ISA_Bridge_IxIntToUInt, rec( isa_from := SKLR(T_Int(16)), isa_to := SKLR(T_UInt(32)))));
ISA_Bridge.add(Class(CVT_SKLR_ui64_i32_wrap, ISA_Bridge_wrap,        rec( isa_from := SKLR(T_Int(32)), isa_to := SKLR(T_UInt(64)))));
ISA_Bridge.add(Class(CVT_SKLR_ui64_i32_sat,  ISA_Bridge_IxIntToUInt, rec( isa_from := SKLR(T_Int(32)), isa_to := SKLR(T_UInt(64)))));


# now from larger to smaller
# helper class: larger data type to smaller with max-min saturation
Class(ISA_Bridge_IxSat, ISA_Bridge_I, rec(
    props    := ["saturation"],
    code     := (self, y, x, opts) >> let(
         r := self.isa_to.t.range(),
         a := self.isa_from.t.value(r.min),
         b := self.isa_from.t.value(r.max),
         assign( self._y(y,0), tcast(self.isa_to.t, max(a, min(b, self._x(x,0))))) )
));

#NOTE: add SReduce for max(Value, a) where Value is leq a.t.range().min
#NOTE: add SReduce for max(Value, max(Value, a))
ISA_Bridge.add(Class(CVT_SKLR_i32_i64_wrap,   ISA_Bridge_wrap,  rec( isa_from := SKLR(T_Int(64)),  isa_to := SKLR(T_Int(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_i32_i64_sat,    ISA_Bridge_IxSat, rec( isa_from := SKLR(T_Int(64)),  isa_to := SKLR(T_Int(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_i64_wrap,  ISA_Bridge_wrap,  rec( isa_from := SKLR(T_Int(64)),  isa_to := SKLR(T_UInt(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_i64_sat,   ISA_Bridge_IxSat, rec( isa_from := SKLR(T_Int(64)),  isa_to := SKLR(T_UInt(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_i32_ui64_wrap,  ISA_Bridge_wrap,  rec( isa_from := SKLR(T_UInt(64)), isa_to := SKLR(T_Int(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_i32_ui64_sat,   ISA_Bridge_IxSat, rec( isa_from := SKLR(T_UInt(64)), isa_to := SKLR(T_Int(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_ui64_wrap, ISA_Bridge_wrap,  rec( isa_from := SKLR(T_UInt(64)), isa_to := SKLR(T_UInt(32)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_ui64_sat,  ISA_Bridge_IxSat, rec( isa_from := SKLR(T_UInt(64)), isa_to := SKLR(T_UInt(32)) )));

ISA_Bridge.add(Class(CVT_SKLR_i16_i32_wrap,   ISA_Bridge_wrap,  rec( isa_from := SKLR(T_Int(32)),  isa_to := SKLR(T_Int(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_i16_i32_sat,    ISA_Bridge_IxSat, rec( isa_from := SKLR(T_Int(32)),  isa_to := SKLR(T_Int(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui16_i32_wrap,  ISA_Bridge_wrap,  rec( isa_from := SKLR(T_Int(32)),  isa_to := SKLR(T_UInt(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui16_i32_sat,   ISA_Bridge_IxSat, rec( isa_from := SKLR(T_Int(32)),  isa_to := SKLR(T_UInt(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_i16_ui32_wrap,  ISA_Bridge_wrap,  rec( isa_from := SKLR(T_UInt(32)), isa_to := SKLR(T_Int(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_i16_ui32_sat,   ISA_Bridge_IxSat, rec( isa_from := SKLR(T_UInt(32)), isa_to := SKLR(T_Int(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui16_ui32_wrap, ISA_Bridge_wrap,  rec( isa_from := SKLR(T_UInt(32)), isa_to := SKLR(T_UInt(16)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui16_ui32_sat,  ISA_Bridge_IxSat, rec( isa_from := SKLR(T_UInt(32)), isa_to := SKLR(T_UInt(16)) )));

ISA_Bridge.add(Class(CVT_SKLR_i8_i16_wrap,    ISA_Bridge_wrap,  rec( isa_from := SKLR(T_Int(16)),  isa_to := SKLR(T_Int(8)) )));
ISA_Bridge.add(Class(CVT_SKLR_i8_i16_sat,     ISA_Bridge_IxSat, rec( isa_from := SKLR(T_Int(16)),  isa_to := SKLR(T_Int(8)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui8_i16_wrap,   ISA_Bridge_wrap,  rec( isa_from := SKLR(T_Int(16)),  isa_to := SKLR(T_UInt(8)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui8_i16_sat,    ISA_Bridge_IxSat, rec( isa_from := SKLR(T_Int(16)),  isa_to := SKLR(T_UInt(8)) )));
ISA_Bridge.add(Class(CVT_SKLR_i8_ui16_wrap,   ISA_Bridge_wrap,  rec( isa_from := SKLR(T_UInt(16)), isa_to := SKLR(T_Int(8)) )));
ISA_Bridge.add(Class(CVT_SKLR_i8_ui16_sat,    ISA_Bridge_IxSat, rec( isa_from := SKLR(T_UInt(16)), isa_to := SKLR(T_Int(8)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui8_ui16_wrap,  ISA_Bridge_wrap,  rec( isa_from := SKLR(T_UInt(16)), isa_to := SKLR(T_UInt(8)) )));
ISA_Bridge.add(Class(CVT_SKLR_ui8_ui16_sat,   ISA_Bridge_IxSat, rec( isa_from := SKLR(T_UInt(16)), isa_to := SKLR(T_UInt(8)) )));


ISA_Bridge.add(Class(CVT_SKLR_f32_f64_trunc,   ISA_Bridge_wrap, rec( props := [], isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_Real(32)) )));

Class(ISA_Bridge_trunc_wrap, ISA_Bridge_I, rec(
    range := (self) >> self.isa_to.t.range(),
    props := ["trunc", "wraparound"],
    code  := (self, y, x, opts) >> assign( self._y(y,0), tcast(self.isa_to.t, self._x(x,0)) ),
));

ISA_Bridge.add(Class(CVT_SKLR_i32_f32_trunc,   ISA_Bridge_trunc_wrap, rec( isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_Int(32))  )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_f32_trunc,  ISA_Bridge_trunc_wrap, rec( isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_UInt(32)) )));

ISA_Bridge.add(Class(CVT_SKLR_i32_f64_trunc,   ISA_Bridge_trunc_wrap, rec( isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_Int(64))  )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_f64_trunc,  ISA_Bridge_trunc_wrap, rec( isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_UInt(64)) )));

ISA_Bridge.add(Class(CVT_SKLR_i64_f32_trunc,   ISA_Bridge_trunc_wrap, rec( isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_Int(32))  )));
ISA_Bridge.add(Class(CVT_SKLR_ui64_f32_trunc,  ISA_Bridge_trunc_wrap, rec( isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_UInt(32)) )));

ISA_Bridge.add(Class(CVT_SKLR_i64_f64_trunc,   ISA_Bridge_trunc_wrap, rec( isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_Int(64))  )));
ISA_Bridge.add(Class(CVT_SKLR_ui64_f64_trunc,  ISA_Bridge_trunc_wrap, rec( isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_UInt(64)) )));

Class(ISA_Bridge_round_wrap, ISA_Bridge_I, rec(
    range := (self) >> self.isa_to.t.range(),
    props := ["round", "wraparound"],
    code  := (self, y, x, opts) >> assign( self._y(y,0), tcast(self.isa_to.t, add(self._x(x,0), self.isa_from.t.value(0.5))) ),
));

ISA_Bridge.add(Class(CVT_SKLR_i32_f32_round,   ISA_Bridge_round_wrap, rec( isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_Int(32))  )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_f32_round,  ISA_Bridge_round_wrap, rec( isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_UInt(32)) )));

ISA_Bridge.add(Class(CVT_SKLR_i32_f64_round,   ISA_Bridge_round_wrap, rec( isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_Int(64))  )));
ISA_Bridge.add(Class(CVT_SKLR_ui32_f64_round,  ISA_Bridge_round_wrap, rec( isa_from := SKLR(T_Real(32)), isa_to := SKLR(T_UInt(64)) )));

ISA_Bridge.add(Class(CVT_SKLR_i64_f32_round,   ISA_Bridge_round_wrap, rec( isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_Int(32))  )));
ISA_Bridge.add(Class(CVT_SKLR_ui64_f32_round,  ISA_Bridge_round_wrap, rec( isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_UInt(32)) )));

ISA_Bridge.add(Class(CVT_SKLR_i64_f64_round,   ISA_Bridge_round_wrap, rec( isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_Int(64))  )));
ISA_Bridge.add(Class(CVT_SKLR_ui64_f64_round,  ISA_Bridge_round_wrap, rec( isa_from := SKLR(T_Real(64)), isa_to := SKLR(T_UInt(64)) )));

ISA_Bridge.add(Class(CVT_SKLR_f32_clip32i, ISA_Bridge_I, rec(
    isa_from    := SKLR(T_Real(32)),
    isa_to      := SKLR(T_Real(32)),
    props       := ["saturation"],
    range       := (self) >> RangeT(self.clip.min, self.clip.max, T_Real(32).range().eps),
    clip        := rec( min := T_Int(32).range().min, max := T_Int(32).range().max ),
    code := (self, y, x, opts) >> assign( self._y(y,0), min(max(self._x(x,0), self.isa_from.t.value(self.clip.min)), self.isa_from.t.value(self.clip.max)) ),
)));

ISA_Bridge.add(Class(CVT_SKLR_f32_clip32ui, CVT_SKLR_f32_clip32i, rec(
    clip        := rec( min := T_UInt(32).range().min, max := T_UInt(32).range().max ),
)));

ISA_Bridge.add(Class(CVT_SKLR_f32_clip16i, CVT_SKLR_f32_clip32i, rec(
    clip        := rec( min := T_Int(16).range().min,  max := T_Int(16).range().max ),
)));

ISA_Bridge.add(Class(CVT_SKLR_f32_clip16ui, CVT_SKLR_f32_clip32i, rec(
    clip        := rec( min := T_UInt(16).range().min, max := T_UInt(16).range().max ),
)));

ISA_Bridge.add(Class(CVT_SKLR_f32_clip8i, CVT_SKLR_f32_clip32i, rec(
    clip        := rec( min := T_Int(8).range().min,  max := T_Int(8).range().max ),
)));

ISA_Bridge.add(Class(CVT_SKLR_f32_clip8ui, CVT_SKLR_f32_clip32i, rec(
    clip        := rec( min := T_UInt(8).range().min, max := T_UInt(8).range().max ),
)));




