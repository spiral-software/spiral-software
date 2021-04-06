
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(SSEUnparser);

_toReal := v -> Cond(IsValue(v), Value(TReal, v.v), tcast(TReal, v));

@Value      := @.cond(IsValue);
@TInt       := @.cond(x->IsIntT(x.t));
@TReal      := @.cond(x->IsRealT(x.t));
@TRealInt   := @.cond(x->IsIntT(x.t) or IsRealT(x.t));
@_scalar    := @.cond(x->IsIntT(x.t) or IsRealT(x.t) or IsPtrT(x.t));
@TVect      := @.cond(x->IsVecT(x.t));
@TVectUChar := @.cond(x->IsVecT(x.t) and ObjId(x.t.t)=TUChar);

_isa := self -> self.opts.vector.isa;
_epi_or_px := (self, o) -> When(_isa(self).isFixedPoint or IsOrdT(o.t.t),
    "epi" :: self.ctype_suffixval(o.t, _isa(self)),
    self.ctype_suffix(o.t, _isa(self)));

_epu_or_px := (self, o) -> When(_isa(self).isFixedPoint,
    "epu" :: self.ctype_suffixval(o.t, _isa(self)),
    self.ctype_suffix(o.t, _isa(self)));

_epi := (self, o) -> Concat("epi", self.ctype_suffixval(o.t, _isa(self)));

_epu_to_epi := (s) -> Cond( s{[1..3]} = "epu", "epi" :: s{[4..Length(s)]}, s );


Class(SSEUnparser, CMacroUnparserProg, rec(
    # -----------------------------
    # ISA independent constructs
    # -----------------------------
    nth := (self, o, i, is) >> self.printf("$1[$2]", [o.loc, o.idx]),
    fdiv := (self, o, i, is) >> self.printf("(((double)$1) / ($2))", o.args),
    div  := (self, o, i, is) >> self.printf("(($1) / ($2))", o.args),
    idiv := (self, o, i, is) >> self.printf("(($1) / ($2))", o.args),

    # --------------------------------
    # ISA constructs, general
    # -------------------------------

    # This is a general suffix for intrinsics that is determine from the data type
    ctype_suffix := (self, t, isa) >> Cond(
        t = TVect(T_Int(128), 1),  "epi128",
        t = TVect(T_UInt(128), 1), "epu128",

        t = TVect(T_Real(32), 4) or
        t = TVect(T_Real(32), 2) and isa=SSE_2x32f or
        t = TVect(TReal, 4) and isa=SSE_4x32f or
        t = TVect(TReal, 2) and isa=SSE_2x32f,
        "ps",

        # no way to create __m64 type directly from floats, have to go thru integers
        t = TVect(T_Real(32), 2) and isa=SSE_4x32f or
        t = TVect(TReal, 2) and isa=SSE_4x32f,
        "ps_half",

        t = TVect(TInt, 4) or
        t = TVect(T_Int(32), 4) or
        t = TVect(TReal, 4) and isa.isFixedPoint,
        "epi32",

        t = TVect(T_UInt(32), 4), "epu32",

        t = TVect(T_Real(64), 2) or
        t = TVect(TReal, 2) and isa=SSE_2x64f,
        "pd",

        t = TVect(TInt, 2) or
        t = TVect(T_Int(64), 2) or
        t = TVect(TReal, 2) and isa.isFixedPoint,
        "epi64",

        t = TVect(T_UInt(64), 2), "epu64",

        t = TVect(T_Int(16),  8), "epi16",
        t = TVect(T_UInt(16), 8), "epu16",
        t = TVect(TInt,       8), "epi16",
        t = TVect(TReal,      8), "epi16",

        t = TVect(T_Int(8),  16), "epi8",
        t = TVect(T_UInt(8), 16), "epu8",
        t = TVect(TReal,     16), When(isa.isSigned, "epi8", "epu8"),
        t = TVect(TInt,      16), "epi8",
        t = TVect(TUChar,    16), "epu8",
        ""
    ),

    mul_suffix := (t,isa) -> Cond(
        t = TVect(T_Real(32), 2), "_ps",
        t = TVect(T_Real(32), 4), "_ps",
        t = TVect(T_Real(64), 2), "_pd",
        t = TVect(TReal, 2) and isa = SSE_2x32f, "_ps",
        t = TVect(TReal, 2), "_pd",
        t = TVect(TInt, 2), "_epi64",
        t = TVect(TReal, 4), "_ps",
        t = TVect(TInt, 4), "_epi32",
        t = TVect(T_Int(32), 4), "lo_epi32",
        t = TVect(TReal, 8), "lo_epi16",
        t = TVect(TReal, 16), Error("16-way multiplication is not supported"),
        t = TVect(TUChar, 16), Error("16-way multiplication is not supported"),
        ""
    ),

    # This is a general suffix for intrinsics that is determine from the data type
    ctype_suffixval := (t, isa) -> Cond(
	t = TVect(TReal, 2), "64",
	t = TVect(TInt, 4), "32",
	t = TVect(TReal, 4), "32",
	t = TVect(T_Int(32), 4), "32",
    t = TVect(T_UInt(32), 4), "32",
	t = TVect(TReal, 8), "16",
	t = TVect(TInt, 8), "16",
	t = TVect(T_Int(16), 8), "16",
    t = TVect(T_UInt(16), 8), "16",
	t = TVect(TReal, 16), "8",
	t = TVect(TInt, 16), "8",
	t = TVect(T_Int(8), 16), "8",
    t = TVect(T_UInt(8), 16), "8",
	t = TVect(TUChar, 16), "8",
	t = TVect(T_Real(32), 4), "32",
	t = TVect(T_Real(64), 2), "64",
	""
    ),

    # This is the type used for declarations of vector variables
    ctype := (self, t, isa) >> Cond(
        # NOTE: used for unaligned vector pointers,for single prec, it should be "float"
	t in [TReal, TVect(TReal, 1)],
            Cond(isa = SSE_2x64f, "double",
		 isa = SSE_2x64i, "__int64",
		 isa = SSE_4x32f, "float",
		 isa = SSE_2x32f, "float",
		 isa = SSE_4x32i, "__int32",
		 isa = SSE_8x16i, "short",
		 isa = SSE_16x8i, Cond(isa.isSigned, "char", "unsigned char"),
		 isa.ctype),

	t = TVect(TReal, 2),
            Cond(isa = SSE_2x64f, "__m128d",
		 isa = SSE_2x64i, "__m128i",
		 isa = SSE_4x32f, "__m64",
		 isa = SSE_8x16i, "__int32",
		 isa = SSE_16x8i, "__int16",
		 isa = SSE_2x32f, "__m64"),

	t = TVect(TReal, 4),
            Cond(isa = SSE_4x32f, "__m128",
		 isa = SSE_2x32f, "__m128",
		 isa = SSE_4x32i, "__m128i"),

        t = TVect(TInt,    2), "__m128i",
	t = TVect(TInt,    4), "__m128i",
	t = TVect(TInt,    8), "__m128i",
	t = TVect(TReal,   8), "__m128i",
	t = TVect(TInt,   16), "__m128i",
	t = TVect(TUChar, 16), "__m128i",
	t = TVect(TReal,  16), "__m128i",

	t = TInt,
            Cond(isa = SSE_2x64i, "__int64",
		 isa = SSE_4x32i, "__int32",
		 isa = SSE_8x16i, "short",
		 isa = SSE_16x8i, "char",
		 "int"),

	t = TVect(T_Int(128), 1), "__m128i",
	t = TVect(T_Int(64),  2), "__m128i",
	t = TVect(T_Int(32),  4), "__m128i",
	t = TVect(T_Int(16),  8), "__m128i",
	t = TVect(T_Int(8),  16), "__m128i",

	t = TVect(T_UInt(128), 1), "__m128i",
	t = TVect(T_UInt(64),  2), "__m128i",
	t = TVect(T_UInt(32),  4), "__m128i",
	t = TVect(T_UInt(16),  8), "__m128i",
	t = TVect(T_UInt(8),  16), "__m128i",

        t = TVect(T_Real(32), 2), "__m64",
	t = TVect(T_Real(32), 4), "__m128",
	t = TVect(T_Real(64), 2), "__m128d",
	Error(self,".ctype doesn't know type ",t)
    ),

    cvalue_suffix  := (self, t)  >> let( isa := _isa(self), Cond(
        (t = TReal and isa in [SSE_2x32f, SSE_4x32f]) or t = T_Real(32), "f",
        (t = TReal) or t = T_Real(64), "",
        Error(self,".cvalue_suffix doesn't know type ",t)
    )),

    vhex := (self, o, i, is) >> Print("_mm_set_", _epi(self, o), "(", self.infix(Reversed(o.p), ", "), ")"),

    Value := (self, o, i, is) >> let(zero := "0" :: self.cvalue_suffix(TReal), Cond(
        o.t = TString, Print(o.v),

        o.t = TReal or ObjId(o.t)=T_Real, let(v := When(IsCyc(o.v), ReComplex(Complex(o.v)), Double(o.v)),
            When(v<0, Print("(", v, self.cvalue_suffix(o.t), ")"), Print(v, self.cvalue_suffix(o.t)))),

        #IsComplexT(o.t),
	#    Print("COMPLEX(", ReComplex(Complex(o.v)), self.cvalue_suffix(o.t.realType()), ", ",
	#        ImComplex(Complex(o.v)), self.cvalue_suffix(o.t.realType()), ")"),

        IsIntT(o.t) or IsUIntT(o.t),
            When(o.v < 0, Print("(", o.v, ")"), Print(o.v)),

        ObjId(o.t) = TVect and _isa(self) = SSE_2x32f,
            Cond(self.cx.isInside(Value) and Length(self.cx.Value) >= 2, # nested in an array
		 Print(          "{", zero, ", ", zero, ", ", self.infix((o.v), ", "), "}"),
		 Print("_mm_set_ps(", zero, ", ", zero, ", ", self.infix(Reversed(o.v), ", "), ")")),

        ObjId(o.t) = TVect and Length(Set(o.v)) = 1,
            Cond(self.cx.isInside(Value) and Length(self.cx.Value) >= 2, # nested in an array
		 Print("{", self.infix(Replicate(o.t.size, o.v[1]), ", "), "}"),
		 Print("_mm_set1_", _epi_or_px(self, o), "(", self(o.v[1], i, is), ")")),

        ObjId(o.t) = TVect,
            Cond(self.cx.isInside(Value) and Length(self.cx.Value) >= 2, # nested in an array
		 Print(                                 "{", self.infix((o.v), ", "), "}"),
		 Print("_mm_set_", _epi_or_px(self, o), "(", self.infix(Reversed(o.v), ", "), ")")),

        IsArray(o.t),
            Print("{", self.infix(o.v, ", "), "}"),

        ObjId(o.t) = TSym,
            Print("(", self.declare(o.t, [], 0, 0), ") ", o.v),

        o.t = TBool, Print(When(o.v = true, "1", "0")),

	Inherited(o, i, is)
    )),

    vpack := (self, o, i, is) >> let(
        sfx := _epi_or_px(self, o),
        Print("_mm_set_", sfx, "(", self.infix(Reversed(o.args), ", "), ")")),

    vdup := (self, o, i, is) >> let(
	sfx := _epi_or_px(self, o),
        CondPat(o,
            [vdup, nth, @.cond(x->x.t=TInt and x.v=2)], self.printf("_mm_loaddup_$1(&($2))", [sfx, o.args[1]]),
            [vdup, @, @TInt], self.printf("_mm_set1_$1($2)", [sfx, o.args[1]]))),

    # --------------------------------
    # Declarations
    _declTVect := (self, t, vars, i, is) >> let(ctype := self.ctype(t, _isa(self)), Print(ctype, " ", self.infix(vars, ", ", i+is))),
    _unparseTVect := (self, t, i, is) >> let(ctype := self.ctype(t, _isa(self)), Print(ctype)),

    TVect := arg >> When(Length(arg)=5, arg[1]._declTVect(arg[2], arg[3], arg[4], arg[5]),
	                                arg[1]._unparseTVect(arg[2], arg[3], arg[4])),
    TReal := ~.TVect,
    TInt  := (self, t, vars, i, is) >> Print("int ", self.infix(vars, ", ", i+is)),
    TBool := (self, t, vars, i, is) >> Print("int ", self.infix(vars, ", ", i+is)),

    # --------------------------------
    # Arithmetic
    #
    mul := (self, o, i, is) >> let(n := Length(o.args), Cond(
        not IsVecT(o.t),
            Print("(",self.pinfix(o.args, ")*("),")"),
	n > 2 and n mod 2 <> 0,
            self(mul(o.args[1], ApplyFunc(mul, Drop(o.args, 1))), i, is),
        n > 2,
            self(mul(ApplyFunc(mul, o.args{[1..n/2]}), ApplyFunc(mul, o.args{[n/2+1..n]})), i, is),
        CondPat(o,
	    [mul, @TReal, @TVect], Cond(_isa(self) = SSE_2x32f,
                self(mul(vdup(o.args[1], 4), o.args[2]), i, is), # NOTE: HACK for SSE_2x32f
                self(mul(vdup(o.args[1], o.t.size), o.args[2]), i, is)),
	    [mul, @TVect, @TReal],  self(mul(o.args[1], vdup(o.args[2],o.t.size)), i, is),
            # NOTE: This hack is probably no longer necessary (was used for PRDFTs)
	    [mul, @(1, cond, e -> e.t=TInt), @TVect],
	        self(mul(cond(o.args[1].args[1],
			      vdup(o.t.t.value(o.args[1].args[2]), o.t.size),
			      vdup(o.t.t.value(o.args[1].args[3]), o.t.size)), o.args[2]), i, is),
	    [mul, @TInt, @TVect],  self(mul(vdup(_toReal(o.args[1]),o.t.size), o.args[2]), i, is),
	    [mul, @TVect, @TInt],  self(mul(o.args[1], vdup(_toReal(o.args[2]),o.t.size)), i, is),
	    [mul, @TVect, @TVect], self.printf("_mm_mul$1($2, $3)", [self.mul_suffix(o.t, _isa(self)), o.args[1], o.args[2]]),
	    Error("Don't know how to unparse <o>. Unrecognized type combination")
    ))),

    fpmul := (self, o, i, is) >> let(isa := _isa(self), CondPat(o,
        # preparing for SSSE3 _mm_mulhrs_epi16 (__m128i a, __m128i b)
        # self.printf("_mm_mulhrs_epi16($1, $2)", [o.args[2], o.args[3].t.value(List(o.args[3].v, i->bin_shl(i,1)))]),
        [fpmul, @, @TVect, @], self.printf("$1($2($3, $4), $5)",
	    [isa.vlshift, isa.vmul, o.args[2], o.args[3], isa.bits-o.args[1]]),

        [fpmul, @, @, @],      self.printf("$1($2(_mm_set1_$3($4), $5), $6)",
	    [isa.vlshift, isa.vmul, self.ctype_suffix(o.t, _isa(self)), o.args[2], o.args[3], isa.bits-o.args[1]]))),

    add := (self, o, i, is) >> let(n := Length(o.args), Cond(
	not IsVecT(o.t),
            self.pinfix(o.args, " + "),
        n > 2 and n mod 2 <> 0,
            self(add(o.args[1], ApplyFunc(add, Drop(o.args, 1))), i, is),
        n > 2,
            self(add(ApplyFunc(add, o.args{[1..n/2]}), ApplyFunc(add, o.args{[n/2+1..n]})), i, is),
        let(isa := _isa(self), # ugly, backward compatibility, use <adds> instead
	    saturated := When(IsBound(isa.isFloat) and IsBound(isa.saturatedArithmetic) and not isa.isFloat and isa.saturatedArithmetic, "s", ""),
	    _sfx      := self.ctype_suffix(o.t, isa),
	    sfx       := Cond( saturated="", _epu_to_epi(_sfx), _sfx),
	    CondPat(o,
		[add, @TVect,   @TVect], self.printf("_mm_add$1_$2($3, $4)", [saturated, sfx, o.args[1], o.args[2]]),
		Error("Don't know how to unparse <o>. Unrecognized type combination"))))),

    adds := (self, o, i, is) >> CondPat(o,
		[adds, @TVect, @TVect, ...],
		    Cond( Length(o.args)>2,
		        self(adds(o.args[1], brackets(ApplyFunc(adds, Drop(o.args, 1)))), i, is),
		        self.printf("_mm_adds_$1($2, $3)", [self.ctype_suffix(o.t, rec()), o.args[1], o.args[2]])),
		Inherited(o, i, is)),

    _sub := (self, t, a, i, is) >> let(
	isa := _isa(self),
	sfx := _epu_to_epi(self.ctype_suffix(t, isa)),
	saturated := When(IsBound(isa.isFloat) and IsBound(isa.saturatedArithmetic) and not isa.isFloat and isa.saturatedArithmetic, "s", ""),
	CondPat(a,
            [ListClass, @TVect,   @TVect], self.printf("_mm_sub$1_$2($3, $4)", [saturated, sfx, a[1], a[2]]),
            [ListClass, @, @],             self.printf("($1 - ($2))", a),
            Error("Don't know how to unparse subtraction of a[1] and a[2]. Unrecognized type combination"))),

    sub := (self, o, i, is) >> self._sub(o.t, o.args, i, is),

    neg := (self, o, i, is) >> CondPat(o,
        [@, @TVect], self._sub(o.t, [o.t.zero(), o.args[1]], i, is),
        self.printf("(-$1)", o.args)),

    stickyNeg := ~.neg,

    sqrt  := (self, o, i, is) >> Cond( IsVecT(o.t),
        Checked( IsRealT(o.t.t), self.printf("_mm_sqrt_$1($2)", [self.ctype_suffix(o.t, _isa(self)), o.args[1]])),
        Inherited(o, i, is)),

    rsqrt := (self, o, i, is) >> Cond( IsVecT(o.t), let( sfx := self.ctype_suffix(o.t, _isa(self)),
        Checked( sfx="ps", self.printf("_mm_rsqrt_ps($1)", [o.args[1]]))),
        Inherited(o, i, is)),

    # assuming we have ICC <ia32intrin.h> here
    log := (self, o, i, is) >> Cond( IsVecT(o.t), let( sfx := self.ctype_suffix(o.t, _isa(self)),
        Checked( sfx in ["ps", "pd"], Cond(
            Length(o.args)>1 and (o.args[2]=2 or o.args[2]=o.t.value(2)),
                self.printf("_mm_log2_$1($2)", [sfx, o.args[1]]),
            Length(o.args)>1 and (o.args[2]=10 or o.args[2]=o.t.value(10)),
                self.printf("_mm_log10_$1($2)", [sfx, o.args[1]]),
            Length(o.args)=1 or o.args[2]=d_exp(1) or o.args[2]=o.t.value(d_exp(1)),
                self.printf("_mm_log_$1($2)", [sfx, o.args[1]]),
            self.printf("_mm_div_$1(_mm_log_$1($2), _mm_log_$1($3))", [sfx, o.args[1]])))),
        Inherited(o, i, is)),

    # assuming we have ICC <ia32intrin.h> here
    exp := (self, o, i, is) >> Cond( IsVecT(o.t), let( sfx := self.ctype_suffix(o.t, _isa(self)),
        Checked( sfx in ["ps", "pd"], self.printf("_mm_exp_$1($2)", [sfx, o.args[1]]))),
        Inherited(o, i, is)),

    # assuming we have ICC <ia32intrin.h> here
    pow := (self, o, i, is) >> Cond( IsVecT(o.t), let( sfx := self.ctype_suffix(o.t, _isa(self)),
        Checked( sfx in ["ps", "pd"], Cond(
            o.args[1]=2 or o.args[1]=o.t.value(2),
                self.printf("_mm_exp2_$1($2)", [sfx, o.args[2]]),
            o.args[1]=d_exp(1) or o.args[1]=o.t.value(d_exp(1)),
                self.printf("_mm_exp_$1($2)", [sfx, o.args[2]]),
            self.printf("_mm_pow_$1($2, $3)", [sfx, o.args[1], o.args[2]])))),
        Inherited(o, i, is)),

    imod  := (self, o, i, is) >> Cond( IsIntT(o.t.base_t()) and Is2Power(o.args[2]),
        # in two's complement arithmetics this will work for both positive and negative o.args[1]
        self(bin_and(o.args[1], o.args[2]-1), i, is),
        self.printf("(($1) % ($2))", o.args)),

    # --------------------------------
    # logic
    #
    arith_shl := (self, o, i, is) >> self.prefix(_isa(self).vlshift, o.args),
    arith_shr := (self, o, i, is) >> CondPat( o,
        [arith_shr, @.cond(x->x.t=TVect(T_Int(32), 4)), @],
            self.prefix("_mm_srai_epi32", o.args),
        [arith_shr, @.cond(x->x.t=TVect(T_Int(16), 8)), @],
            self.prefix("_mm_srai_epi16", o.args),
        [arith_shr, @TVect, @],
            self.prefix(_isa(self).vrshift, o.args),
        Inherited(o, i, is)),

    bin_xor := (self, o, i, is) >> CondPat(o,
                [bin_xor, @TVect, @TVect], self.prefix("_mm_xor_si128", o.args),
                Inherited(o, i, is)),
    bin_and := (self, o, i, is) >> CondPat(o,
                [bin_and, @TVect, @TVect], self.prefix("_mm_and_si128", o.args),
                Inherited(o, i, is)),

    bin_andnot := (self, o, i, is) >> self.prefix("_mm_andnot_si128", o.args),

    bin_or := (self, o, i, is) >> CondPat(o,
        [bin_or, @TVect, @TVect], let(sfx := self.ctype_suffix(o.t, _isa(self)),
	    Cond( not (sfx in ["ps", "pd", "ps_half"]), #was: _isa(self).isFixedPoint,
		self.printf("_mm_or_si128($1, $2)", o.args),
		self.printf("_mm_castsi128_$3(_mm_or_si128(_mm_cast$3_si128($1), _mm_cast$3_si128($2)))",
		            o.args :: [sfx]))),
        [bin_or, @TReal, @TReal], self.printf("(($1) | ($2))", o.args),

	Inherited(o, i, is)),

    min := (self, o, i, is) >> CondPat(o,
        [min, @TVect, @TVect], self.prefix("_mm_min_" :: self.ctype_suffix(o.t, _isa(self)), o.args),
            Inherited(o, i, is)),

    max := (self, o, i, is) >> let(n := Length(o.args), When(
    	IsVecT(o.t) and n > 2, self.printf("_mm_max_$1($2, $3)", [self.ctype_suffix(o.t, _isa(self)), o.args[1], ApplyFunc(max, Drop(o.args, 1))]),
        CondPat(o,
            [max, @TVect, @TVect], self.prefix("_mm_max_" :: self.ctype_suffix(o.t, _isa(self)), o.args),
                Inherited(o, i, is)))),

    abs := (self, o, i, is) >> CondPat(o,
        [abs, @TVect], let( sfx := self.ctype_suffix(o.t, _isa(self)),
                            Cond( sfx = "ps", self.printf("_mm_castsi128_ps(_mm_and_si128(_mm_castps_si128($1), _mm_set_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF)))", o.args),
                                  sfx = "pd", self.printf("_mm_castsi128_pd(_mm_and_si128(_mm_castpd_si128($1), _mm_set_epi32(0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF)))", o.args),
                                  Error("not implemented"))),
            Inherited(o, i, is)),

    bin_shl := (self, o, i, is) >> CondPat(o,
        [bin_shl, @TVect, @TInt], let(
            sfx := self.ctype_suffix(o.t, _isa(self)),
	    Cond( _isa(self).isFixedPoint,                            # legacy
		     self.printf("_mm_slli_si128($1, $2)", o.args),   # legacy
		  sfx in ["epi16", "epi32", "epi64", "epu16", "epu32", "epu64"],
		      self.printf("_mm_slli_$3($1, $2)", o.args :: [_epu_to_epi(sfx)]),
		  sfx in ["epi8", "epu8"],
		      Error("bin_shl is undefined for epi8 and epu8"),
		  sfx in ["epi128", "epu128"], # shift with byte granularity
		      self.printf("_mm_slli_si128($1, $2)", [o.args[1], idiv(o.args[2], 8)] ),
		  # else, shift whole register with shift argument specified in bytes (legacy, fix using epi128 in ISAs first)
		  self.printf("_mm_castsi128_$3(_mm_slli_si128(_mm_cast$3_si128($1), $2))", o.args :: [sfx]))),
        [bin_shl, @TReal, @TInt], self.printf("(($1) << ($2))", o.args),
        [bin_shl, @TInt, @TInt], self.printf("(($1) << ($2))", o.args),
        [bin_shl, @, @], self.prefix("_mm_slli_" :: self.ctype_suffix(o.t, _isa(self)), o.args)),

    bin_shr := (self, o, i, is) >> CondPat(o,
        [bin_shr, @TVect, @TInt], let(
            sfx := self.ctype_suffix(o.t, _isa(self)),
	    Cond( _isa(self).isFixedPoint,                            # legacy
		      self.printf("_mm_srli_si128($1, $2)", o.args),  # legacy
		  sfx in ["epi16", "epi32", "epi64", "epu16", "epu32", "epu64"],
		      self.printf("_mm_srli_$3($1, $2)", o.args :: [_epu_to_epi(sfx)]),
		  sfx in ["epi8", "epu8"],
		      Error("bin_shr is undefined for epi8 and epu8"),
		  sfx in ["epi128", "epu128"], # shift with byte granularity
		      self.printf("_mm_srli_si128($1, $2)", [o.args[1], idiv(o.args[2], 8)] ),
		  # else, shift whole register with shift argument specified in bytes (legacy, fix using epi128 in ISAs first)
		  self.printf("_mm_castsi128_$3(_mm_srli_si128(_mm_cast$3_si128($1), $2))", o.args :: [sfx]))),
	# default
	[bin_shr, @TReal, @TInt], self.printf("(($1) >> ($2))", o.args),
	[bin_shr, @TInt, @TInt], self.printf("(($1) >> ($2))", o.args),
	# what's this?
	[bin_shr, @, @], self.prefix("_mm_srli_" :: self.ctype_suffix(o.t, _isa(self)), o.args)),

    # vector shifts	
    vec_shr := (self, o, i, is) >> let(
        isa := _isa(self),
        sfx := self.ctype_suffix(o.t, isa),
        # making sure this is SSE data type
        t   := Checked(IsVecT(o.t) and sfx<>"", o.t),
        a   := o.args[1],
        s   := o.args[2] * 16 / t.size,
        # may need typecasts to please compiler
        Cond( self.ctype(t, isa)="__m128i",
            self.printf("_mm_srli_si128($1, $2)", [a, s] ),
            self.printf("_mm_castsi128_$3(_mm_srli_si128(_mm_cast$3_si128($1), $2))", [a, s, sfx])
        )),

    vec_shl := (self, o, i, is) >> let(
        isa := _isa(self),
        sfx := self.ctype_suffix(o.t, isa),
        # making sure this is SSE data type
        t   := Checked(IsVecT(o.t) and sfx<>"", o.t),
        a   := o.args[1],
        s   := o.args[2] * 16 / t.size,
        # may need typecasts to please compiler
        Cond( self.ctype(t, isa)="__m128i",
            self.printf("_mm_slli_si128($1, $2)", [a, s] ),
            self.printf("_mm_castsi128_$3(_mm_slli_si128(_mm_cast$3_si128($1), $2))", [a, s, sfx])
        )),

    # --------------------------------
    # comparison
    #
    eq := (self, o, i, is) >> let( ctype := self.ctype_suffix(o.args[1].t, _isa(self)),
        sfx := _epu_to_epi(ctype),
        Cond(IsVecT(o.t), self.prefix("_mm_cmpeq_" :: sfx, o.args),
            Inherited(o, i, is))),

    lt := (self, o, i, is) >> Cond(IsVecT(o.t),
        self.prefix("_mm_cmplt_" :: self.ctype_suffix(o.args[1].t, _isa(self)), o.args),
        Inherited(o, i, is)),

    gt := (self, o, i, is) >> Cond(ObjId(o.t)=TVect,
        self.prefix("_mm_cmpgt_" :: self.ctype_suffix(o.args[1].t, _isa(self)), o.args),
        Inherited(o, i, is)),

    mask_eq := ~.eq,
    mask_lt := ~.lt,
    mask_gt := ~.gt,

    vparam := (self, o, i, is) >> iclshuffle(o.p),

    # --------------------------------
    # ISA specific : SSE_2x64f
    #
    vunpacklo_2x64f := (self, o, i, is) >> self.prefix("_mm_unpacklo_pd", o.args),
    vunpackhi_2x64f := (self, o, i, is) >> self.prefix("_mm_unpackhi_pd", o.args),
    vshuffle_2x64f  := (self, o, i, is) >> self.prefix("_mm_shuffle_pd", o.args),
    vushuffle_2x64f := (self, o, i, is) >> self(o.binop(o.args[1], o.args[1], o.args[2]), i, is),

    vload1sd_2x64f := (self, o, i, is) >> self.prefix("_mm_load_sd", o.args),
    vload_1l_2x64f := (self, o, i, is) >> self.prefix("_mm_loadl_pd", o.args),
    vload_1h_2x64f := (self, o, i, is) >> self.prefix("_mm_loadh_pd", o.args),
    vloadu_2x64f   := (self, o, i, is) >> self.prefix("_mm_loadu_pd", o.args),

    vstore_1l_2x64f := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storel_pd", o.args), ";\n"),
    vstore_1h_2x64f := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storeh_pd", o.args), ";\n"),
    vstoreu_2x64f   := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storeu_pd", o.args), ";\n"),

    addsub_2x64f := (self, o, i, is) >> Checked(Length(o.args) = 2,
        CondPat(o,
           [addsub_2x64f, @TReal, @TVect], self(addsub_2x64f(vdup(o.args[1],o.t.size), o.args[2]), i, is),
           [addsub_2x64f, @TVect, @TReal], self(addsub_2x64f(o.args[1], vdup(o.args[2], o.t.size)), i, is),
           [addsub_2x64f, @TInt,  @TVect], self(addsub_2x64f(vdup(_toReal(o.args[1]), o.t.size), o.args[2]), i, is),
           [addsub_2x64f, @TVect, @TInt],  self(addsub_2x64f(o.args[1], vdup(_toReal(o.args[2]), o.t.size)), i, is),
           [addsub_2x64f, @TVect, @TVect], self.printf("_mm_addsub_pd($1, $2)", o.args),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
    )),

    hadd_2x64f := (self, o, i, is) >> self.printf("_mm_hadd_pd($1, $2)", [o.args[1], o.args[2]]),

    chslo_2x64f := (self, o, i, is) >> self.printf(
	"_mm_castsi128_pd(_mm_xor_si128(_mm_castpd_si128($1), _mm_set_epi32(0, 0, 0x80000000, 0)))", o.args),
    chshi_2x64f := (self, o, i, is) >> self.printf(
	"_mm_castsi128_pd(_mm_xor_si128(_mm_castpd_si128($1), _mm_set_epi32(0x80000000, 0, 0, 0)))", o.args),
    chshi_4x32f := (self, o, i, is) >> self.printf(
	"_mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128($1), _mm_set_epi32(0x80000000, 0, 0x80000000, 0)))", o.args),
    chslo_4x32f := (self, o, i, is) >> self.printf(
	"_mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128($1), _mm_set_epi32(0, 0x80000000, 0, 0x80000000)))", o.args),

    vcvt_64f32f := (self, o, i, is) >> self.prefix("_mm_cvtps_pd", o.args),

    cmpge_2x64f := (self, o, i, is) >> self.prefix("_mm_cmpge_pd", o.args),

    cmple_2x64f := (self, o, i, is) >> self.prefix("_mm_cmple_pd", o.args),
    cmpeq_2x64f := (self, o, i, is) >> self.prefix("_mm_cmpeq_pd", o.args),


    # --------------------------------
    # ISA specific : SSE_2x32f
    #
    vunpacklo_2x32f := (self, o, i, is) >> self.prefix("_mm_unpacklo_ps", o.args),
    vunpackhi_2x32f := (self, o, i, is) >> self.prefix("_mm_unpackhi_ps", o.args),
    vshuffle_2x32f  := (self, o, i, is) >> self.prefix("_mm_shuffle_ps", o.args),
    vushuffle_2x32f := (self, o, i, is) >> self(o.binop(o.args[1], o.args[1], o.args[2]), i, is),
    vload_2x32f     := (self, o, i, is) >> self.prefix("_mm_loadl_pi", o.args),
    vstore_2x32f    := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storel_pi", o.args), ";\n"),
    vstoreu_2x32f   := (self, o, i, is) >> Print(Blanks(i), self.printf("_mm_storel_epi64($1, _mm_castps_si128($2));\n", o.args)),
    vloadu_2x32f    := (self, o, i, is) >> self.printf("_mm_castsi128_ps(_mm_loadl_epi64($1))", o.args),

    # --------------------------------
    # ISA specific : SSE_4x32f
    #
    prefix_cast := (self, prefix, t, o) >> Cond( self.ctype(t, _isa(self)) = self.ctype(o.t, _isa(self)), self.prefix(prefix, o.args),
                                             self(tcast(o.t, ApplyFunc(ObjId(o), List(o.args, a -> Cond(IsExp(a), tcast(t, a), a)))), 0, 1)),

    vunpacklo_4x32f := (self, o, i, is) >> self.prefix_cast("_mm_unpacklo_ps", TVect(T_Real(32), 4), o),
    vunpackhi_4x32f := (self, o, i, is) >> self.prefix_cast("_mm_unpackhi_ps", TVect(T_Real(32), 4), o),
    vshuffle_4x32f  := (self, o, i, is) >> self.prefix_cast("_mm_shuffle_ps", TVect(T_Real(32), 4), o),
    vushuffle_4x32f := (self, o, i, is) >> self(o.binop(o.args[1], o.args[1], o.args[2]), i, is),
    hadd_4x32f  := (self, o, i, is) >> self.printf("_mm_hadd_ps($1, $2)", [o.args[1], o.args[2]]),
    vldup_4x32f := (self, o, i, is) >> self.prefix("_mm_moveldup_ps", o.args),
    vhdup_4x32f := (self, o, i, is) >> self.prefix("_mm_movehdup_ps", o.args),

    vinsert_4x32f  := (self, o, i, is) >> self.printf(
	"_mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128($1), $2, $3))", [o.args[1], o.args[2], o.args[3].p-1]),
    vextract_4x32f := (self, o, i, is) >> Print(Blanks(i),
	self.printf("$1 = _mm_extract_ps($2, $3)", [deref(o.args[1]), o.args[2], o.args[3]-1]), ";\n"),

    vload1_4x32f   := (self, o, i, is) >> self.prefix("_mm_load_ss", o.args),
    vload_2l_4x32f := (self, o, i, is) >> self.prefix("_mm_loadl_pi", o.args),
    vload_2h_4x32f := (self, o, i, is) >> self.prefix("_mm_loadh_pi", o.args),
    vloadu_4x32f   := (self, o, i, is) >> self.printf("_mm_loadu_ps($1)", o.args),
    vloadu2_4x32f  := (self, o, i, is) >> self.printf("_mm_castsi128_ps(_mm_loadl_epi64($1))", o.args),

    vstore1_4x32f   := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_store_ss",  o.args), ";\n"),
    vstore_2l_4x32f := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storel_pi", o.args), ";\n"),
    vstore_2h_4x32f := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storeh_pi", o.args), ";\n"),
    vstoreu_4x32f   := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storeu_ps", o.args), ";\n"),
    vstoreu2_4x32f  := (self, o, i, is) >> Print(Blanks(i), self.printf("_mm_storel_epi64($1, _mm_castps_si128($2));\n",
        o.args)),

    vstoremsk_4x32f := (self, o, i, is) >> Print(Blanks(i),
	self.printf("_mm_maskmoveu_si128(_mm_castps_si128($2), _mm_set_epi32($3, $4, $5, $6), $1);\n",
            [o.args[1], o.args[2]] :: List(Reversed(o.args[3].v), e->e.v))),

    alignr_4x32f := (self, o, i, is) >> self.printf(
	"_mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128($1), _mm_castps_si128($2), $3))", [o.args[1], o.args[2], o.args[3].p]),

    # --------------------------------
    # ISA specific : SSE_8x16i
    #
    vzero_8x16i := (self, o, i, is) >> Print("_mm_setzero_si128()"),

    vunpacklo_8x16i  := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi16", o.args),
    vunpackhi_8x16i  := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi16", o.args),
    vunpacklo2_8x16i := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi32", o.args),
    vunpackhi2_8x16i := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi32", o.args),
    vunpacklo4_8x16i := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi64", o.args),
    vunpackhi4_8x16i := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi64", o.args),

    vpacks_8x16i     := (self, o, i, is) >> self.prefix("_mm_packs_epi16",    o.args),
    vpackus_8x16i    := (self, o, i, is) >> self.prefix("_mm_packus_epi16",   o.args),

    vshuffle2_8x16i := (self, o, i, is) >> self.printf(
	"_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps($1), _mm_castsi128_ps($2), $3))", o.args),

    vshuffle4_8x16i := (self, o, i, is) >> self.printf(
	"_mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd($1), _mm_castsi128_pd($2), $3))", o.args),

    vload1_8x16i := (self, o, i, is) >> self.printf("_mm_insert_epi16($1, $2, $3)", o.args),
    vload2_8x16i := (self, o, i, is) >> self.prefix("_mm_cvtsi32_si128", o.args),
    vload4_8x16i := (self, o, i, is) >> self.prefix("_mm_loadl_epi64", o.args),
    vloadu_8x16i := (self, o, i, is) >> self.prefix("_mm_loadu_si128", o.args),

    vextract1_8x16i := (self, o, i, is) >> self.prefix("_mm_extract_epi16", o.args),
    vextract2_8x16i := (self, o, i, is) >> self.prefix("_mm_cvtsi128_si32", o.args),
    vstoreu_8x16i   := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storeu_si128", o.args), ";\n"),
    vstore4_8x16i   := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storel_epi64", o.args), ";\n"),
    vstoremsk_8x16i := (self, o, i, is) >> Print(Blanks(i),
	self.printf("_mm_maskmoveu_si128($2, _mm_set_epi16($3), $1);\n",
        [o.args[1], o.args[2], () -> PrintCS(Reversed(o.args[3]))])),

    vushuffle2_8x16i  := (self, o, i, is) >> self(o.binop(o.args[1], o.args[1], o.args[2]), i, is),
    vushufflelo_8x16i := (self, o, i, is) >> self.prefix("_mm_shufflelo_epi16", o.args),
    vushufflehi_8x16i := (self, o, i, is) >> self.prefix("_mm_shufflehi_epi16", o.args),

    interleavedmask_8x16i := (self, o, i, is) >> self.printf(
	"_mm_movemask_epi8(_mm_unpacklo_epi8(_mm_packs_epi16($1, _mm_setzero_si128()), _mm_packs_epi16($2, _mm_setzero_si128())))",
	o.args),

    alignr_8x16i := (self, o, i, is) >> self.printf("_mm_alignr_epi8($1, $2, $3)", [o.args[1], o.args[2], o.args[3].p]),

    # FF: NOTE: couldnt figure out how to use the general case with type propagation etc...
    cmplt_8x16i := (self, o, i, is) >> self.printf("_mm_cmplt_epi16($1, $2)", o.args),

    # SSSE3 8x16i instructions
    chs_8x16i := (self, o, i, is) >> self.prefix("_mm_sign_epi16", o.args),
    vushuffle_8x16i := (self, o, i, is) >> self.prefix("_mm_shuffle_epi8", o.args),

    # --------------------------------
    # ISA specific : SSE_16x8i
    #
    vloadu_16x8i  := (self, o, i, is) >> self.prefix("_mm_loadu_si128", o.args),
    vstoreu_16x8i := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storeu_si128", o.args), ";\n"),

    vunpacklo_16x8i  := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi8", o.args),
    vunpackhi_16x8i  := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi8", o.args),
    vunpacklo2_16x8i := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi16", o.args),
    vunpackhi2_16x8i := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi16", o.args),
    vunpacklo4_16x8i := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi32", o.args),
    vunpackhi4_16x8i := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi32", o.args),
    vunpacklo8_16x8i := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi64", o.args),
    vunpackhi8_16x8i := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi64", o.args),

    vushufflelo2_16x8i := (self, o, i, is) >> self.prefix("_mm_shufflelo_epi16", o.args),
    vushufflehi2_16x8i := (self, o, i, is) >> self.prefix("_mm_shufflehi_epi16", o.args),
    vushuffle4_16x8i   := (self, o, i, is) >> self.prefix("_mm_shuffle_epi32", o.args),

    interleavedmasklo_16x8i := (self, o, i, is) >> Print("_mm_movemask_epi8(_mm_unpacklo_epi8(",o.args[1],",",o.args[2],"))"),
    interleavedmaskhi_16x8i := (self, o, i, is) >> Print("_mm_movemask_epi8(_mm_unpackhi_epi8(",o.args[1],",",o.args[2],"))"),
    average_16x8i           := (self, o, i, is) >> Print("_mm_avg_epu8(",o.args[1],",",o.args[2],")"),
    vmovemask_16x8i         := (self, o, i, is) >> self.prefix("_mm_movemask_epi8", o.args),

    # XXX NOTE XXXX
    # Also fix other vstoremsk's. The problem here is that after latest changes to Spiral
    # the last argument (list of strings) in vstoremsg gets wrapped into V, and strings too
    # this is super stupid+ugly
    vstoremsk_16x8i := (self, o, i, is) >> Print(Blanks(i), self.printf("_mm_maskmoveu_si128($2, _mm_set_epi8($3), $1);\n",
        [o.args[1], o.args[2], () -> PrintCS(Reversed(List(_unwrapV(o.args[3]), _unwrapV)))])),

    addsub_4x32f := (self, o, i, is) >> Checked(Length(o.args) = 2,
        CondPat(o,
           [addsub_4x32f, @TReal, @TVect], self(addsub_4x32f(vdup(o.args[1],o.t.size), o.args[2]), i, is),
           [addsub_4x32f, @TVect, @TReal], self(addsub_4x32f(o.args[1], vdup(o.args[2],o.t.size)), i, is),
           [addsub_4x32f, @TInt,  @TVect], self(addsub_4x32f(vdup(_toReal(o.args[1]),o.t.size), o.args[2]), i, is),
           [addsub_4x32f, @TVect, @TInt],  self(addsub_4x32f(o.args[1], vdup(_toReal(o.args[2]),o.t.size)), i, is),
           [addsub_4x32f, @TVect, @TVect], self.printf("_mm_addsub_ps($1, $2)", [o.args[1], o.args[2]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
    )),

    hadd_4x32f   := (self, o, i, is) >> self.printf("_mm_hadd_ps($1, $2)", [o.args[1], o.args[2]]),

    vloadu_16x8i := (self, o, i, is) >>  self.prefix("_mm_loadu_si128", o.args),

    # --------------------------------
    # ISA specific : SSE_4x32i
    #
    vunpacklo_4x32i := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi32", o.args),
    vunpackhi_4x32i := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi32", o.args),
    vpacks_4x32i    := (self, o, i, is) >> self.prefix("_mm_packs_epi32", o.args),
    # 32 bit integer shuffles are *not* the same as 32 bit float, but similar to 16 bit integer shuffles
    vushuffle_4x32i := (self, o, i, is) >> self.prefix("_mm_shuffle_epi32", o.args),
    vshuffle_4x32i  := (self, o, i, is) >> self.printf(
	"_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps($1), _mm_castsi128_ps($2), $3))", o.args),

    # subvector unparsing not yet done...
    vload1_4x32i := (self, o, i, is) >> self.prefix("_mm_cvtsi32_si128", o.args), #svpcprint guy
    vload2_4x32i := (self, o, i, is) >> self.prefix("_mm_loadl_epi64", o.args),
    vload2_4x32i := (self, o, i, is) >> self.prefix("_mm_loadl_epi64", o.args),
    vloadu_4x32i := (self, o, i, is) >> self.prefix("_mm_loadu_si128", o.args),

    vextract_4x32i  := (self, o, i, is) >> self.prefix("_mm_cvtsi128_si32", o.args),
    vstoreu_4x32i   := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storeu_si128", o.args), ";\n"),
    vstore2_4x32i   := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm_storel_epi64", o.args), ";\n"),
    vstoremsk_4x32i := (self, o, i, is) >> Print(Blanks(i),
	self.printf("_mm_maskmoveu_si128($2, _mm_set_epi16($3), $1);\n", [o.args[1], o.args[2], ()->PrintCS(Reversed(o.args[3]))])),
        # complicated, buggy

    interleavedmask_4x32i := (self, o, i, is) >> self.printf(
	"_mm_movemask_epi8(_mm_packs_epi16(_mm_unpacklo_epi16(_mm_packs_epi16($1, $3), _mm_packs_epi16($2, $3)), $3))",
	o.args :: ["_mm_setzero_si128()"]),

    vcvt_4x32_i2f  := (self, o, i, is) >> self.prefix("_mm_cvtepi32_ps", o.args),
    vcvt_4x32_f2i  := (self, o, i, is) >> self.prefix("_mm_cvtps_epi32", o.args),
    vcvtt_4x32_f2i := (self, o, i, is) >> self.prefix("_mm_cvttps_epi32", o.args),

    testz_4x32i := (self, o, i, is) >> self.prefix("_mm_testz_si128", o.args),
    testc_4x32i := (self, o, i, is) >> self.prefix("_mm_testc_si128", o.args),
    testnzc_4x32i := (self, o, i, is) >> self.prefix("_mm_testnzc_si128", o.args),

    # --------------------------------
    # ISA specific : SSE_2x64i
    #
    vunpacklo_2x64i := (self, o, i, is) >> self.prefix("_mm_unpacklo_epi64", o.args),
    vunpackhi_2x64i := (self, o, i, is) >> self.prefix("_mm_unpackhi_epi64", o.args),
    vshuffle_2x64i  := (self, o, i, is) >> self.printf(
	"_mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd($1), _mm_castsi128_pd($2), $3))", o.args),
    vushuffle_2x64i := (self, o, i, is) >> self(o.binop(o.args[1], o.args[1], o.args[2]), i, is),

    # --------------------------------
    # tcast __m128 <-> __m128i

    tcast := (self, o, i, is) >> let(
        isa := _isa(self),
        i128 := @.cond(x-> let( t := When(IsType(x), x, x.t), IsVecT(t) and self.ctype(t, isa)="__m128i")),
        f128 := @.cond(x-> let( t := When(IsType(x), x, x.t), IsVecT(t) and self.ctype(t, isa)="__m128" )),
        d128 := @.cond(x-> let( t := When(IsType(x), x, x.t), IsVecT(t) and self.ctype(t, isa)="__m128d" )),
        CondPat(o,
            [tcast, i128, f128], self.prefix("_mm_castps_si128", [o.args[2]]),
            [tcast, i128, d128], self.prefix("_mm_castpd_si128", [o.args[2]]),
            [tcast, f128, i128], self.prefix("_mm_castsi128_ps", [o.args[2]]),
            [tcast, i128, i128], self(o.args[2], i, is),
            [tcast, f128, f128], self(o.args[2], i, is),
            Inherited(o, i, is))),

    tcvt := (self, o, i, is) >> self.printf("(($1)($2))", [o.args[1], o.args[2]]),

    #NOTE: finish this, it should look at TVect.size and instruction set for figuring out exactly what to do
    vcastizxlo := (self, o, i, is) >> self(vunpacklo_16x8i(o.args[1], o.t.zero()), i, is),
    vcastizxhi := (self, o, i, is) >> self(vunpackhi_16x8i(o.args[1], o.t.zero()), i, is),
    vcastuzxlo := ~.vcastizxlo,
    vcastuzxhi := ~.vcastizxhi,

    average := (self, o, i, is) >> CondPat(o,
        [average, @TVect, @TVect], let(
            sfx := self.ctype_suffix(o.t, _isa(self)),
            Cond( sfx in ["epu8", "epu16"],
                self.printf("_mm_avg_$1($2, $3)",[sfx, o.args[1],o.args[2]]),
                Error("finish SSE unparser"))),
        Inherited(o, i, is)),
));
