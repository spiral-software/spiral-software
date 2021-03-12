
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


@TInt := @.cond(x->x.t=TInt);
@TReal := @.cond(x->IsRealT(x.t));
@_scalar := @.cond(x->IsOrdT(x.t) or IsRealT(x.t) or ObjId(x.t)=TPtr);
@TVect := @.cond(x->IsVecT(x.t));
@Value := @.cond(x->IsValue(x));
@nth := @.cond(x->ObjId(x)=nth);

_isa := self -> self.opts.vector.isa;
_epi := (self, o) -> Concat("epi", self.ctype_suffixval(o.t, _isa(self)));
_px := (self, o) -> self.ctype_suffixval(o.t, _isa(self));
_vp := (o) -> let(pp := Last(o.args), 
    Cond(ObjId(pp)=vparam, pp.p,
	 IsValue(pp), pp.v,
	 pp));

Class(AVXUnparser, SSEUnparser, rec(
    # --------------------------------
    # ISA constructs, general
    # -------------------------------

    # This is a general suffix for intrinsics that is determine from the data type
    ctype_suffix := (self, t, isa) >> Cond(
        t = TVect(T_Real(64), 4), "pd",
        t = TVect(T_Real(32), 8), "ps",
        t = TVect(TReal, 4) and isa=AVX_4x64f, "pd",
        t = TVect(TReal, 8) and isa=AVX_8x32f, "ps",
        Inherited(t, isa)
    ),

    ctype_prefix := (self, t) >> Cond( _avxT(t, self.opts), "_mm256", "_mm" ),

    # This is the type used for declarations of vector variables
    ctype := (self, t, isa) >> Cond(
        t in [TReal, TVect(TReal, 1)],
          Cond(
            isa = AVX_4x64f, "double",
            isa = AVX_8x32f, "float",
            "UNKNOWN_TYPE"),
        t = T_Real(64), "double",
        t = T_Real(32), "float",
        # else
          Cond(
            t = TVect(TReal, 2), 
              Cond(
                isa = AVX_4x64f, "__m128d",
                isa = AVX_8x32f, "__m64",
                "UNKNOWN_TYPE"),
            t = TVect(TReal, 4), 
              Cond(
                isa = AVX_4x64f, "__m256d",
                isa = AVX_8x32f, "__m128",
                "UNKNOWN_TYPE"),
            t = TVect(TReal, 8), 
              Cond(
                isa = AVX_8x32f, "__m256",
                "UNKNOWN_TYPE"),
            t = TVect(T_Real(64), 4), "__m256d",
            t = TVect(T_Real(32), 8), "__m256",
            t = TVect(T_Int(32),  8), "__m256i",
            t = TVect(T_UInt(32), 8), "__m256i",
            t = TVect(T_Real(64), 2), "__m128d",
            t = TVect(T_Real(32), 4), "__m128",
            t = TVect(T_Real(32), 2), "__m64",
            Inherited(t, isa))
    ),

    cvalue_suffix  := (self, t)  >> let( isa := _isa(self), Cond(
        (t = TReal and isa = AVX_8x32f) or t = T_Real(32), "f",
        (t = TReal and isa = AVX_4x64f) or t = T_Real(64), "",
        Inherited(t)
    )),


    vhex := (self, o, i, is) >> Print("_mm_set_", _epi(self, o), "(", self.infix(Reversed(o.p), ", "), ")"),

    vparam := (self, o, i, is) >> When(Length(o.p)=1, Print(o.p[1]), iclshuffle(o.p)),

    Value := (self, o, i, is) >> Cond(
        o.t = TString, Print(o.v),

        o.t = TReal or ObjId(o.t) = T_Real, let(v := When(IsCyc(o.v), ReComplex(Complex(o.v)), Double(o.v)),
            When(v<0, Print("(", v, self.cvalue_suffix(o.t), ")"), Print(v, self.cvalue_suffix(o.t)))),

        o.t = TComplex, Print("COMPLEX(", ReComplex(Complex(o.v)), self.cvalue_suffix(TReal), ", ", ImComplex(Complex(o.v)), self.cvalue_suffix(TReal), ")"),

	o.t in [TInt, TUChar, TChar],
            When(o.v < 0, Print("(", o.v, ")"), Print(o.v)),

        ObjId(o.t) = TVect and Length(Set(o.v)) = 1,
            Cond( self.cx.isInside(Value) and Length(self.cx.Value) >= 2, # nested in an array
		    Print("{", self.infix(Replicate(o.t.size, o.v[1]), ", "), "}"),
                  # else
                    Cond( _avxT(o.t, self.opts), let( sfx := self.ctype_suffix(o.t, _isa(self)), pfx := self.ctype_prefix(o.t),
                        Cond( o.v[1] = 0, 
                            self.printf("$1_setzero_$2()", [pfx, sfx]),
                            self.printf("$1_set1_$2($3)", [pfx, sfx, o.v[1]]))),
                        Inherited(o, i, is))),
        ObjId(o.t) = TVect,
            Cond( self.cx.isInside(Value) and Length(self.cx.Value) >= 2, # nested in an array
		    Print("{", self.infix((o.v), ", "), "}"),
                # else
                    Cond( _avxT(o.t, self.opts), let( sfx := self.ctype_suffix(o.t, _isa(self)), pfx := self.ctype_prefix(o.t),
		        Print(pfx, "_set_", sfx, "(", self.infix(Reversed(o.v), ", "), ")")),
                        Inherited(o, i, is))),

        IsArray(o.t),
           Print("{", self.infix(o.v, ", "), "}"),

        ObjId(o.t) = TSym,
            Print("(", self.declare(o.t, [], 0, 0), ") ", o.v),

	o.t = TBool, Print(When(o.v, "1", "0")),

	#Error(self,".Value cannot unparse type ",o.t)
        Inherited(o, i, is)
    ),

    vpack := (self, o, i, is) >> Cond( _avxT(o.t, self.opts),
            Print("_mm256_set_", self.ctype_suffix(o.t, _isa(self)), "(", self.infix(Reversed(o.args), ", "), ")"),
        
        Inherited(o, i, is)),

    vdup := (self, o, i, is) >> CondPat(o,
            [vdup, @(1, [nth, deref]), @TInt], let( isa := _isa(self),
                t := o.args[1].t, pfx := self.ctype_prefix(o.t),
                Cond( t = T_Real(32) or (t=TReal and isa=AVX_8x32f),  
                        self.printf("$1_broadcast_ss($2)", [pfx, o.args[1].toPtr(t)]),
                      t = T_Real(64) or (t=TReal and isa=AVX_4x64f), 
                        self.printf("$1_broadcast_sd($2)", [pfx, o.args[1].toPtr(t)]),
                      t = TVect(T_Real(32), 2) or (t=TVect(TReal, 2) and isa=AVX_8x32f),
                        self.printf("$1_castpd_ps($1_broadcast_sd($2))", [pfx, o.args[1].toPtr(T_Real(64))]),
                      t = TVect(T_Real(64), 2) or (t=TVect(TReal, 2) and isa=AVX_4x64f),
                        self.printf("$1_broadcast_pd($2)", [pfx, o.args[1].toPtr(t)]),
                      t = TVect(T_Real(32), 4) or (t=TVect(TReal, 4) and isa=AVX_8x32f),
                        self.printf("$1_castpd_ps($1_broadcast_pd($2))", [pfx, o.args[1].toPtr(TVect(T_Real(64), 2))]),
                      Error("unexpected vdup load"))),
            #[vdup, @nth,@.cond(x->x.t=TInt and x.v=2)],
            #    Print("_mm_loaddup_", sfx, "(&(", self(o.args[1], i, is), "))"),
            [vdup, @, @TInt],
                Print(When(_isa(self).v = 4, "_mm256_set1_pd", "_mm256_set1_ps"), "(", self(o.args[1], i, is), ")")
        ),

    # Declarations
    TVect := (self, t, vars, i, is) >> let( ctype := self.ctype(t, _isa(self)),
        Print(ctype, " ", self.infix(vars, ", "))),

    TReal := ~.TVect,

    TInt := (self, t, vars, i, is) >> Print("int ", self.infix(vars, ", ")),

    TBool := (self, t, vars, i, is) >> Print("BOOL ", self.infix(vars, ", ")),

    # Arithmetic
    mul := (self, o, i, is) >> let(n := Length(o.args), Cond(
        not IsVecT(o.t),
            Print("(", self.pinfix(o.args, ")*("), ")"),
        not _avxT(o.t, self.opts),
            Inherited(o, i, is),
	n > 2 and n mod 2 <> 0,
            self(mul(o.args[1], ApplyFunc(mul, Drop(o.args, 1))), i, is),
        n > 2, 
            self(mul(ApplyFunc(mul, o.args{[1..n/2]}), ApplyFunc(mul, o.args{[n/2+1..n]})), i, is),
        let( sfx := self.ctype_suffix(o.t, _isa(self)),
         CondPat(o,
           [mul, @TReal, @TVect],
              self(mul(vdup(o.args[1],o.t.size), o.args[2]), i, is),
           [mul, @TVect, @TReal],
              self(mul(o.args[1], vdup(o.args[2],o.t.size)), i, is),
           [mul, @TInt, @TVect],
              self(mul(vdup(_toReal(o.args[1]),o.t.size), o.args[2]), i, is),
           [mul, @TVect, @TInt],
              self(mul(o.args[1], vdup(_toReal(o.args[2]),o.t.size)), i, is),
           [mul, @TVect,   @TVect],
              self.printf("_mm256_mul_$1($2, $3)", [sfx, o.args[1], o.args[2]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")))
    )),

    # -- add --
    add := (self, o, i, is) >> let(n := Length(o.args), Cond(
	not IsVecT(o.t),
            self.pinfix(o.args, " + "),
        not _avxT(o.t, self.opts),
            Inherited(o, i, is),
        n > 2 and n mod 2 <> 0,
            self(add(o.args[1], ApplyFunc(add, Drop(o.args, 1))), i, is),
        n > 2, 
            self(add(ApplyFunc(add, o.args{[1..n/2]}), ApplyFunc(add, o.args{[n/2+1..n]})), i, is),
        let(sfx := self.ctype_suffix(o.t, _isa(self)), saturated:= When(_isa(self).isFixedPoint and _isa(self).saturatedArithmetic, "s", ""), 
          CondPat(o,
            [add, @TReal, @TVect],
                self(add(vdup(o.args[1],o.t.size), o.args[2]), i, is),
            [add, @TVect, @TReal],
                self(add(o.args[1], vdup(o.args[2],o.t.size)), i, is),
            [add, @TInt, @TVect],
                self(add(vdup(_toReal(o.args[1]),o.t.size), o.args[2]), i, is),
            [add, @TVect, @TInt],
                self(add(o.args[1], vdup(_toReal(o.args[2]),o.t.size)), i, is),
            [add, @TVect,   @TVect],
                self.printf("_mm256_add$1_$2($3, $4)", [saturated, sfx, o.args[1], o.args[2]]),
            Error("Don't know how to unparse <o>. Unrecognized type combination")))
    )),

    sub := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), let(
            isa       := _isa(self),
            sfx       := self.ctype_suffix(o.t, isa),
            saturated := When(isa.isFixedPoint and isa.saturatedArithmetic, "s", ""),
            self.printf("_mm256_sub$1_$2($3, $4)", [saturated, sfx, o.args[1], o.args[2]])),
        # else
            Inherited(o, i, is)),

    neg := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), 
        self(mul(neg(o.t.one()), o.args[1]), i, is),
        Inherited(o, i, is)),

    stickyNeg := ~.neg,

    sqrt  := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), 
        Checked( IsRealT(o.t.t), self.printf("_mm256_sqrt_$1($2)", [self.ctype_suffix(o.t, _isa(self)), o.args[1]])),
        Inherited(o, i, is)),

    rsqrt := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), 
        let( sfx := self.ctype_suffix(o.t, _isa(self)),
            Checked( sfx="ps", self.printf("_mm256_rsqrt_ps($1)", [o.args[1]]))),
        Inherited(o, i, is)),

    max := (self, o, i, is) >> let(n := Length(o.args), Cond(
        not _avxT(o.t, self.opts),
            Inherited(o, i, is),
	n > 2 and n mod 2 <> 0,
            self(max(o.args[1], ApplyFunc(max, Drop(o.args, 1))), i, is),
        n > 2, 
            self(max(ApplyFunc(max, o.args{[1..n/2]}), ApplyFunc(max, o.args{[n/2+1..n]})), i, is),
        let( sfx := self.ctype_suffix(o.t, _isa(self)),
         CondPat(o,
           [max, @TReal, @TVect],
              self(max(vdup(o.args[1],o.t.size), o.args[2]), i, is),
           [max, @TVect, @TReal],
              self(max(o.args[1], vdup(o.args[2],o.t.size)), i, is),
           [max, @TInt, @TVect],
              self(max(vdup(_toReal(o.args[1]),o.t.size), o.args[2]), i, is),
           [max, @TVect, @TInt],
              self(max(o.args[1], vdup(_toReal(o.args[2]),o.t.size)), i, is),
           [max, @TVect,   @TVect],
              self.printf("_mm256_max_$1($2, $3)", [sfx, o.args[1], o.args[2]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")))
    )),

    min := (self, o, i, is) >> let(n := Length(o.args), Cond(
        not _avxT(o.t, self.opts),
            Inherited(o, i, is),
	n > 2 and n mod 2 <> 0,
            self(min(o.args[1], ApplyFunc(min, Drop(o.args, 1))), i, is),
        n > 2, 
            self(min(ApplyFunc(min, o.args{[1..n/2]}), ApplyFunc(min, o.args{[n/2+1..n]})), i, is),
        let( sfx := self.ctype_suffix(o.t, _isa(self)),
         CondPat(o,
           [min, @TReal, @TVect],
              self(min(vdup(o.args[1],o.t.size), o.args[2]), i, is),
           [min, @TVect, @TReal],
              self(min(o.args[1], vdup(o.args[2],o.t.size)), i, is),
           [min, @TInt, @TVect],
              self(min(vdup(_toReal(o.args[1]),o.t.size), o.args[2]), i, is),
           [min, @TVect, @TInt],
              self(min(o.args[1], vdup(_toReal(o.args[2]),o.t.size)), i, is),
           [min, @TVect,   @TVect],
              self.printf("_mm256_min_$1($2, $3)", [sfx, o.args[1], o.args[2]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")))
    )),
    
    # assuming we have ICC <ia32intrin.h> here
    log := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), let( sfx := self.ctype_suffix(o.t, _isa(self)),
        Checked( sfx in ["ps", "pd"], Cond(
            Length(o.args)>1 and (o.args[2]=2 or o.args[2]=o.t.value(2)),
                self.printf("_mm256_log2_$1($2)", [sfx, o.args[1]]),
            Length(o.args)>1 and (o.args[2]=10 or o.args[2]=o.t.value(10)),
                self.printf("_mm256_log10_$1($2)", [sfx, o.args[1]]),
            Length(o.args)=1 or o.args[2]=d_exp(1) or o.args[2]=o.t.value(d_exp(1)),
                self.printf("_mm256_log_$1($2)", [sfx, o.args[1]]),
            self.printf("_mm256_div_$1(_mm256_log_$1($2), _mm256_log_$1($3))", [sfx, o.args[1]])))),
        Inherited(o, i, is)),

    # assuming we have ICC <ia32intrin.h> here
    exp := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), let( sfx := self.ctype_suffix(o.t, _isa(self)),
        Checked( sfx in ["ps", "pd"], self.printf("_mm256_exp_$1($2)", [sfx, o.args[1]]))),
        Inherited(o, i, is)),

    # assuming we have ICC <ia32intrin.h> here
    pow := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), let( sfx := self.ctype_suffix(o.t, _isa(self)),
        Checked( sfx in ["ps", "pd"], Cond( 
            o.args[1]=2 or o.args[1]=o.t.value(2),
                self.printf("_mm256_exp2_$1($2)", [sfx, o.args[2]]),
            o.args[1]=d_exp(1) or o.args[1]=o.t.value(d_exp(1)),
                self.printf("_mm256_exp_$1($2)", [sfx, o.args[2]]),
            self.printf("_mm256_pow_$1($2, $3)", [sfx, o.args[1], o.args[2]])))),
        Inherited(o, i, is)),

    # --------------------------------
    # logic
    #
    bin_xor := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), 
        self.printf("_mm256_xor_$1($2, $3)", [self.ctype_suffix(o.t, _isa(self)), o.args[1], o.args[2]]),
        Inherited(o, i, is)),
    bin_and := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), 
        self.printf("_mm256_and_$1($2, $3)", [self.ctype_suffix(o.t, _isa(self)), o.args[1], o.args[2]]),
        Inherited(o, i, is)),
    bin_or := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), 
        self.printf("_mm256_or_$1($2, $3)", [self.ctype_suffix(o.t, _isa(self)), o.args[1], o.args[2]]),
        Inherited(o, i, is)),
    bin_andnot := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), 
        self.printf("_mm256_andnot_$1($2, $3)", [self.ctype_suffix(o.t, _isa(self)), o.args[1], o.args[2]]),
        Inherited(o, i, is)),

    # --------------------------------
    # ISA specific : AVX_4x64f
    #
    cmpge_4x64f := (self, o, i, is) >> self.prefix("_mm256_cmp_pd", Concat(o.args, ["_CMP_GE_OQ"])),
    logic_and_4x64f := (self, o, i, is) >> self.prefix("_mm256_and_pd", o.args),
    logic_xor_4x64f := (self, o, i, is) >> self.prefix("_mm256_xor_pd", o.args),


    vloadu_4x64f   := (self, o, i, is) >> self.prefix("_mm256_loadu_pd", o.args),
    vstoreu_4x64f  := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm256_storeu_pd", o.args), ";\n"),

    vinsert_2l_4x64f := (self, o, i, is) >> self.prefix("_mm256_insertf128_pd", o.args),
    vloadmask_4x64f  := (self, o, i, is) >> self.printf("_mm256_maskload_pd($1, _mm256_set_epi32($2))", [o.args[1], ()->PrintDel(_vp(o), ", ")]),
    vbroadcast_4x64f := (self, o, i, is) >> self.printf("_mm256_broadcast_sd($1)", [o.args[1]]),
    # doublecheck this
    vextract_2l_4x64f := (self, o, i, is) >> self.prefix("_mm256_extractf128_pd", o.args),

    vstore_2l_4x64f := (self, o, i, is) >> self.prefix("_mm256_extractf128_pd", o.args),
    vstoremask_4x64f := (self, o, i, is) >> Print(Blanks(i), self.printf("_mm256_maskstore_pd($1, _mm256_set_epi32($3), $2)", [o.args[1], o.args[2], ()->PrintDel(List(_vp(o), e->e.v), ", ")]), ";\n"),

    vunpacklo_4x64f := (self, o, i, is) >> self.prefix("_mm256_unpacklo_pd", o.args),
    vunpackhi_4x64f := (self, o, i, is) >> self.prefix("_mm256_unpackhi_pd", o.args),
    vpermf128_4x64f := (self, o, i, is) >> self.printf("_mm256_permute2f128_pd($1, $2, ($3) | (($4) << 4))",
        let(l := _vp(o)-1, [o.args[1], o.args[2], l[1], l[2]])),
    vshuffle_4x64f := (self, o, i, is) >> self.printf("_mm256_shuffle_pd($1, $2, ($3) | (($4) << 1) | (($5) << 2) | (($6) << 3))",
        let(l := _vp(o)-1, [o.args[1], o.args[2], l[1], l[2], l[3], l[4]])),
    vperm_4x64f := (self, o, i, is) >> self.printf("_mm256_permute_pd($1, ($2) | (($3) << 1) | (($4) << 2) | (($5) << 3))",
        let(l := _vp(o)-1, [o.args[1], l[1], l[2], l[3], l[4]])),
    vblend_4x64f := (self, o, i, is) >> self.printf("_mm256_blend_pd($1, $2, ($3) | (($4) << 1) | (($5) << 2) | (($6) << 3))",
        let(l := _vp(o)-1, [o.args[1], o.args[2], l[1], l[2], l[3], l[4]])),

    vuunpacklo_4x64f := (self, o, i, is) >> self(o.toBinop(), i, is),
    vuunpackhi_4x64f := (self, o, i, is) >> self(o.toBinop(), i, is),
    vupermf128_4x64f := (self, o, i, is) >> self(o.toBinop(), i, is),
    vushuffle_4x64f  := (self, o, i, is) >> self(o.toBinop(), i, is),
    vuperm2_4x64f    := (self, o, i, is) >> self(o.toBinop(), i, is),
    
    
    addsub_4x64f := (self, o, i, is) >> When(Length(o.args) > 2,
        Error("addsub_2x64f is strictly binary"),
        CondPat(o,
           [addsub_4x64f, @TReal, @TVect],
              self(_computeExpType(addsub_4x64f(vdup(o.args[1],o.t.size), o.args[2])), i, is),
           [addsub_4x64f, @TVect, @TReal],
              self(_computeExpType(addsub_4x64f(o.args[1], vdup(o.args[2],o.t.size))), i, is),
           [addsub_4x64f, @TInt, @TVect],
              self(_computeExpType(addsub_4x64f(vdup(_toReal(o.args[1]),o.t.size), o.args[2])), i, is),
           [addsub_4x64f, @TVect, @TInt],
              self(_computeExpType(addsub_4x64f(o.args[1], vdup(_toReal(o.args[2]),o.t.size))), i, is),
           [addsub_4x64f, @TVect,   @TVect],
              self.printf("_mm256_addsub_pd($1, $2)", [o.args[1], o.args[2]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
        )),

    fmaddsub_4x64f := (self, o, i, is) >> When(
        Length(o.args) <> 3, Error("fmaddsub_4x64f is strictly ternary"),
        CondPat(o,
           [fmaddsub_4x64f, @TVect, @TVect, @TVect],
              self.printf("_mm256_fmaddsub_pd($1, $2, $3, 0)", [o.args[1], o.args[2], o.args[3]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
        )),

    vzero_4x64f := (self, o, i, is) >> Print("_mm256_setzero_pd()"),

    # --------------------------------
    # ISA specific : AVX_8x32f
    #
    logic_and_8x32f := (self, o, i, is) >> self.prefix("_mm256_and_ps", o.args),
    logic_xor_8x32f := (self, o, i, is) >> self.prefix("_mm256_xor_ps", o.args),

    vloadu_8x32f      := (self, o, i, is) >> self.prefix("_mm256_loadu_ps", o.args),
    vstoreu_8x32f     := (self, o, i, is) >> Print(Blanks(i), self.prefix("_mm256_storeu_ps", o.args), ";\n"),

    vinsert_4l_8x32f  := (self, o, i, is) >> self.prefix("_mm256_insertf128_ps", o.args),
    vloadmask_8x32f   := (self, o, i, is) >> self.printf("_mm256_maskload_ps($1, _mm256_set_epi32($2))", [o.args[1], ()->PrintDel(_vp(o), ", ")]),
    vhdup_8x32f       := (self, o, i, is) >> self.printf("_mm256_movehdup_ps($1)", [o.args[1]]),
    vldup_8x32f       := (self, o, i, is) >> self.printf("_mm256_moveldup_ps($1)", [o.args[1]]),

    vextract_4l_8x32f := (self, o, i, is) >> self.prefix("_mm256_extractf128_ps", o.args),
    vstore_4l_8x32f   := (self, o, i, is) >> self.prefix("_mm256_extractf128_ps", o.args),
    vstoremask_8x32f  := (self, o, i, is) >> Print(Blanks(i), self.printf("_mm256_maskstore_ps($1, _mm256_set_epi32($3), $2)", [o.args[1], o.args[2], ()->PrintDel(List(_vp(o), e->e.v), ", ")]), ";\n"),

    vunpacklo_8x32f   := (self, o, i, is) >> self.prefix("_mm256_unpacklo_ps", o.args),
    vunpackhi_8x32f   := (self, o, i, is) >> self.prefix("_mm256_unpackhi_ps", o.args),

    vpermf128_8x32f   := (self, o, i, is) >> self.printf("_mm256_permute2f128_ps($1, $2, ($3) | (($4) << 4))",
        let(l := _vp(o)-1, [o.args[1], o.args[2], l[1], l[2]])),

    vshuffle_8x32f    := (self, o, i, is) >> self.printf("_mm256_shuffle_ps($1, $2, ($3) | (($4) << 2) | (($5) << 4) | (($6) << 6))",
        let(l := _vp(o)-1, [o.args[1], o.args[2], l[1], l[2], l[3], l[4]])),

    vperm_8x32f       := (self, o, i, is) >> self.printf("_mm256_permute_ps($1, ($2) | (($3) << 2) | (($4) << 4) | (($5) << 6))",
        let(l := _vp(o)-1, [o.args[1], l[1], l[2], l[3], l[4]])),
    vblend_8x32f := (self, o, i, is) >> self.printf("_mm256_blend_ps($1, $2, ($3) | (($4) << 1) | (($5) << 2) | (($6) << 3) | (($7) << 4) | (($8) << 5) | (($9) << 6) | (($10) << 7))",
        let(l := _vp(o)-1, [o.args[1], o.args[2], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]])),

    vuunpacklo_8x32f := (self, o, i, is) >> self(o.toBinop(), i, is),
    vuunpackhi_8x32f := (self, o, i, is) >> self(o.toBinop(), i, is),
    vupermf128_8x32f := (self, o, i, is) >> self(o.toBinop(), i, is),
    vushuffle_8x32f  := (self, o, i, is) >> self(o.toBinop(), i, is),
    
    addsub_8x32f := (self, o, i, is) >> When(Length(o.args) > 2,
        Error("addsub_2x64f is strictly binary"),
        CondPat(o,
           [addsub_8x32f, @TReal, @TVect],
              self(_computeExpType(addsub_8x32f(vdup(o.args[1],o.t.size), o.args[2])), i, is),
           [addsub_8x32f, @TVect, @TReal],
              self(_computeExpType(addsub_8x32f(o.args[1], vdup(o.args[2],o.t.size))), i, is),
           [addsub_8x32f, @TInt, @TVect],
              self(_computeExpType(addsub_8x32f(vdup(_toReal(o.args[1]),o.t.size), o.args[2])), i, is),
           [addsub_8x32f, @TVect, @TInt],
              self(_computeExpType(addsub_8x32f(o.args[1], vdup(_toReal(o.args[2]),o.t.size))), i, is),
           [addsub_8x32f, @TVect,   @TVect],
              self.printf("_mm256_addsub_ps($1, $2)", [o.args[1], o.args[2]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
        )),

    fmaddsub_8x32f := (self, o, i, is) >> When(
        Length(o.args) <> 3, Error("fmaddsub_8x32f is strictly ternary"),
        CondPat(o,
           [fmaddsub_8x32f, @TVect, @TVect, @TVect],
              self.printf("_mm256_fmaddsub_ps($1, $2, $3, 0)", [o.args[1], o.args[2], o.args[3]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
        )),

    vzero_8x32f := (self, o, i, is) >> Print("_mm256_setzero_ps()"),

    #--------------------------------
    # Conversion

    vcvt_8x32f_4x64f  := (self, o, i, is) >> self.printf("_mm256_cvtpd_ps($1)", [o.args[1]]),
    vcvt_4x64f_4x32f  := (self, o, i, is) >> self.printf("_mm256_cvtps_pd($1)", [o.args[1]]),
    vcvt_4x64f_4x32i  := (self, o, i, is) >> self.printf("_mm256_cvtepi32_pd($1)", [o.args[1]]),
    vcvt_4x32i_4x64f  := (self, o, i, is) >> self.printf("_mm256_cvtpd_epi32($1)", [o.args[1]]),
    vcvtt_4x32i_4x64f := (self, o, i, is) >> self.printf("_mm256_cvttpd_epi32($1)", [o.args[1]]),
    vcvt_8x32f_8x32i  := (self, o, i, is) >> self.printf("_mm256_cvtepi32_ps($1)", [o.args[1]]),
    vcvt_8x32i_8x32f  := (self, o, i, is) >> self.printf("_mm256_cvtps_epi32($1)", [o.args[1]]),
    vcvtt_8x32i_8x32f := (self, o, i, is) >> self.printf("_mm256_cvttps_epi32($1)", [o.args[1]]),



));
