
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(assign_cmd, ExpCommand, rec(
    op_in := self >> Set( List(Drop(self.args,1), ArgsExp)), 
    op_out := self >> Set([self.args[1]]),
    op_inout := self >> Set([]),
    unparse := "assign_cmd",
));

Class(neg_cmd, assign_cmd);

Class(fma_cmd,  assign_cmd, rec(exp_op := fma));
Class(fms_cmd,  assign_cmd, rec(exp_op := fms));
Class(nfma_cmd, assign_cmd, rec(exp_op := nfma));

# Examples: 
#   assign(t1, add(t1, t2)) == assign_add(t1, t2),
#   assign(t1, add(t2, t3)) == chain(assign(t1, t2), assign_add(t2, t3)).
#

_threeop_bintab := WeakRef(tab());
_threeop_binlist := [];

ThreeOpFromBinOp := function(arg)
    local op;
    for op in arg do
        _threeop_bintab.(op.exp_op.__name__) := op;
        Add(_threeop_binlist, op.exp_op);
    od;
end;


ThreeOpFromBinOp(
   Class(add_cmd, assign_cmd, rec( exp_op := add )),
   Class(mul_cmd, assign_cmd, rec( exp_op := mul )),
   Class(sub_cmd, assign_cmd, rec( exp_op := sub ))
);


Class(ThreeOpRuleSet, RuleSet);
RewriteRules(ThreeOpRuleSet, rec(
    neg := Rule([assign, @(1), [neg, @(2)]], e -> neg_cmd(@(1).val, @(2).val)),

    fma  := Rule([assign, @(1), [ fma, @(2), @(3), @(4)]], e ->  fma_cmd(@(1).val, @(2).val, @(3).val, @(4).val)), 
    fms  := Rule([assign, @(1), [ fms, @(2), @(3), @(4)]], e ->  fms_cmd(@(1).val, @(2).val, @(3).val, @(4).val)), 
    nfma := Rule([assign, @(1), [nfma, @(2), @(3), @(4)]], e -> nfma_cmd(@(1).val, @(2).val, @(3).val, @(4).val)), 

    binop := Rule([assign, @(1), [@(0,_threeop_binlist) , @(2), @(3)]], e -> _threeop_bintab.(@(0).val.__name__)(@(1).val, @(2).val, @(3).val)), 
));

ThreeOpMacroUnparser_Mixin := rec(

    add_cmd := (self,o,i,is) >> Print(Blanks(i), self.prefixTTT("ADD", o.args[1].t, o.args[2].t, o.args[3].t, o.args), ";\n"),
    sub_cmd := (self,o,i,is) >> Print(Blanks(i), self.prefixTTT("SUB", o.args[1].t, o.args[2].t, o.args[3].t, o.args), ";\n"),
    mul_cmd := (self,o,i,is) >> Print(Blanks(i), self.prefixTTT("MUL", o.args[1].t, o.args[2].t, o.args[3].t, o.args), ";\n"),
    neg_cmd := (self,o,i,is) >> Print(Blanks(i), self.prefixTT( "NEG", o.args[1].t, o.args[2].t, o.args), ";\n"),

    fma_cmd  := (self,o,i,is) >> Print(Blanks(i), self.prefixTTTT("FMA", o.args[1].t, o.args[2].t, o.args[3].t, o.args[4].t, o.args), ";\n"),
    fms_cmd  := (self,o,i,is) >> Print(Blanks(i), self.prefixTTTT("FMS", o.args[1].t, o.args[2].t, o.args[3].t, o.args[4].t, o.args), ";\n"),
    nfma_cmd := (self,o,i,is) >> Print(Blanks(i), self.prefixTTTT("NFMA", o.args[1].t, o.args[2].t, o.args[3].t, o.args[4].t, o.args), ";\n"),

);

DoThreeOp := function(c, opts)
    c := MarkDefUse(c);
    c := BinSplit(c);
    c := ThreeOpRuleSet(c);
    return c;
end;

ThreeOp_CS := Concatenation(BaseIndicesCS, [
    MarkDefUse, #
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
    MarkDefUse, #
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
    DoThreeOp,
    Compile.declareVars
]);

Class(ThreeOpUnparser, CMacroUnparserProg, rec(
    add_cmd := (self,o,i,is) >> self.prefixTTT("ADD", o.args[1].t, o.args[2].t, o.args[3], o.args),

    sub_cmd := (self,o,i,is) >> self.prefixTTT("SUB", o.args[1].t, o.args[2].t, o.args[3], o.args),

    neg_cmd := (self,o,i,is) >> self.prefixT("NEG", o.t, o.args),
   
    # first argument of mul_cmd is the result, 2nd and 3rd are operands
    mul_cmd := (self,o,i,is) >> When(Length(o.args)=3,
            let(
            # check if constant ended up in slot #3
            a := When(IsValue(o.args[3]), o.args[3], o.args[2]),
            b := When(IsValue(o.args[3]), o.args[2], o.args[3]),
            When(not (IsValue(a) or (IsVar(a) and IsBound(a.value))),
                # <a> is not a constant
                self.prefixTTT("MUL", o.args[1], a, b, [o.args[1],a,b]),
                #self.prefix(Concat("MUL_", self._pfx(a.t), "_", self._pfx(b.t)), [o.args[1],a,b]),
                # <a> is a constant
                let(fmt := self._const( When(IsValue(a), a, a.value) ),
                    self.prefix(Concat("MUL_", o.args[1]._pfx(o.args[1].t), "CNST", "_", self._pfx(b.t)), [o.args[1], GetExponent(a.value), GetMantissa(a.value), b]))))),
#                   self.prefix(Concat("MUL_", o.args[1]._pfx(o.args[1].t), fmt[1], "_", self._pfx(b.t)), [o.args[1], a, b]))))),
#                        # check if constant is special, and does not go into MUL args
#                        # for example fmt[1]="I" denotes sqrt(-1), one such constant
#                        #When(fmt[2]=[], [b], [a, b])))))),

#    add_cmd := (self, o, i, is) >> Print(Blanks(i), self.prefix(self, i, is), ";\n"),
#    sub_cmd := ~.add_cmd

    prefixT := (self, funcname, t, args) >>
        self.prefix(Concat(funcname, "_", self._pfx(t)), args),

    prefixTT := (self, funcname, t1, t2, args) >>
        self.prefix(Concat(funcname, "_", self._pfx(t1), "_", self._pfx(t2)), args),

    prefixTTT := (self, funcname, t1, t2, t3, args) >> 
        self.prefix(Concat(funcname, "_", self._pfx(t1), "_", self._pfx(t2), "_", self._pfx(t3)), args),
    
    _pfx := (self, t) >> Cond(
        ObjId(t) = T_Complex, "CPX",
        ObjId(t) = T_Real,    "FLT",
        t = TComplex, "CPX",
        t = TReal,    "FLT",
        t = TInt or ObjId(t) in [TArray, TPtr], "INT",
        t = TUnknown, "UNK",
        ObjId(t) = TSym, "SYM",
        IsVecT(t), Cond(
            t.t = TReal, Concat("FV",StringInt(t.size)),
            t.t = TComplex, Concat("FC",StringInt(t.size)),
            t.t = TInt, Concat("IV",StringInt(t.size)),
            Error("Can't handle type ", t)
        ),
        Error("Can't handle type ", t)),

    # returns a tuple [suffix, args], where suffix is used for MUL_XXX or C_XXX,
    # and args are additional parameters into C_XXX
    _const := (self,o) >> Cond(
        o.t = TReal and IsCyc(o.v),        ["FLT", [ReComplex(Complex(o.v))]],
        o.t = TReal,                       ["FLT", [o.v]],
        o.t = TInt,                        ["INT", [o.v]],
        o.t = TUnknown,                    ["INT", [o.v]], # NOTE: there is a bug that creates V(0) with TUnknown
        o.t = TString,                     ["STR", [o.v]],
        o.t = TBool,                       ["INT", [When(o.v in [true, 1], 1, 0)]],
        Error("Don't know how to handle constant of type ", o.t)
    ),
));


EncodeFloat := function(f)
    local a, exp, mant, tmp, flag;

    #shift by max exponent, which is +/- 127 in IEEE floating point
    if (IntDouble(f)=0) then
        exp := Log2Int(IntDouble(f*2^127))-127;
    else
        exp := Log2Int(IntDouble(f));
    fi;

    tmp := f/(2^exp);
    mant := tmp - IntDouble(tmp);
    mant := IntDouble(mant * 2^23);
   
        a:= rec(exp:= exp+127, mant:=mant);

    return a;
end;

DecodeToFloat := function(exp, mant)
    local f;  

    if (mant<0) then
        mant := -mant;
        f := mant / (2^23);
        f := f + 1;
        f:= f * (2^(exp-127));
        f:= -f;
    else
        f := mant / (2^23);
        f := f + 1;
        f:= f * (2^(exp-127));
    fi;

    return f;
end;

GetExponent := function(f)
    local exp;

    #shift by max exponent, which is +/- 127 in IEEE floating point
    if (IntDouble(f)=0) then
        exp := Log2Int(IntDouble(f*2^127))-127;
    else
        exp := Log2Int(IntDouble(f));
    fi;

    return exp + 127;
end;

GetMantissa := function(f)
    local a, exp, mant, tmp, flag;

    #shift by max exponent, which is +/- 127 in IEEE floating point
    if (IntDouble(f)=0) then
        exp := Log2Int(IntDouble(f*2^127))-127;
    else
        exp := Log2Int(IntDouble(f));
    fi;

    tmp := f/(2^exp);
    mant := tmp - IntDouble(tmp);
    mant := IntDouble(mant * 2^23);

    return mant;
end;
