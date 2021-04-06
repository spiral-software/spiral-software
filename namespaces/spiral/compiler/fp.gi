
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


SignedFixedPointValue := (val, b, fb) ->  Checked(fb <= b-1, Cond(
    IsIntT(val.t),    val,

    IsRealT(val.t),   T_Int(b).value(_fpdouble(val.v, b, fb)),

    IsArrayT(val.t), 
        let(newv := List(val.v, v -> SignedFixedPointValue(v, b, fb)),
	    newt := newv[1].t,
	    CopyFields(val.t, rec(t:=newt)).value(newv)),

    IsVecT(val.t), TVect(T_Int(b), val.t.size).value(List(val.v, x->_fpdouble(x,b,fb))),

    IsComplexT(val.t), let(v := Complex(val.v),
	                   re := ReComplex(v),
                           im := ImComplex(v),
			   T_Complex(T_Int(b)).value(
			       Complex(_fpdouble(re, b, fb), _fpdouble(im, b, fb)))),
    val
));

FixedPointCode := function(code, bitwidth, fracbits)
    code := SubstBottomUp(code, [mul, @(1).cond(e->not IsValue(e) or not IsIntT(e.t)), 
	                              @(2).cond(e->not IsValue(e) or not IsIntT(e.t))], 
	e -> fpmul(fracbits, @(1).val, @(2).val));
    code := SubstTopDownNR(code, Value, v -> SignedFixedPointValue(v, bitwidth, fracbits));
    return code;
end;

# FixedPointCode2(<code>)
# YSV
# New, cleaner way to construct fixed point code
# NOTE: 'fixValueTypes' rule makes the below totally unnecessary
#
FixedPointCode2 := c -> SubstBottomUpRules(c, [
	[ @(1, mul, x->Length(x.args) > 2), e -> FoldR1(e.args, mul), "binsplit_mul" ],
	[ [mul, @(1, Value, x -> not IsFixedPtT(x.t)), @(2).cond(x->IsFixedPtT(x.t))],
	  e -> fpmul(@(2).val.t.fracbits, 
	             SignedFixedPointValue(@(1).val, @(2).val.t.bits, @(2).val.t.fracbits), 
		     @(2).val), 
	  "mul_val_fixedpt" ],
	[ [mul, @(1, Value, x -> not IsFixedPtT(x.t)), @(2).cond(x->IsFixedPtT(x.t))],
	  e -> fpmul(@(2).val.t.fracbits, 
	             SignedFixedPointValue(@(1).val, @(2).val.t.bits, @(2).val.t.fracbits), 
		     @(2).val), 
	  "mul_val_fixedpt" ]
    ]);


UnsignedFixedPointValue := (val, b, fb) ->  Checked(fb <= b, Cond(
    val.v < 0, Error("UnsignedFixedPointValue: unexpected signed value ", val),
    IsIntT(val.t),    val,
    IsRealT(val.t),   T_UInt(b).value(_ufpdouble(val.v, b, fb)),

    IsArrayT(val.t), 
        let(newv := List(val.v, v -> UnsignedFixedPointValue(v, b, fb)),
	    newt := newv[1].t,
	    CopyFields(val.t, rec(t:=newt)).value(newv)),

    IsVecT(val.t), TVect(T_Int(b), val.t.size).value(List(val.v, x->_ufpdouble(x,b,fb))),

    IsComplexT(val.t), let(re := IntDouble(ReComplex(Complex(val.v)) * 2.0^fb),
                           im := IntDouble(ImComplex(Complex(val.v)) * 2.0^fb),
		           T_Complex(T_UInt(b)).value(
			       Complex(_ufpdouble(re, b, fp), _ufpdouble(re, b, fp)))),
    val
));

UnsignedFixedPointCode := function(code, bitwidth, fracbits)
    code := SubstBottomUp(code, [mul, @(1).cond(e->not IsValue(e) or not IsIntT(e.t)), 
	                              @(2).cond(e->not IsValue(e) or not IsIntT(e.t))], 
	e -> fpmul(fracbits, @(1).val, @(2).val));
    code := SubstTopDownNR(code, Value, v -> UnsignedFixedPointValue(v, bitwidth, fracbits));
    return code;
end;

# X.t.t := TInt;
# Y.t.t := TInt;
# TComplex.ctype := "struct{int r,i;}";
