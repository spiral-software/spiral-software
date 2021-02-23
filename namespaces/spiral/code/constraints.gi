
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


IsIntSym     := x -> (IsSymbolic(x) and IsOrdT(x.t)) or (IsValue(x) and IsOrdT(x.t)) or IsInt(x);
IsPosIntSym  := x -> (IsSymbolic(x) and IsOrdT(x.t)) or (IsValue(x) and IsOrdT(x.t) and x.v > 0)  or (IsInt(x) and x > 0);
IsPosInt0Sym := x -> (IsSymbolic(x) and IsOrdT(x.t)) or (IsValue(x) and IsOrdT(x.t) and x.v >= 0) or (IsInt(x) and x >= 0);
IsRatSym     := x -> (IsSymbolic(x) and (IsOrdT(x.t) or IsRealT(x.t) or x.t=TUnknown)) or IsRat(x) or (IsValue(x) and (IsOrdT(x.t) or IsRealT(x.t)));
IsBoolSym    := x -> x.t=TBool and (IsSymbolic(x) or IsValue(x));

AnySyms := arg -> ForAny(arg, IsSymbolic);

_ev := x->When(IsRec(x) and IsBound(x.ev), x.ev(), x);

# in functions below
# vec elements could be non-values (i.e. symbolic == expressions)

isValueZero := e ->
    (e.v=0 or IsDouble(e.v) and AbsFloat(e.v)<TReal.cutoff) or
    (IsVecT(e.t) and ForAll(e.v, x->not IsSymbolic(x) and AbsFloat(_ev(x)) < TDouble.cutoff));

isValueOne := e ->
    (e.v=1 or IsDouble(e.v) and AbsFloat(e.v-1)<TReal.cutoff) or
    (IsVecT(e.t) and ForAll(e.v, x->not IsSymbolic(x) and AbsFloat(_ev(x)-1) < TDouble.cutoff));

isValueNegOne := e ->
    (e.v=-1 or IsDouble(e.v) and AbsFloat(e.v+1)<TReal.cutoff) or
    (IsVecT(e.t) and ForAll(e.v, x->not IsSymbolic(x) and AbsFloat(_ev(x)+1) < TReal.cutoff));

_is0none := e -> Cond(IsValue(e), isValueZero(e), ObjId(e)=noneExp or e=0); 
_is1 := e -> Cond(IsValue(e), isValueOne(e),  e=1);

# in the patterns below 'e' could be a plan GAP integer or other non-class GAP object
_0    := @(0).cond(e -> Cond(IsValue(e), isValueZero(e), e=0)); 
_0none:= @(0).cond(_is0none); 
_1    := @(0).cond(_is1);
_2    := @(0).cond(x -> x=2);
_neg1 := @(0).cond(e -> Cond(IsValue(e), isValueNegOne(e), e=-1)); 

_v0    := @(0, Value, isValueZero); 
_v0none:= @(0, [Value, noneExp], e -> Cond(ObjId(e)=noneExp, true, isValueZero(e))); 
_v1    := @(0, Value, isValueOne); 
_vneg1 := @(0, Value, isValueNegOne); 

_vtrivial := @(0, Value, e->isValueZero(e) or isValueOne(e) or isValueNegOne(e));


_vtrue    := @(0, Value, e -> e.v=true); 
_vfalse    := @(0, Value, e -> e.v=false); 


Declare(_divides_rec);

_divides_rec := (d,n) -> Cond(
    ObjId(n) = add and ForAll( n.args, e -> _divides_rec(d, e) ), true,
    ObjId(n) = mul and ForAny( n.args, e -> _divides_rec(d, e) ), true,
    IsSymbolic(d) or IsSymbolic(n), d=n or n=0,
    (EvalScalar(n) mod EvalScalar(d)) = 0);

_divides := (d,n) -> CondPat( d, 
    [mul, param, param, ...], # product of distinct params
        ForAll( d.args, p -> IsParam(p) and _divides_rec(p,n) ) and Length(Set(d.args))=Length(d.args),
    [mul, Value, param],
        _divides_rec(d.args[1], n) and _divides_rec(d.args[2], n),
    # else
    _divides_rec(d,n));

_isEven := x -> _divides(2, x);

# We can't know for sure that d|n, because _dividesUnsafe returns true if either d or n is a variable
_dividesUnsafe := (d,n) -> Cond(IsSymbolic(d) or IsSymbolic(n), true,
    (EvalScalar(n) mod EvalScalar(d)) = 0);


_unwrap := n -> Cond(IsValue(n), n.v, n);
Declare(_stripval);
_stripval := n -> Cond( IsValue(n), _stripval(n.v),
                        IsList(n),  List(n, _stripval),
                        n );

_unwrapV := (n) -> let(_n := EvalScalar(n), Cond(IsValue(_n), n.v, _n));
