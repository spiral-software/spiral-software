
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


UnifyPair := (aa,bb) -> let(
    # NOTE: this converts 2->TInt, this is a hack
    a := Cond(IsInt(aa) or IsSymbolic(aa), TInt, ObjId(aa)=TFunc, _convType(aa), aa),
    b := Cond(IsInt(bb) or IsSymbolic(bb), TInt, ObjId(bb)=TFunc, _convType(bb), bb), 
    aid := ObjId(a),    bid := ObjId(b),  avec := IsVecT(aa), bvec := IsVecT(bb), 
    Cond(

     # YSV: this handles subtraction of pointers (=integer index)
     #      addition of pointers is not valid
#     aid=TPtr and bid=TPtr, TInt, 

     a=b, a,
     a = TUnknown and b <> TUnknown, b,
     b = TUnknown and a <> TUnknown, a,

#     aid in [TPtr, TArray] and (b=TInt or bid in [T_Int, T_UInt]), TPtr(a.t, When(aid=TArray, [], a.qualifiers), When(aid=TArray, ptrAligned, a.alignment)),
#     bid in [TPtr, TArray] and (a=TInt or aid in [T_Int, T_UInt]), TPtr(b.t, When(bid=TArray, [], b.qualifiers), When(bid=TArray, ptrAligned, b.alignment)),
     #NOTE: this looks like an horrible hack but how else can it be done?
     #isn't this a fundamental weakness of C/C++ ?

#     aid=TPtr and bid=TArray, a,
#     aid=TArray and bid=TPtr, b,
 
     avec and bvec and a.size<>b.size,     TVect(UnifyPair(a.t, b.t), Maximum(a.size, b.size)),
     avec and bvec, Checked(a.size=b.size, TVect(UnifyPair(a.t, b.t), a.size)),
     avec, TVect(UnifyPair(a.t, b), a.size),
     bvec, TVect(UnifyPair(a, b.t), b.size),

     aid=TFixedPt and bid=TFixedPt,
        When(a = b, a, Error("Can't unify fixed point types of different bit width")),

     a=TComplex and b in [TReal, TComplex, TInt, TUInt], TComplex,
     b=TComplex and a in [TReal, TComplex, TInt, TUInt], TComplex,

     (aid=T_Real or a=TReal) and (bid in [T_UInt, T_Int] or b in [TReal, TInt, TUInt]), a,
     (bid=T_Real or b=TReal) and (aid in [T_UInt, T_Int] or a in [TReal, TInt, TUInt]), b,

     a=TInt and b in  [TInt,TBool,TUInt], TInt,
     a in [TInt,TUInt,TBool] and b=TInt,  TInt,

     aid=T_Complex and b in [TComplex, TReal, TInt], a,
     bid=T_Complex and a in [TComplex, TReal, TInt], b,

     aid in [T_Real, T_Int] and b=TComplex, T_Complex(a),
     bid in [T_Real, T_Int] and a=TComplex, T_Complex(b),

     aid=T_Complex and bid=T_Complex, T_Complex(UnifyPair(a.params[1], b.params[1])),
     aid=T_Complex and bid in [TFixedPt, T_Real, T_UInt, T_Int], 
         T_Complex(UnifyPair(a.params[1], b)),
     bid=T_Complex and aid in [TFixedPt, T_Real, T_UInt, T_Int], 
         T_Complex(UnifyPair(a, b.params[1])),

     a=TComplex and bid = TFixedPt, T_Complex(b), 
     b=TComplex and aid = TFixedPt, T_Complex(a), 

     aid=T_Real and bid=T_Real, T_Real(Maximum(a.params[1], b.params[1])),

     aid in [T_Int, T_UInt] and b in [TInt, TUInt], a,
     a in [TInt, TUInt] and bid in [T_Int, T_UInt], b,

     aid=T_UInt and bid=T_UInt, T_UInt(Maximum(a.params[1], b.params[1])),
     aid in [T_Int, T_UInt] and bid in [T_Int, T_UInt], T_Int(Maximum(a.params[1], b.params[1])),
     

     aid=TFixedPt, a,
     bid=TFixedPt, b,

     a=TDummy or b=TDummy, TDummy,

     IsArrayT(a) and IsArrayT(b), Checked(a.size=b.size, TArray(UnifyPair(a.t, b.t), a.size)),
     IsArrayT(a), TArray(UnifyPair(a.t, b), a.size),
     IsArrayT(b), TArray(UnifyPair(a, b.t), b.size),

     IsBound(a.unifyWith), a.unifyWith(b),
     IsBound(b.unifyWith), b.unifyWith(a),

     #MRT handles addition/subtraction of pointers and ints.
     aid = TPtr and b = TInt, b,
     a = TInt and bid = TPtr, a,
     #MRT END

     Error("Can't unify ", a, " and ", b)));

#F UnifyTypes(<types>) - given a list of types returns a
#F   most general type
#F   NOTE: complete this.
#F
UnifyTypes := function(types)
    local t, l, i, a, b;
    l := Length(types);
    
    if   l=0 then return TUnknown; 
    elif l=1 then return types[1]; 
    fi;

    [i, t] := [2, types[1]];
    while i <= l do
        t := UnifyPair(t, types[i]); 
	i := i+1;
    od;
    return t;
end;

UnifyTypesL := function(args)
    local t, l, i, a, b;
    l := Length(args);
    
    if   l=0 then return TUnknown; 
    elif l=1 then return args[1].t; 
    fi;

    [i, t] := [2, args[1].t];
    while i <= l do
        t := UnifyPair(t, args[i].t); 
	i := i+1;
    od;
    return t;
end;

Declare(InferType);

UnifyTypesV := values -> UnifyTypes(List(values, InferType));

InferType := v -> let(gt := BagType(v),
   Cond(
       IsInt(v),
           TInt,
       gt = T_RAT or gt = T_DOUBLE,
           TReal,
       gt = T_CPLX,
           TComplex,
       gt = T_CYC,
           When(Im(v)=0, TReal, TComplex),
       gt = T_CHAR,
           TChar,
       gt = T_BOOL,
           TBool,
       IsString(v),
           TString,
       IsExp(v) or IsValue(v),
           When(IsBound(v.t), v.t, TUnknown),

       gt = T_RANGE, Checked(Length(v) >= 0,
       TArray(TInt, Length(v))),

       IsList(v), Checked(Length(v) >= 0,
       TArray(UnifyTypes(List(v, InferType)), Length(v))),

       Error("Can't infer the type of ", v)));

#NOTE: when 'v' is a list with expressions inside it will create TArray value which is wrong
V := v -> Cond(IsValue(v), v,
               IsExp(v), v.eval(),
               InferType(v).value(v));
