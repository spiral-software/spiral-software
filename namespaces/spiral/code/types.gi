
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
# Typ ::
#
# Value ::
#    t = <type>
#    v = <.>
#
# Types:
#   TReal 
#   TComplex
#   TInt
#   TArray(<type>, <size>)
#   TVect(<type>, <vlen>)
#
# Value(<type>, <.>)
# V(<.>)   infers type automatically
#

Declare(Value, IsExp, IsValue, TArray, TVect, BitVector, T_UInt);

_evInt := v -> Cond(IsInt(v), v, IsList(v), List(v, i->_evInt(i)), v.ev());

#----------------------------------------------------------------------------------------------
# Typ : data types
#----------------------------------------------------------------------------------------------
Class(TypOps, rec(
    Print := x-> When(IsBound(x.print), x.print(), Print(x.__name__)),
    \= := RewritableObjectOps.\=,
    \< := RewritableObjectOps.\<
));
Class(TypOpsNoPrint, ClassOps, rec(
    \= := RewritableObjectOps.\=,
    \< := RewritableObjectOps.\<
));


Declare(RangeT);

Class(RangeTOps, PrintOps, rec(
    \= := (v1,v2) -> When( ObjId(v1)<>RangeT or ObjId(v2)<>RangeT, false,
        v1.max=v2.max and v1.min=v2.min and v1.eps=v2.eps),
    \< := (v1,v2) -> Error("Operation '<' is undefined for RangeT."),
    \+ := (v1,v2) -> When( ObjId(v1)<>RangeT or ObjId(v2)<>RangeT, Error("'+' is defined for RangeT only"),
        RangeT(Min2(v1.min, v2.min), Max2(v1.max, v2.max), Max2(v1.eps, v2.eps))),
    \* := (v1,v2) -> When( ObjId(v1)<>RangeT or ObjId(v2)<>RangeT, Error("'*' is defined for RangeT only"),
        RangeT(Max2(v1.min, v2.min), Min2(v1.max, v2.max), Max2(v1.eps, v2.eps))),
));

#F RangeT(<min>, <max>, <eps>): data type range
#F  <min> smallest value, <max> largest value, <eps> unit roundoff
Class(RangeT, rec(
    __call__ := (self, min, max, eps) >> 
        WithBases(self, rec( min := min, max := max, eps := eps, operations := RangeTOps)),
    print    := self >> Print(self.__name__, "(", self.min, ", ", self.max, ", ", self.eps, ")"),
));

Class(Typ, rec(
    operations := TypOps,
    isType := true,
    isSigned := self >> true,
    doHashValues := false,
    check := v -> v,
    #normalize := (self, v) >> Value(self,v),
    eval := self >> self,

    vbase := rec(),

    value := meth(self, v)
        local ev;
        if IsExp(v) then
            ev := v.eval();
            if IsSymbolic(ev) and not IsValue(ev) then
                v.t := self;
                return v;
            fi;
        fi;

        if IsValue(v) then return Value.new(self, self.check(v.v));
        else return Value.new(self,self.check(v));
        fi;
    end,

    realType := self >> self,

    product  := (v1, v2) -> v1 * v2,
    sum      := (v1, v2) -> v1 + v2,
    base_t   := self >> self, # composite types should return base data type (without recursion).
    saturate := abstract(), # (self, v) >> ...
    # range should return RangeT
    range    := abstract(), # (self) >> ...
));

Class(CompositeTyp, Typ, rec(operations := TypOpsNoPrint));

Class(AtomicTyp, Typ, rec(
    doHashValues := true,
    isAtomic := true,
    rChildren := self >> [],
    from_rChildren := (self, rch) >> Checked(rch=[], self),
    free := self >> Union(List(self.rChildren(), FreeVars)),
    vtype := (self,v) >> TVect(self, v),
    csize := self >> sizeof(self)
));

IsType := x -> IsRec(x) and IsBound(x.isType) and x.isType;

Class(TFunc, RewritableObject, Typ, rec(
    check := v -> v, #Checked(IsFunction(v), v),
    product := (v1, v2) -> Error("Can not multiply functions"),
    sum  := (v1, v2) -> Error("Can not add functions"),
    zero := (v1, v2) -> Error("TFunc.zero() is not supported"),
    one  := (v1, v2) -> Error("TFunc.one() is not supported"),
    free := self >> Union(List(self.params, FreeVars)),
    updateParams := self >> Checked(ForAll(self.params, e->IsType(e) or IsValue(e) or IsInt(e) or IsSymbolic(e)), true),
    csize := self >> sizeof(self)
));

IsFuncT := x -> IsType(x) and ObjId(x)=TFunc;

Class(ValueOps, PrintOps, rec(
    \= := (v1,v2) -> Cond(
        not IsValue(v2), v1.v=v2,
        not IsValue(v1), v1=v2.v,
        IsBound(v1.t.vequals), v1.t.vequals(v1.v, v2.v),
        IsBound(v2.t.vequals), v2.t.vequals(v1.v, v2.v),
        v1.v = v2.v),
    \< := (v1,v2) -> Cond(
        not IsValue(v2), When(IsRec(v2), ObjId(v1) < ObjId(v2), v1.v < v2),
        not IsValue(v1), When(IsRec(v1), ObjId(v1) < ObjId(v2), v1   < v2.v),
        v1.v < v2.v)
 ));

#----------------------------------------------------------------------------------------------
# Value : values or constants
# NB: All values are automatically hashed in GlobalConstantHash
#     This can reduce memory footprint, since lots of values are repetitive,
#     like 1s and 0s, and also float values that are too close to each other will
#     hash to same value (by virtue of ValueOps.\=), which will prevent compiler
#     from putting them in separate registers, and thus degrading performance.
#
#     This has negligible effect on accuracy, as long as <type>.vequals is valid.
#     
#----------------------------------------------------------------------------------------------
Class(Value, rec(
    isValue := true,
    __call__ := (self, t, v) >> t.value(v),

    new := (self,t,v) >> # HashedValue(GlobalConstantHash,  <-- this hashes the Value upon construction, disabled now
                         # due to slowness with large data() blocks, which aren't hashed, unless this option is used
	Cond(t.vbase=rec(),
            WithBases(self, rec(t:=t, v:=v, operations := ValueOps)),
            WithBases(self, Inherit(t.vbase, rec(t:=t, v:=v, operations:=ValueOps)))
	),
    #),

    ev := self >> self.v,
    eval := self >> self,
    free := self >> Set([]),

    from_rChildren := (self, rch) >> self,
#   print := self >> Print(self.__name__, "(", self.t, ",", self.v, ")"),
#   print := self >> Print(self.v),
    print := self >> Cond(IsString(self.v), Print("V(\"",self.v, "\")"), Print("V(", self.v, ")")),

    dims := self >> Cond(
	IsArrayT(self.t), self.t.dims(),
	Error("<self>.dims() is only valid when self.t is a TArray"))
));

IsValue := x -> IsRec(x) and IsBound(x.isValue) and x.isValue; 

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

Declare(TComplex);

Class(TUnknown, AtomicTyp, rec( one := self >> 1, zero := self >> 0));
Class(TVoid,    AtomicTyp);
Class(TDummy,   AtomicTyp); # used in autolib for Lambda parameters that are ignored

Class(TReal, AtomicTyp, rec(
    cutoff := 1e-15,
    hash := (val, size) -> 1 + (DoubleRep64(Double(val)) mod size), #(IntDouble(1.0*val*size) mod size),

    check := (self,v) >> Cond(
        IsExp(v),     ReComplex(Complex(code.EvalScalar(v))),
        IsInt(v),     Double(v),
        IsRat(v),     v,
        IsDouble(v),  When(AbsFloat(v) < self.cutoff, 0.0, v),
        IsCyc(v),     ReComplex(Complex(v)),
        IsComplex(v), ReComplex(v),
        Error("<v> must be a double or an expression")),

    vequals := (self, v1,v2) >> When(
        (IsDouble(v1) or IsInt(v1) or IsRat(v1)) and (IsDouble(v2) or IsInt(v2) or IsRat(v2)),
        AbsFloat(Double(v1)-Double(v2)) < self.cutoff,
        false),

    zero := self >> self.value(0.0),
    one := self >> self.value(1.0),
    strId := self >> "f",
    
    complexType := self >> TComplex,
));


TDouble:=TReal;

#
# format:  | sign | integer bits | frac bits |
# # make sure we have space at least for the sign bit
#
_fpdouble := (val,b,fb) -> let(res := IntDouble(val * 2.0^fb),
    Cond(
	val = 1 and (fb = b-1), # we can represent 1 as 0.999999, if we use frac bits only
	    2^fb - 1,
	val = -1 and (fb = b-1), # we can represent 1 as 0.999999, if we use frac bits only
	    -(2^fb - 1),
	Log2Int(res)+2 > b, Error("Overflow, value=", val, ", signed width=",
                                   Log2Int(res)+2, ", max width=", b),
         res));

# format:  | integer bits | frac bits |
#
_ufpdouble := (val,b,fb) -> let(res := IntDouble(val * 2.0^fb),
    When(Log2Int(res)+1 > b, Error("Overflow, value=", val, ", unsigned width=",
                                   Log2Int(res)+1, ", max width=", b),
         res));

#F TFixedPt(<bits>, <fracbits>)   -- fixed point data type
#F
#F   <bits> -- total # of bits (including sign bit)
#F   <fracbits> -- number of fractional bits
#F
#F   Number of integer bits is assumed to be bits-1-fracbits (1 = sign bit)
#F
Class(TFixedPt, TReal, rec(
    operations := TypOpsNoPrint, # NOTE: do not inherit from TReal, and then this line won't be needed

    __call__ := (self, bits, fracbits) >> WithBases(self, rec(
            bits := bits,
            fracbits := fracbits,
            operations := TypOps)),

    rChildren := self >> [self.bits, self.fracbits],
    rSetChild := rSetChildFields("bits", "fracbits"),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),

    print := self >> Print(self.__name__, "(", self.bits, ", ", self.fracbits, ")"),

    check := (self, v) >> _fpdouble(v, self.bits, self.fracbits)
));

# TUFixedPt(<bits>, <fracbit>)  -- unsigned fixed point data type
#
Class(TUFixedPt, TFixedPt);

Class(TComplex, AtomicTyp, rec(
    hash := (val, size) -> let(
        cplx := Complex(val), 
        #h := IntDouble(size * (ReComplex(cplx)+ImComplex(cplx))),
        h := DoubleRep64(ReComplex(cplx)) + DoubleRep64(ImComplex(cplx)),
        1 + (h mod size)),

    check := v -> Cond(
        BagType(v) in [T_CYC, T_RAT, T_INT], v, # exact representation
        IsDouble(v),  When(AbsFloat(v) < TReal.cutoff, 0, v),
        IsComplex(v), Complex(TReal.check(ReComplex(v)), TReal.check(ImComplex(v))),
        IsExp(v),     Complex(v.ev())),

    realType    := self >> TReal,
    complexType := self >> self,

    zero := self >> self.value(0.0),
    one := self >> self.value(1.0),
));

Class(TBool, AtomicTyp, rec(
    hash := (val, size) -> 1 + (InternalHash(val) mod size),
    check := v -> Cond(IsBool(v), v, Error("<v> must be a boolean")),
    one := self >> self.value(true),
    zero := self >> self.value(false),
));

Class(TInt_Base, AtomicTyp, rec(
    hash    := (val, size) -> 1 + (10047871*val mod size),
    bits    := 32,
    check   := v -> Cond(IsExp(v), Int(v.ev()),
                         IsInt(v), v,
                         IsDouble(v) and IsInt(IntDouble(v)), IntDouble(v),
                         Error("<v> must be an integer or an expression")),
    one  := self >> self.value(1),
    zero := self >> self.value(0),

    complexType := self >> TComplex,
));

Class(TInt, TInt_Base, rec(strId := self >> "i"));
Class(TUInt, TInt_Base, rec(isSigned := False, strId := self >> "ui"));
Class(TULongLong, TInt_Base);

IsChar := (x)->When(BagType(x)=T_CHAR, true, false);

Class(TChar, TInt_Base, rec(
    hash := (val, size) -> When(IsChar(val), 1 + (InternalHash(val) mod size), TInt_Base.hash(val, size)),
    bits := 8,
    check := v -> Cond(IsExp(v), Int(v.ev()),
                       IsInt(v), v, 
                       IsChar(v), v,  
                       Error("<v> must be an integer or an expression")),
));

Class(TUChar, TInt_Base, rec(
    bits := 8,
    isSigned := self >> false,
    check := v -> Cond(IsExp(v), Int(v.ev()),
                       IsInt(v), v,
                       Error("<v> must be an integer or an expression")),
));

Class(TString, AtomicTyp, rec(
    doHashValues := true,
    hash := (val, size) -> 1 + (InternalHash(val) mod size),
    check := v -> Cond(IsString(v), v, Error("<v> must be a string")),
    one := self >> Error("TString.one() is not allowed"),
    zero := self >> Error("TString.zero() is not allowed"),
));

Class(TList, CompositeTyp, rec(
    isListT := true,
    hash := (val, size) -> Error("Not implemented"),
    __call__ := (self, t) >>
        WithBases(self, rec(
        t    := Checked(IsType(t), t),
        operations := PrintOps)),
    print := self >> Print(self.__name__, "(", self.t, ")"),
    check := v -> Cond(IsList(v), v, Error("<v> must be a list")),

    one  := self >> Error("TList.one() is not allowed"),
    zero := self >> Error("TList.zero() is not allowed"),

    rChildren := self >> [self.t],
    rSetChild := rSetChildFields("t"),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),

));

Class(TSym, CompositeTyp, rec(
    hash := (val, size) -> Error("Not implemented"),
    check := v -> v,
    __call__ := (self, id) >>
        WithBases(self, rec(
        id    := Checked(IsString(id), id),
        operations := TypOps)),

    rChildren := self >> [self.id],
    rSetChild := rSetChildFields("id"),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),

    print := self >> Print(self.__name__, "(\"", self.id, "\")"),
    csize := self >> sizeof(self)
));

#F TArrayBase -- base class for array-like element collection types
#F
#F Subclasses: TPtr, TArray, TVect, BitVector
#F
#F Default constructor:
#F
#F  __call__(<element-type>, <size>) - array type of <size> elements of <element-type>
#F
Class(TArrayBase, CompositeTyp, rec(
     __call__ := (self, t, size) >>
        WithBases(self, rec(
        t    := Checked(IsType(t), t),
        size := Checked(IsPosInt0Sym(size), size),
        operations := TypOps)),

    hash := (self, val, size) >> (Sum(val, x -> x.t.hash(x.v, size)) mod size) + 1,

    rChildren := self >> [self.t, self.size],
    rSetChild := rSetChildFields("t", "size"),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),

    isSigned := self >> self.t.isSigned(),

    print := self >> Print(self.__name__, "(", self.t, ", ", self.size, ")"),

    check := (self, v) >> Checked(IsList(v), Length(v) = self.size,
    ForAll(v, el -> el.t = self.t), v),

    # these fields go into values
    vbase := rec(
        free := self >> Union(List(self.v, e -> e.free())),
        rChildren := self >> self.v,
        rSetChild := meth(self, n, newC) self.v[n] := newC; end
    ),

    zero := self >> self.value(Replicate(_unwrap(self.size), self.t.zero())),
    one  := self >> self.value(Replicate(_unwrap(self.size), self.t.one())),

    value := (self, v) >> let(vv := When(IsValue(v), v.v, v),
        Cond(IsExp(vv), vv,
             Checked(IsList(vv),
                     Value.new(self, List(vv, e->self.t.value(e)))))),

    # array type can have free variables in .size field
    free := self >> Union(FreeVars(self.size), FreeVars(self.t)),

    csize := self >> self.t.csize() * self.size,

    realType := self >> ObjId(self)(self.t.realType(), self.size),

    base_t := self >> self.t,
    range  := self >> self.t.range(),
));

Declare(TPtr, TArray);

#F TArray(<element-type>, <size>) - array type of <size> elements of <element-type>
#F
Class(TArray, TArrayBase, rec(
    isArrayT := true,
    vtype := (self, v) >> TArray(self.t.vtype(v), self.size/v),
    toPtrType := self >> TPtr(self.t),
    doHashValues := true,

    dims := self >> Cond(
        ObjId(self.t)=TArray, [self.size] :: self.t.dims(),
        [self.size])
));

# [ptrAligned, ptrUnaligned] are TPtr.alignment values
ptrUnaligned := [1,0];
ptrAligned4  := [4,0];
ptrAligned8  := [8,0];
ptrAligned16 := [16,0];
ptrAligned := ptrAligned16;

# NOTE: this is a hack, esp because 16 byte boundary is hardcoded in ptrAligned
#        It should be in SpiralDefaults somehow
TArray.alignment := ptrAligned; 
TArray.qualifiers := [];

TypeDomain := (dom, els) ->
    Cond(Same(dom, Rationals) or Same(dom, Scalars) or Same(dom, Doubles), TReal,
     Same(dom, Complexes), TComplex,
     Same(dom, Cyclotomics), When(ForAll(els, x->Im(x)=0), TReal, TComplex),
     Same(dom, Integers), TInt,
     Error("Unrecognized domain <dom>"));

# IsArrayT(<t>) - checks whether <t> is an array type object
IsArrayT := x -> IsType(x) and IsBound(x.isArrayT) and x.isArrayT;

# IsListT(<t>) - checks whether <t> is a list type object
IsListT := x -> IsType(x) and IsBound(x.isListT) and x.isListT;

# IsVecT(<t>) - checks whether <t> is an vector type object
IsVecT := x -> IsType(x) and IsBound(x.isVecT) and x.isVecT;

# IsPtrT(<t>) - checks whether <t> is a pointer type object
IsPtrT := x-> IsType(x) and IsBound(x.isPtrT) and x.isPtrT;

# IsUnalignedPtrT(<t>) - checks whether <t> is a unaligned pointer type object,
#       unaligned means aligned with smaller granularity than child type (t.t)
#       size.
IsUnalignedPtrT := x -> IsPtrT(x) and x.alignment<>ptrAligned;


# obsolete, use IsArrayT
IsArray := IsArrayT;

Class(TPtr, TArrayBase, rec(
     isPtrT := true,
     __call__ := arg >> let(self := arg[1],
         t          := arg[2],
         qualifiers := When(IsBound(arg[3]), arg[3], []),
         alignment  := When(IsBound(arg[4]), arg[4], ptrAligned16),
         WithBases(self, rec(
            t    := Checked(IsType(t), t),
            size := 0,
            qualifiers := qualifiers,
            _restrict  := false,
            alignment  := alignment,
            operations := TypOps)).normalizeAlignment()),

     # value := (self, v) >> Error("Values of TPtr type are not allowed"),
     value := Typ.value,

     check := (self, v) >> Cond(
        IsList(v), Checked(
           Length(v) = self.size,
           ForAll(v, el -> el.t = self.t), 
           v
        ),
        IsInt(v), v,
        Error("TPtr needs to either point to an array or some value")
     ),

     # this looks crazy, but sometimes this happens (in LRB backend actually) : X - X
     # where X is a pointer. Internally this can become X + (-X), and then becomes 0
     isSigned := self >> true,

     rChildren := self >> [self.t, self.qualifiers, self.alignment],
     rSetChild := rSetChildFields("t", "qualifiers", "alignment"),

     zero := self >> TInt.zero(),
     one := self >> TInt.one(),

     print := self >> Print(self.__name__, "(", self.t,
         When(self.qualifiers<>[], Print(", ", self.qualifiers)), ")",
         When(self._restrict, ".restrict()", ""),
         ".aligned(", self.alignment, ")"
         ),

     restrict := (self) >> CopyFields(self, rec(_restrict := true)),
     unRestricted := (self) >> CopyFields(self, rec(_restrict := false)),

     csize := self >> sizeof(self),

     realType := self >> Cond(self._restrict,
         ObjId(self)(self.t.realType(), self.qualifiers).restrict(),
         ObjId(self)(self.t.realType(), self.qualifiers)
     ),

     aligned   := (self, a) >> CopyFields(self, rec( alignment := Checked(IsList(a) and Length(a)=2, a)  )).normalizeAlignment(),
     unaligned := (self) >> CopyFields(self, rec( alignment := [1,0] )),

     normalizeAlignment := meth(self)
         if IsValue(self.alignment[2]) then self.alignment[2] := self.alignment[2].v;
         elif IsSymbolic(self.alignment[2]) then self.alignment := ptrUnaligned; # NOTE: Conservative assumption
         fi;
         Constraint(IsInt(self.alignment[2]));
         self.alignment[2] := self.alignment[2] mod self.alignment[1];
         return self;
     end,

     withAlignment := (self, value) >> CopyFields(self, rec( 
         alignment := When(IsPtrT(value), value.alignment, value))),

     # things get a little strange here because we allow pointers
     # to be set to some int based offset 
     # 
     vbase := rec(
         free := self >> Cond(
             IsList(self.v), Union(List(self.v, e -> e.free())),
             IsInt(self.v), [],
             Error("hmm.")
         ),
         rChildren := self >> Cond(
             IsList(self.v), self.v,
             IsInt(self.v), [], 
             Error("hmm.")
         ),

         rSetChild := meth(arg)
             local _self;
             _self := arg[1];

             if Length(arg) = 3 then
                _self.v[arg[2]] := arg[3];
             elif Length(arg) = 2 then
                _self.v := arg[2];
             else
                Error("choke");
             fi;    
         end,
     ),
));


# -- TVect ----------------------------------------------------------------------

Class(TVect, TArrayBase, rec(
    isVecT := true,
    doHashValues := true,
    __call__ := (self, t, size) >> Cond(t=T_UInt(1), BitVector(size), Inherited(t, size)),

    product := (v1, v2) -> Checked(IsList(v1) or IsList(v2), let(
        vv1 := When(not IsList(v1), Replicate(Length(v2), v1), v1),
        vv2 := When(not IsList(v2), Replicate(Length(v1), v2), v2),
        l := Length(vv1),
        Checked(l = Length(vv2),
            List([1..l], i -> vv1[i]*vv2[i])))),

    sum := (v1, v2) -> Checked(IsList(v1) or IsList(v2), let(
        vv1 := When(not IsList(v1), Replicate(Length(v2), v1), v1),
        vv2 := When(not IsList(v2), Replicate(Length(v1), v2), v2),
        l := Length(vv1),
        Checked(l = Length(vv2),
            List([1..l], i -> vv1[i]+vv2[i])))),

    value := (self, v) >> Cond( IsValue(v) and self=v.t, v, let(
        vv := When(IsValue(v), v.v, v),
        Cond(IsExp(vv), vv,
             IsList(vv), Value.new(self, List(vv,                       e -> self.t.value(e))),
             <#else#>    
                         Value.new(self, List(Replicate(self.size, vv), e -> self.t.value(e)))))),

    saturate := (self, v) >> let( vv := _unwrap(v), Cond(not IsList(vv) or Length(vv)<>self.size, v,
        Value.new(self, List(vv, e -> self.t.saturate(e)))) ),
  
    toUnsigned := self >> TVect(self.t.toUnsigned(), self.size),
    toSigned   := self >> TVect(self.t.toSigned(),   self.size),
    double     := self >> TVect(self.t.double(),     self.size/2),

));

IsTVectDouble := x -> IsVecT(x) and x.t = TReal;

TVectDouble := vlen -> TVect(TReal, vlen);

#Class(T_Type, Typ, rec(
#     __call__ := (self, bits) >>
#        WithBases(self, rec(
#        bits := Checked(IsPosInt(bits), bits),
#        operations := TypOps)),
#
#     hash := (self, val, size) >>  1 + (10047871*val mod size),
#
#     rChildren := self >> [],
#     rSetChild := self >> Error("This function should not be called"),
#     print := self >> Print(self.__name__, "(", self.bits, ")"),
#     free := self >> Union(List(self.rChildren(), FreeVars)),
#     vtype := (self,v) >> TVect(self, v)
#));

Class(T_Type, RewritableObject, rec(
    isType := true,

    isSigned := self >> true,
    realType := self >> self,

    doHashValues := false, 
    check := v -> v,
    vbase := rec(),

    value := meth(self, v)
        local ev;
        if IsExp(v) then
            ev := v.eval();
            if IsSymbolic(ev) and not IsValue(ev) then
                v.t := self;
                return v;
            fi;
        fi;
        if IsValue(v) then
            return Value.new(self, self.check(v.v));
        else
            return Value.new(self, self.check(v));
        fi;
    end,

    eval     := self >> self,
    product  := (v1, v2) -> v1 * v2,
    sum      := (v1, v2) -> v1 + v2,
    zero     := self >> self.value(0),
    one      := self >> self.value(1),
    csize    := self >> sizeof(self),
    base_t   := self >> self, # composite types should return base type (without recursion).
    saturate := abstract(), # (self, v) >> ...
    range    := abstract(), # (self) >> ...
));                           

Declare(T_Int, T_UInt, T_Complex);

Class(T_Ord, T_Type, rec(
    hash := (val, size) -> 1 + (10047871*val mod size),
    saturate := (self, v) >> let( b := self.range(),
        Cond( IsExp(v), v, self.value(Max2(b.min, Min2(b.max, _unwrap(v)))))),
));

Class(T_Int, T_Ord, rec(
    check := (self, v) >> let(
        i := Cond(IsDouble(v), IntDouble(v),
                  IsRat(v), Int(v),
                  Error("Can't convert <v> to an integer")),
        b := self.params[1],
        ((i + 2^(b-1)) mod 2^b) - 2^(b-1)),

    strId    := self >> "i"::StringInt(self.params[1]),
    range    := self >> RangeT(-2^(self.params[1]-1),  2^(self.params[1]-1)-1,  1),
    
    isSigned   := True,
    toUnsigned := self >> T_UInt(self.params[1]),
    toSigned   := self >> self,
    double     := self >> T_Int(2*self.params[1]),
));

Class(T_UInt, T_Ord, rec(
    check := (self, v) >> let(
        i := Cond(IsDouble(v), IntDouble(v),
                  IsRat(v), Int(v),
                  Error("Can't convert <v> to an integer")),
        b := self.params[1],
        i mod 2^b),

    strId    := self >> "ui"::StringInt(self.params[1]),
    range    := self >> RangeT(0, 2^self.params[1]-1, 1),

    isSigned   := False,
    toUnsigned := self >> self,
    toSigned   := self >> T_Int(self.params[1]),
    double     := self >> T_UInt(2*self.params[1]),
));

Class(BitVector, TArrayBase, rec(
    isVecT := true,
    __call__ := (self, size) >> Inherited(T_UInt(1), size),

    print := self >> Print(self.__name__, "(", self.size, ")"),
    vbase := rec(
        print := self >> let(n:=Length(self.v), Print("h'", HexStringInt(Sum([1..n], i->self.v[i] * 2^(n-i))))), 
    ),
    
    isSigned := self >> false,

    rChildren := self >> [self.size],
    rSetChild := rSetChildFields("size"),

    one := self >> self.value(Replicate(self.size, 1)),
    zero := self >> self.value(Replicate(self.size, 0)),

    hash := (self, val, size) >> let(n:=Length(val),
        1 + (Sum([1..n], i -> val[i] * 2^(n-i)) mod size)),
 
    product := TVect.product,
    sum := TVect.sum,

    _uint1 := T_UInt(1),

    value := (self, v) >> When( IsValue(v) and v.t = self, v,
        let(vv := When(IsValue(v), v.v, v),
            Cond(IsExp(vv), vv,
                 Checked(IsList(vv),
                     Value.new(self, List(vv, e->self._uint1.check(e))))))),
));

Class(T_Real, T_Type, rec(
   #correct cutoffs are floor(log10(2^(mantissa bits + 1)))
   cutoff := self>>Cond(
       self.params[1] = 128, 1e-34,
       self.params[1] = 80, 1e-19,
       self.params[1] = 64, 1e-15,
       self.params[1] = 32, 1e-7,
       Error("cutoff not supported")
   ),

   hash := TReal.hash, 
   
   check := (self,v) >> let( r := Cond(
            IsExp(v),     ReComplex(Complex(code.EvalScalar(v))),
            IsInt(v),     Double(v),
            IsRat(v),     v,
            IsDouble(v),  v,
            IsCyc(v),     ReComplex(Complex(v)),
            IsComplex(v), ReComplex(v),
            # else
                Error("<v> must be a double or an expression")),
        When(AbsFloat(r) < self.cutoff(), 0.0, r)),

   vequals := (self, v1,v2) >> When(
        (IsDouble(v1) or IsInt(v1)) and (IsDouble(v2) or IsInt(v2)),
        AbsFloat(Double(v1)-Double(v2)) < self.cutoff(),
        false),

   zero := self >> self.value(0.0),
   one := self >> self.value(1.0),
   isSigned := (self) >> true,
   strId := self >> "f"::StringInt(self.params[1]),
   range := self >> Cond( 
       self.params[1] = 128, RangeT(
           -1.7976931348623157e+308 - 10e291, #INF
           1.7976931348623157e+308 + 10e291, #INF
           1e-34
       ),
       self.params[1] = 80, RangeT(
           -1.7976931348623157e+308 - 10e291, #INF
           1.7976931348623157e+308 + 10e291, #INF
           1e-19
       ),
       self.params[1] = 64, RangeT(
           -1.7976931348623157e+308,
            1.7976931348623157e+308,
            1.1102230246251565e-016
       ),
       self.params[1] = 32, RangeT(
           -3.4028234e+038,
            3.4028234e+038,
            5.96046448e-008
       )),
   
   complexType := self >> T_Complex(self),
));

Class(T_Complex, T_Type, rec(
    hash := TComplex.hash,
    realType    := self >> self.params[1],
    complexType := self >> self,
    
    isSigned := self >> self.params[1].isSigned(),
    strId    := self >> "c"::self.params[1].strId(),

    check := (self, v) >> let(
	realt := self.params[1],
	cpx := Complex(v),
	Complex(realt.check(ReComplex(cpx)),
	        realt.check(ImComplex(cpx))))
));

# # complex type is made up of TWO T_Real, T_Uint, or T_Int types.

# Class(T_Complex, TArrayBase, rec(
#     isComplex := true,
#     __call__ := (arg) >> let(
#         self := arg[1],
#         t := arg[2],
#         Checked(
#             ObjId(t) in [T_Real, T_UInt, T_Int],
#             WithBases(self, rec(
#                 t := t,
#                 qualifiers := When(Length(arg) > 2, arg[3], []),
#                 operations := TypOps,
#                 size := 0
#             ))
#         )
#     ),

#     rChildren := self >> [self.t, self.qualifiers],
#     rSetChild := rSetChildFields("t", "qualifiers"),

#     print := self >> Print(self.__name__, "(", self.t,
#         When(self.qualifiers <> [],
#             Print(", ", self.qualifiers)
#         ),
#         ")"
#     )
# ));

_IsVar := (e) -> code.IsVar(e);

#F T_Struct: structure type.
#F
#F T_Struct("structname", [<var1>, <var2>, ... , <varN>])
#F
Class(T_Struct, T_Type, rec(
    updateParams := meth(self)
        Constraint(IsString(self.params[1]));
        Constraint(IsList(self.params[2]));
        Constraint(ForAll(self.params[2], e -> _IsVar(e)));
    end,

    getName := self >> self.params[1],
    getVars := self >> self.params[2]
));

IsIntT := (t) -> IsType(t) and t in [TChar, TInt] or ObjId(t) = T_Int;
IsUIntT := (t) -> IsType(t) and t in [TUChar, TUInt] or ObjId(t) = T_UInt;
IsOrdT := (t) -> IsIntT(t) or IsUIntT(t);

IsFixedPtT := (t) -> IsType(t) and ObjId(t)=TFixedPt; 

IsRealT := (t) -> IsType(t) and t=TReal or ObjId(t)=T_Real;

IsComplexT := (t) -> IsType(t) and t=TComplex or ObjId(t)=T_Complex;

IsOddInt :=  n -> When(IsValue(n), n.v mod 2 = 1, IsInt(n) and n mod 2 = 1);

IsEvenInt := n -> When(IsValue(n), n.v mod 2 =0, IsInt(n) and n mod 2 = 0);
