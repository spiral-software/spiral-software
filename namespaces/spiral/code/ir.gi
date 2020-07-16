# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details


Declare(apack, nth, var, toExpArg, ExpOps, VarOps, NthOps, ExprFuncs, AnySyms, errExp, funcExp);

Class(Symbolic, rec(
    isSymbolic := true,
    visitAs := "Symbolic",

    setType := meth(self)
        self.t := self.computeType();
	return self;
    end,

    dims := self >> Cond(
	IsArrayT(self.t), self.t.dims(),
	Error("<self>.dims() is only valid when self.t is a TArray"))

    # must be implemented in subclasses
    #computeType := self >> ..type of self..
));

IsSymbolic := o -> IsRec(o) and IsBound(o.isSymbolic) and o.isSymbolic;
IsExpArg := o -> IsSymbolic(o) or IsValue(o);
IsLoc := x -> IsRec(x) and IsBound(x.isLoc) and x.isLoc;
IsNth := x -> IsRec(x) and IsBound(x.__bases__) and x.__bases__[1] = nth;
IsVar := x -> IsRec(x) and IsBound(x.__bases__) and x.__bases__[1] = var;
IsExp := x -> IsRec(x) and IsBound(x.isExp) and x.isExp;

toRange := rng -> Cond(
    rng = [], 0,
    IsRange(rng), Checked(rng[1]=0, Last(rng)+1),
    IsInt(rng), rng,
    IsValue(rng), rng.v,
    IsSymbolic(rng), rng,
    Error("<rng> must be a range, an integer, or a symbolic expression"));

listRange := rng -> Cond(
    IsRange(rng), Checked(rng[1]=0, rng),
    IsInt(rng), [0..rng-1],
    Error("<rng> must be a range or an integer"));

# _ListElmOp: executes operation on evaluated list elements
Declare(_ListElmOp);
_ListElmOp := (a, b, op) ->
    Cond( IsList(a) and IsList(b) and not IsString(a) and not IsString(b),
              Checked(Length(a)=Length(b), List([1..Length(a)], i -> _ListElmOp(a[i], b[i], op))),
          IsRec(a) and IsBound(a.ev),
              _ListElmOp(a.ev(), b, op),
          IsRec(b) and IsBound(b.ev),
              _ListElmOp(a, b.ev(), op),
          IsList(a) and not IsString(a),
              List(a, e -> _ListElmOp(e, b, op)),
          IsList(b) and not IsString(b),
              List(a, e -> _ListElmOp(e, b, op)),
          op(a, b) );

Class(Loc, Symbolic, rec(
    isLoc := true,
    isExp := true,
    free := self >> Set(ConcatList(self.rChildren(), FreeVars)),
    print := self >> Print(self.__name__, "(", PrintCS(self.rChildren()), ")"),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),
));

#F nth(<loc>, <idx>) -- symbolic representation of array access
#F
Class(nth, Loc, rec(
    __call__ := (self, loc, idx) >> WithBases(self,
        rec(operations := NthOps,
            loc := toExpArg(loc),
            idx := toExpArg(idx))).setType().cfold(),

    can_fold := self >> self.idx _is funcExp or (IsValue(self.idx) and
                  (IsValue(self.loc) or (IsVar(self.loc) and IsBound(self.loc.value)) or self.loc _is apack)),
    cfold := self >> When(self.can_fold(), self.eval(), self),

    rChildren := self >> [self.loc, self.idx],
    rSetChild := rSetChildFields("loc", "idx"),

    ev := self >> let(e := self.eval(),
	Cond(IsBound(e.v), e.v, e)),

    eval := self >> let(loc := self.loc.eval(), idx := self.idx.eval(),
        evself := CopyFields(self, rec(loc := loc, idx := idx)), # Simply return expression in case value cannot be returned (although it appears this wasn't desired originally?)
        Cond(idx _is funcExp,
                 self.t.value(idx.args[1]),
             not IsValue(idx),
                 evself, # self,
             idx.v < 0,
                 errExp(self.t),
             loc _is apack,
                 Cond(idx.v >= Length(loc.args), errExp(self.t), loc.args[idx.v+1]),
             IsValue(loc),
                 Cond(idx.v >= Length(loc.v), errExp(self.t), V(loc.v[idx.v+1])),
             IsVar(loc) and IsBound(loc.value),
                 Cond(idx.v >= Length(loc.value.v), errExp(self.t), V(loc.value.v[idx.v+1])),
             evself)), # self)),

    computeType := self >> Cond(
	IsPtrT(self.loc.t) or IsArrayT(self.loc.t) or IsListT(self.loc.t), self.loc.t.t,
        ObjId(self.loc.t)=TSym, TSym("Containee"), #used with C++ container objects (EnvList)
        self.loc.t = TUnknown,  self.loc.t,
	Error("Unknown types of 1st argument <self.loc> in ", ObjId(self))
    ),

    isExpComposite := true
));

#F deref(<loc>)  -- symbolic representation of pointer dereference, equivalent to nth(<loc>, 0)
#F
Class(deref, nth, rec(
    __call__ := (self, loc) >> Inherited(loc, TInt.value(0)),
    rChildren := self >> [self.loc],
    rSetChild := rSetChildFields("loc"),
));

#F addrof(<loc>) -- symbolic representation of address of <loc>.
#F
#F For a variable 'foo', addrof(foo) is the equivalent of &(foo) in C
#F
Class(addrof, Loc, rec(
    __call__ := (self, loc) >> WithBases(self,
	rec(operations := NthOps, loc := loc, idx := 0)).setType(),

    computeType := self >> TPtr(self.loc.t),

    rChildren := self >> [self.loc],
    rSetChild := rSetChildFields("loc"),
    can_fold := False,
));

#F var(<id>, <t>)
#F var(<id>, <t>, <range>)
#F var.fresh(<id>, <t>, <range>)
#F var.fresh_t(<id>, <t>)
#F
#F Create symbolic variables. variables are kept in a global hash, and thus
#F two variables with same name will refer to same physical object.
#F Namely
#F     Same(var("zz", TInt), var("zz", TInt)) == true
#F Moreover,
#F   spiral> v1 := var("zz", TInt);;
#F   spiral> v2 := var("zz", TReal);;
#F   spiral> v1.t;
#F       TReal;
#F   spiral> v2.t;
#F       TReal;
#F
Class(var, Loc, rec(
    rChildren := self >> [],
    from_rChildren := (self, rch) >> self,
    free := self >> Set([self]),
    equals := (self,o) >> Same(self,o),

    setAttr := meth(self, attr) self.(attr) := true; return self; end,
    setAttrTo := meth(self, attr, val) self.(attr) := val; return self; end,

    computeType := self >> self.t,

    __call__ := meth(arg)
        local self, id, range, t, v;
        self := arg[1];
        id := arg[2];

        if Length(arg) >= 3 then t := arg[3]; else t := TUnknown; fi;
        if Length(arg) >= 4 then range := arg[4]; else range := false; fi;

        if not IsBound(self.table.(id)) then
            v := WithBases(self, rec(operations := VarOps, id := id, t := t));
            if range <> false then v.range := range; fi;
            self.table.(id) := CantCopy(v);
            v.uid := [BagAddr(v),1];
            return v;
        else
            v := self.table.(id);
            if t<>TUnknown then v.t := t; fi;
            if range <> false then v.range := range; fi;
            #if Length(arg) >= 3 then
            #return Error("Variable '", id, "' is already defined, use var(..).xxx to update fields");
            #fi;
            return v;
        fi;
    end,

    setRange := meth(self, r)
       self.range := r;
       return self;
    end,

    setValue := meth(self, v)
       self.value := v;
       return self;
    end,

    clone := self >> When(
        IsBound(self.range),
        var.fresh(self.id{[1]}, self.t, self.range),
        var.fresh_t(self.id{[1]}, self.t)
    ),

    printFull := self >> Print(
        self.__name__, "(\"", self.id, "\", ", self.t,
        When(
            IsBound(self.range),
            Print(", ", self.range), ""
        ),
        ")"
    ),

#    printShort := self >> Print(self.__name__, "(\"", self.id, "\")"),
    printShort := self >> Print(self.id),

    print := ~.printShort,

    fresh := (self,id,t,range) >> self(self._id(id), t, range),

    fresh_t := (self,id,t) >> Cond(
	IsInt(t) or IsScalar(t),
	    self(self._id(id), TInt, t),
	IsType(t),
	    self(self._id(id), t),
	Error("<t> must be a type or an integer that represents an interval")),

    _id := meth(self, id)
       local cnt, st;

       cnt := When(IsBound(self.counter.(id)), self.counter.(id), 1);
       # Intel compiler (ver 8 and 9)
       # in linux uses variable i386 as a keyword.
       if cnt = 385 then
           self.counter.(id) := cnt+2;
       else
           self.counter.(id) := cnt+1;
       fi;
       st := Concat(id, String(cnt));
#       st := Concat(id, VarNameInt(cnt));
       if IsBound(self.table.(st)) then
       self.counter.(id) := cnt+1000;
       return self._id(id);
       else return st;
       fi;
    end,

    nth := (self, idx) >> nth(self, idx),

    ev := self >> self, #When(IsBound(self.value), self.value.ev(), self),
    eval := self >> self,
    can_fold := False,

    flush := meth(self)
        self.table := WeakRef(tab());
        self.counter := tab();
    end,

    table := WeakRef(tab()),
    counter := tab(),
    has_range := self >> IsInt(self.range)
));

#F ----------------------------------------------------------------------------------------------
#F Exp : expressions
#F ----------------------------------------------------------------------------------------------

Class(Exp, Symbolic, rec(
   isExp := true,
   isExpComposite := true,

   __call__ := arg >> WithBases(arg[1],
       rec(args := List(Drop(arg, 1), toExpArg), operations := ExpOps)).setType(),

   print := self >> Print(self.__name__, "(", PrintCS(self.args), ")"),

   ev := self >> Error("not implemented"),

   eval := meth(self)
       local evargs, res, type;
       evargs := List(self.args, e -> e.eval());

       if evargs <> [] and ForAll(evargs, IsValue) then
           res := ShallowCopy(self);
           res.args := evargs;
           res := res.ev();
           type := self.computeType();
	   return type.value(res);
       else
           res := ApplyFunc(ObjId(self), evargs);
           res.t := self.t; # NOTE: why is this line here?
           return res;
       fi;
   end,

   rChildren := self >> self.args,
   rSetChild := meth(self, n, newChild)
       self.args[n] := newChild;
   end,
   from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),

   free := self >> Set(ConcatList(self.args, FreeVars)),

   can_fold := self >> not IsPtrT(self.t) and
       let(rch := self.rChildren(), rch<>[] and ForAll(rch, c -> IsValue(c) or (IsVar(c) and IsBound(c.value)))),

   cfold := self >> When(self.can_fold(), self.eval(), self),

   setType := meth(self)
        if IsBound(self.computeType) then
	    self.t := self.computeType();
	else
	    self.t := UnifyTypes(List(self.args, x->x.t));
	    PrintErr("Warning: ", ObjId(self), " needs a computeType() method. ",
		     "(default type = ", self.t, ")\n");
	fi;
	return self;
    end
));

Class(AutoFoldExp, Exp, rec(
   __call__ := arg >> ApplyFunc(Inherited, Drop(arg, 1)).cfold()
));

#F AutoFoldRealExp -- scalar or vector expression with floating point result type
Class(AutoFoldRealExp, AutoFoldExp, rec(
    computeType := self >> let(
        t := UnifyTypesL(self.args),
        Cond( IsRealT(t.base_t()), t, Cond(IsVecT(t), TVect(TReal. t.size), TReal)))
));

Class(ListableExp, Exp, rec(
   __call__ := meth(arg)
       local self, res;
       self := arg[1];
       arg := Drop(arg, 1);
       if Length(arg)=2 then
       if IsList(arg[1]) then return List(arg[1], e->self(e, arg[2]));
       elif IsList(arg[2]) then return List(arg[2], e->self(arg[1], e));
       fi;
       fi;
       res := WithBases(self, rec(args := List(arg, toExpArg), operations := ExpOps));
       return res.cfold();
   end
));


Declare(apack);

# TArray expression
Class(apack, AutoFoldExp, rec(
    ev := self >> List(self.args, x->x.ev()),
    computeType := self >> TArray(UnifyTypes(List(self.args, x->x.t)), Length(self.args)),
    can_fold := False, # apack expected to be in nth, let nth to fold first instead of apack

    fromList := (lst, func) -> ApplyFunc(apack, Map(lst, func)),
    fromMat  := (mat, func) -> apack.fromList(mat, r -> apack.fromList(r, func)),
));

#F cxpack(<re>, <im>) -- packs <re> <im> pair into complex number

Class( cxpack, AutoFoldExp, rec(
    ev          := self >> ApplyFunc(Complex, List(self.args, x->x.ev())),
    computeType := self >> UnifyTypesV(self.args).complexType(),
));

#F brackets(<exp>) -- symbolic representation of brackets

Class(brackets, Exp, rec(
   __call__ := arg >> Checked( Length(arg) = 2, ApplyFunc(Inherited, Drop(arg, 1))),
   computeType := self >> self.args[1].t,
   can_fold := self >> Inherited() and Length(self.args)=1,
));

#F fcall(<func>, <arg1>, ...) -- symbolic representation of a function call
#F   <func> could be a variable or a Lambda
#F   Example:
#F     f := var("f", TFunc(TInt, TInt));
#F     fcall(f, 1);
#F     fcall(L(16,4).lambda(), 1);
#F
Class(fcall, Exp, rec(
    __call__ := arg >> let(
        self := arg[1],
	args := List(Drop(arg, 1), toExpArg),
	Cond(Length(args) < 1,
                 Error("fcall must have at least 1 argument: function"),
	     IsLambda(args[1]),
                 ApplyFunc(args[1].at, Drop(args, 1)),
	     #else
	         WithBases(self, rec(args := args, operations := ExpOps)).setType())),

    computeType := self >> let(ft := self.args[1].t, Cond(
        (ft in [TString, TUnknown]) or (ObjId(ft)=TPtr and ObjId(ft.t)=TSym), TUnknown,
        ObjId(ft) = TFunc, Last(ft.params),
        Error("<self.args[1].t> must be TFunc(..) or TUnknown"))),

    eval := self >> ApplyFunc(ObjId(self), List(self.args, e->e.eval())),
    can_fold := False,
));

Class(gapcall, Exp, rec(
    __call__ := meth(arg)
        local res, fname;
        res := WithBases(arg[1], rec(args := List(Drop(arg, 1), toExpArg),
                                     operations := ExpOps,
                                     t := TUnknown));
    if Length(res.args) < 1
        then Error("gapcall must have at least 1 argument: function name"); fi;
    if IsVar(res.args[1]) then
        fname  := res.args[1].id;
        elif IsString(res.args[1]) then
            fname := res.args[1];
        else
            return res;
        fi;

    if IsBound(ExprFuncs.(fname)) then
        return ApplyFunc(ExprFuncs.(fname), Drop(res.args, 1));
    else
        return res;
        fi;
    end,

    ev := self >> ApplyFunc(Eval(DelayedValueOf(self.args[1].id)),
                            List(Drop(self.args,1), x->x.ev())),
    eval := meth(self)
        local evargs;
        evargs := List(Drop(self.args,1), e->e.eval());
        if ForAll(evargs, IsValue) then return V(self.ev());
        else return ApplyFunc(ObjId(self), Concatenation([self.args[1]], evargs));
        fi;
    end
));

ExprDelay := function(d)
   d := FunccallsDelay(d);
   d := DelaySubst(d, e->Global.Type(e) in [T_VAR, T_VARAUTO],
       e -> var(NameOf(e)));
   d := DelaySubst(d, e->Global.Type(e) = T_FUNCCALL,
       e -> ApplyFunc(gapcall, e{[1..Length(e)]}));
   return When(IsExp(d), d, V(d));
end;

toExpArg := x -> Cond(IsRec(x) or IsFunction(x), x,
                      IsDelay(x), ExprDelay(x),
                      V(x));

toAssignTarget := x -> x;


#F ----------------------------------------------------------------------------------------------
#F Expressions: Basic Arithmetic
#F
#F add(<a>, <b>, ...)
Class(add, AutoFoldExp, rec(

    # __sum is overriden in descendant 'adds' (saturated addition)
    __sum := (self, a, b) >> self.t.value(self.t.sum(_stripval(a), _stripval(b))),

#    ev := self >> FoldL(self.args, (acc, e)->self.__sum(acc, e.ev()), self.t.zero()).ev(),
    ev := self >> let(fe := FoldL(self.args, (acc, e)->self.__sum(acc, e.ev()), self.t.zero()), When(self = fe, self, fe.ev()) ),

    # the intricate logic below is for computing the new alignment when dealing
    # with pointer types
    _ptrPlusOfs := (ptr_t, ofs) ->
        TPtr(ptr_t.t, ptr_t.qualifiers, [ptr_t.alignment[1], (ptr_t.alignment[2] + ofs) mod ptr_t.alignment[1]]),

    _addPtrT := function(ptr_args)
        local align, el_t, t;
	if Length(ptr_args)=1 then return ptr_args[1].t; fi;
	align := [ Gcd(List(ptr_args, x->x.t.alignment[1])) ];
	align[2] := Sum(ptr_args, x->x.t.alignment[2]) mod align[1];
	el_t := UnifyTypes(List(ptr_args, x->x.t.t));
	return TPtr(el_t, ConcatList(ptr_args, x->x.t.qualifiers), align);
    end,

    computeType := meth(self)
        local len, t, ptr_args, other_args, sum;
	len := Length(self.args);
	if   len=0  then return TInt;
	elif len=1  then return self.args[1].t;
	else
	    [ptr_args, other_args] := SplitBy(self.args, x->IsPtrT(x.t) or IsArrayT(x.t));
	    if Length(ptr_args)=0 then
		return UnifyTypesL(self.args);
	    elif Length(ptr_args)=1 then
		sum := Sum(other_args);
		if other_args<>[] and not IsIntT(sum.t) then Error("Can't add non-integer to a pointer"); fi;
		return self._ptrPlusOfs(ptr_args[1].t, sum);
	    elif Length(other_args)=0 then
	        return self._addPtrT(ptr_args);
	    else
	        return Error("Addition of more than one pointer and integers is not defined");
	    fi;
	fi;
    end,

    # premultiplies all constants, removes 0s
    cfold := meth(self)
        local cons, sym, e, a, t, zero;
        a := self.args;
        # Processing size 1 first allows to skip computation of the type
        # and of the zero
        if Length(a)=1 then
            return a[1];
        # fast special case for 2 terms, i.e., add(a, b)
        elif Length(a)=2 then
            if IsBound(self.t.zero) then
                t    := self.t;
                zero := self.t.zero();
                return Cond((a[1]=0 or a[1]=zero) and a[2].t = t, a[2],
                            (a[2]=0 or a[2]=zero) and a[1].t = t, a[1],
                            IsValue(a[1]) and IsValue(a[2]), t.value(self.__sum(a[1].v, a[2].v)),
                            self);
            else
                return self;
            fi;
        # general case for add with >2 terms
        else
            t := self.t;
            if IsBound(t.zero) then
                zero := t.zero(); cons := zero; sym := [];
                for e in self.args do
                    if IsSymbolic(e) then Add(sym, e);
                    else cons := self.__sum(cons, e);
                    fi;
                od;
                if sym=[]                then return cons;
                elif (cons=0 or cons=zero) and CopyFields(self, rec(args:=sym)).computeType() = t then self.args := sym;
                else self.args := [When(IsPtrT(t), TInt.value(cons.v), cons)] :: sym;
                fi;
                if Length(self.args)=1 then return self.args[1]; fi;
            fi;
            return self;
        fi;
    end,
    has_range := self >> ForAll(self.args, e -> Cond(IsValue(e), true, IsBound(e.has_range), e.has_range(), false) ),
    range := self >> let(ranges := List(self.args, e -> Cond(IsValue(e), e, IsVar(e), V(e.range-1), e.range())), Sum(ranges))
));

#F adds(<a>, <b>, ...) saturated addition
Class(adds, add, rec(
    __sum := (self, a, b) >> self.t.saturate(_stripval(a) + _stripval(b)),
));

Class(neg, AutoFoldExp, rec(
    ev := self >> -self.args[1].ev(),
    computeType := self >> let(t := self.args[1].t,
	Cond(IsPtrT(t),
	     t.aligned([t.alignment[1], -t.alignment[2] mod t.alignment[1]]),
	     t)),
));

#F sub(<a>, <b>)
Class(sub,  AutoFoldExp, rec(

    # __sub is overriden in descendant 'subs' (saturated substraction)
    __sub := (self, a, b) >> let(type := self.computeType(), type.value(a - b)),

    ev := self >> let(eve := self.__sub(self.args[1].ev(), self.args[2].ev()), When(self = eve, self, eve.ev()) ),

    computeType := meth(self)
        local a, b, isptr_a, isptr_b;
	[a, b] := self.args;
	[isptr_a, isptr_b] := [IsPtrT(a.t) or IsArrayT(a.t), IsPtrT(b.t) or IsArrayT(b.t)];
	if not isptr_a and not isptr_b then return UnifyPair(a.t, b.t);
	elif isptr_a and isptr_b then
	    return add._addPtrT([a, neg(b)]);
	elif isptr_a then
	    return add._ptrPlusOfs(a.t, -b);
	else #isptr_b
	    return add._ptrPlusOfs(neg(b).t, a);
	fi;
    end,


    cfold := self >> let(a := self.args[1], b := self.args[2], zero := self.t.zero(),
        Cond((a=0 or a=zero) and b.t=self.t, neg(b),
             (b=0 or b=zero) and a.t=self.t, a,
             a=b, zero,
             IsValue(a) and IsValue(b), self.__sub(a, b),
             self)),
));

#F subs(<a>, <b>) saturated substraction
Class(subs,  sub, rec(
    __sub := (self, a, b) >> self.t.saturate(_stripval(a) - _stripval(b)),
));

Class(mul,  AutoFoldExp, rec(
    ev := self >> let(eve := FoldL(self.args, (z, x) -> self.t.product(_stripval(z), x.ev()), self.t.one()), When(self = eve, self, V(eve).ev())),

    _ptrMul := function(ptr_t, mult)
        local t;
	t := Copy(ptr_t);
	t.alignment[2] := (t.alignment[2] * mult) mod t.alignment[1];
	return t;
    end,

    computeType := meth(self)
        local len, t, ptr_t, ptr_args, other_args, prod, args;
	args := self.args;

	len := Length(args);
	if   len=0  then return TInt;
	elif len=1  then return args[1].t;
	# elif len=2  then
	#     if IsPtrT(args[1].t) then
	# 	if not IsIntT(args[2].t) then Error("Can't multiply a pointer by a non-integer"); fi;
	# 	return self._ptrMul(args[1].t, args[2]);
	#     elif IsPtrT(args[2].t) then
	# 	if not IsIntT(args[1].t) then Error("Can't multiply a pointer by a non-integer"); fi;
	# 	return self._ptrMul(args[2].t, args[1]);
	#     else
	# 	return UnifyPair(args[1].t, args[2].t);
	#     fi;

	else
	    [ptr_args, other_args] := SplitBy(args, x->IsPtrT(x.t));
	    if ptr_args=[] then
		return UnifyTypesL(args);
	    elif Length(ptr_args) > 1 then Error("Can't multiply pointers");
	    else
		prod := Product(other_args);
		if other_args<>[] and not IsIntT(prod.t) then Error("Can't multiply a pointer by a non-integer"); fi;
		return  self._ptrMul(ptr_args[1].t, prod);
	    fi;
	fi;
    end,

    # premultiplies all constants, removes 1s, and returns 0 if any factors is = 0
    cfold := meth(self)
        local cons, sym, e, a, one, zero, t;
        t := self.t;   one := t.one();    zero := t.zero();
        a := self.args;
        # fast special case for 2 factors, i.e., mul(a, b)
        if Length(a)=2 then
            return Cond((a[1]=1 or a[1]=one) and t=a[2].t, a[2],
                        (a[2]=1 or a[2]=one) and t=a[1].t, a[1],
                        a[1]=0 or a[2]=0 or a[1] = zero or a[2] = zero, zero,
                        IsValue(a[1]) and IsValue(a[2]), t.value(t.product(a[1].v, a[2].v)),
                        self);
        elif Length(a)=1 then return a[1];
        # general case for mul with >2 factors
        else
            cons := one; sym := [];
            for e in self.args do
                if IsSymbolic(e) then Add(sym, e);
                elif e=0 or e=zero then return zero;
                else cons := cons * e;
                fi;
            od;
            if sym=[] then return cons;
            elif (cons=1 or cons=one) and t=UnifyTypesL(sym) then self.args := sym;
            else self.args := [cons] :: sym;
            fi;
            if Length(self.args)=1 then return self.args[1]; fi;
            return self;
        fi;
    end,
    has_range := self >> ForAll(self.args, e -> Cond(IsValue(e), true, IsBound(e.has_range), e.has_range(), false) ),
    range := self >> let(ranges := List(self.args, e -> Cond(IsValue(e), e, IsVar(e), V(e.range-1), e.range())), Product(ranges))
));

Class(pow,  AutoFoldExp, rec(
    ev := self >> self.args[1].ev() ^ self.args[2].ev(),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));


# max(...) is derived from min(...) by overloading _ev(<vars list>) method

Class(min, AutoFoldExp, rec(

    ev := self >> self._ev(self.args).ev(),

    computeType := self >> UnifyTypes(List(self.args, x->x.t)),
    cfold := meth(self)
        local m, vals, exps, args, i, a, op;
        op := ObjId(self);
        m  := Set(self.args);
        if Length(m)=1 then
            return m[1];
        else
            i := 1; vals := []; exps := [];
            while i<=Length(m) do
                a := m[i];
                if a _is Value then
                    Add(vals, a);
                elif a _is op then
                    Append(m, a.args);
                else
                    Add(exps, a);
                fi;
                i := i+1;
            od;
            args := When(vals<>[], [self._ev(vals)], []) :: exps;
            if args = self.args then
                return self;
            else
                return ApplyFunc(op, args);
            fi;
        fi;
    end,

    _ev := (self, vals) >> self.t.value(FoldL1(vals, (a,b) -> _ListElmOp(a, b, Min2))),

));

Class(max, min, rec(
    _ev := (self, vals) >> self.t.value(FoldL1(vals, (a,b) -> _ListElmOp(a, b, Max2))),
));

#F average(<a>, <b>)
Class(average,  AutoFoldExp, rec(
    ev := self >> _ListElmOp(self.t.sum(self.args[1].ev(), self.args[2].ev()), 2, QuoInt),
    computeType := self >> UnifyTypes(List(self.args, x->x.t)),
));

Class(re, AutoFoldExp, rec(
    ev := self >> let(
	t := InferType(self.args[1]),
        v := self.args[1].ev(),
        Cond(IsVecT(t), List(v, e -> ReComplex(Complex(e.ev()))),
                        ReComplex(Complex(v)))
    ),
    computeType := self >> self.args[1].t.realType()
));


Class(im, AutoFoldExp, rec(
    ev := self >> let(
	t := InferType(self.args[1]),
        v := self.args[1].ev(),
        Cond(IsVecT(t), List(v, e -> ImComplex(Complex(e.ev()))),
                        ImComplex(Complex(v)))
    ),
    computeType := self >> self.args[1].t.realType()
));

Class(conj, AutoFoldExp, rec(
    ev := self >> let(a := self.args[1].ev(),
        When(IsCyc(a), Global.Conjugate(a),
             ReComplex(a)-Cplx(0,1)*ImComplex(a))),
    computeType := self >> self.args[1].t
));

#F ----------------------------------------------------------------------------------------------
#F Expressions: Division
#F
#F fdiv(<a>, <b>) - divides two integers or two reals, result if always TReal
#F This is different from idiv and div.
#F
#F fdiv(TInt, TInt) == TReal
#F idiv(TInt, TInt) == TInt (rounding can happen)
#F ddiv(TInt, TInt) == TInt (rounding can happen, but unlike idiv add(ddiv(a,b),ddiv(c,b), ...) => ddiv(add(a,c, ...), b) allowed)
#F  div(TInt, TInt) == TInt (arguments are expected to be divisible)
#F

#F fdiv(a, b). Floating-point division
#F    fdiv(TInt, TInt) = TReal.
Class(fdiv,  AutoFoldExp, rec(
    ev := self >> (self.args[1].ev() / self.args[2].ev()),
    computeType := self >> TReal));

Declare(idiv);

_handle_idiv_mul := function(div_obj)
  local factors, d, gcd, f, m, den, ranges;
  m := div_obj.args[1];
  den := div_obj.args[2];
  
  if m.has_range() and m.range() < den then
  	return V(0);
  fi;

  factors := [];
  d := den;
  for f in m.args do
    if IsValue(f) then
      gcd := Gcd(f.ev(), d.ev());
      Add(factors, f/gcd);
      d := d/gcd;
    else Add(factors, f);
    fi;
  od;
  if d = 1 then return ApplyFunc(mul, factors);
  else return div_obj;
  fi;
end;

_handle_idiv_add := function(div_obj)  
  local values, addends, a, den, ranges;
  a := div_obj.args[1];
  den := div_obj.args[2];

  if a.has_range() and a.range() < den then
	return V(0);
  fi;

  values := Filtered(a.args, v -> IsValue(v));
  if ForAny(values, v -> (v.ev() mod den.ev()) <> 0) then
    return div_obj;
  fi;
  addends := List(a.args, v -> idiv(v, den));
  if ForAll(addends, v -> ObjId(v) <> idiv) then return ApplyFunc(add, addends);
  else return div_obj;
  fi;
end;

#F idiv(a, b). Integer division with rounding.
#F    idiv(a, b) = floor(fdiv(a, b))
Class(idiv, AutoFoldExp, rec(
    ev := self >> _ListElmOp(self.args[1], self.args[2], QuoInt),
    cfold := self >> let(a := self.args[1], b := self.args[2],
        Cond(a=a.t.zero(),                 self.t.zero(),
             a=b,                          self.t.one(),
             b=b.t.one() and a.t = self.t, a,
             IsValue(a) and IsValue(b),    self.t.value(self.ev()),
#Dani: Simplifying expr
             ObjId(a) = mul and IsValue(b), _handle_idiv_mul(self),
             ObjId(a) = add and IsValue(b), _handle_idiv_add(self),
             IsVar(a) and IsValue(b) and IsInt(a.range), When((a.range-1)<b.ev(), V(0), self),
             self)),
    computeType := self >> let( t := UnifyTypes(List(self.args, e -> e.t)),
                                Checked(IsOrdT(t.base_t()), t) ),
    has_range := self >> ForAll(self.args, e -> Cond(IsValue(e), true, IsBound(e.has_range), e.has_range(), false) ),
    range := self >> let(a := self.args[1], a_range := Cond(IsValue(a), a.ev(), IsVar(a), a.range-1, a.range().ev()), V(QuoInt(a_range, self.args[2].ev())) )
));

#F idivmod(i, n, d) = imod( idiv(i, d), n ).
#F Assume N-dim tensor dimension where d is the stride of dimension D and n*d of dimension D+1.
#F idivmod isolates the index i_D from the linearized i = .. + i_{D+1}*n*d + i_D*d + ...
Class(idivmod,  AutoFoldExp, rec(
    ev := self >> idiv( self.args[1].ev(), self.args[3].ev() ) mod self.args[2].ev() ,
    computeType := self >> TInt));


idiv_ceil := (a, b) -> idiv(a+b-1, b);

#F ddiv(a, b). Integer division with rounding. Same as idiv but
#F     add(ddiv(a,b),ddiv(c,b), ...) => ddiv(add(a,c, ...), b) allowed
Class(ddiv, idiv);

#F div(a, b). Exact integer (no rounding) or floating-point division
#F    If <a> and <b> are integers, they are expected to be divisible.
#F    If both are reals, then div(a, b) = fdiv(a, b)
#F
Class(div,  AutoFoldExp, rec(
    ev := self >> self.args[1].ev() / self.args[2].ev(),
    cfold := self >> let(a := self.args[1], b := self.args[2],
        Cond(a=0,                       self.t.zero(), # what if b==0?
             a=b,                       self.t.one(),
             b=1 and a.t = self.t,      a,
             IsValue(a) and IsValue(b), self.t.value(a.v / b.v),
             self)),
    computeType := self >> UnifyTypes(List(self.args, x->x.t))
));

#param div, a division that is propagated to params, when it is known that
#the params are divisible
Class(pdiv, div);

# In Spiral (unlike C) mod from negative number is a positive number: -5 mod 3 = 1
Class(imod, AutoFoldExp, rec(
    ev := self >> self.args[1].ev() mod self.args[2].ev(),
    cfold := self >> let(a := self.args[1], b := self.args[2],
        Cond(a=0, a,
             b=1, self.t.zero(),
             IsValue(a) and IsValue(b), self.t.value(a.v mod b.v),
	     ObjId(a)=ObjId(self) and a.args[2] = b, a,
	     	 IsBound(a.has_range) and a.has_range() and IsValue(b), let(r := When(IsVar(a), V(a.range-1), a.range()), When(r < b, a, self)),
             self)),
    computeType := self >> When(IsPtrT(self.args[1].t), TInt, UnifyTypes([self.args[1].t, self.args[2].t]))
));

Class(floor, AutoFoldExp, rec(
    ev := self >> let(f := self.args[1].ev(), Cond(
	IsDouble(f), d_floor(f),
	IsRat(f),    spiral.approx.FloorRat(f),
	Error("Don't know how take floor of <f>"))),

    computeType := self >> TInt));

Class(ceil, AutoFoldExp, rec(
    ev := self >> let(f := self.args[1].ev(), Cond(
	IsDouble(f), d_ceil(f),
	IsRat(f),    spiral.approx.CeilingRat(f),
	Error("Don't know how take ceiling of <f>"))),

    computeType := self >> TInt));

#F ----------------------------------------------------------------------------------------------
#F Expressions: Various functions
#F
V_true := V(true);
V_false := V(false);

Class(null, Exp, rec(
    computeType := self >> TPtr(TVoid)
));

# powmod(<phi>, <g>, <exp>, <N>) = phi * g^exp mod N
Class(powmod, AutoFoldExp, rec(
    ev := self >> self.args[1].ev () * PowerMod(self.args[2].ev(), self.args[3].ev(), self.args[4].ev())
                  mod self.args[4].ev(),
    computeType := self >> TInt));

# ilogmod(<n>, <g>, <N>) --  solution <exp> in powmod(1, <g>, <exp>, <N>) = <n> [g^exp mod N = n]
Class(ilogmod, AutoFoldExp, rec(
    ev := self >> LogMod(self.args[1].ev(), self.args[2].ev(), self.args[3].ev()),
    computeType := self >> TInt));

#F abs(<a>)  -- absolute value
Class(abs, AutoFoldExp, rec(
    ev := self >> _ListElmOp(self.args[1], self.args[1], (a,b) -> SignInt(a)*b),
    computeType := self >> self.args[1].t
));


#F absdiff(<a>,<b>)  -- absolute difference |<a>-<b>|
Class(absdiff, AutoFoldExp, rec(
    ev := self >> sub(max(self.args[1], self.args[2]), min(self.args[1], self.args[2])).ev(),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));

#F absdiff2(<a>,<b>)  -- absolute difference between a and b where 0<=b && (a==0 || a==2^n-1 && b<=a)
#F this operation can be implemented as a xor b for integer a and b
Class(absdiff2, absdiff);

#F sign(<a>) -- returns 1 if a is positive, -1 if negative, 0 if a=0
Class(sign, AutoFoldExp, rec(
    ev := self >> let(a:=self.args[1].ev(),
        Cond(a>0, 1, a<0, -1, 0)),
    computeType := self >> self.args[1].t
));

#F fpmul(fracbits, a, b)  -- fixed point multiplication, computes (a*b) >> fracbits
Class(fpmul, AutoFoldExp, rec(
    ev := self >> (self.args[2].ev() * self.args[3].ev()) / 2^self.args[1].ev(),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));

#F ----------------------------------------------------------------------------------------------
#F Expressions: Boolean Arithmetic and Conditions
#F
Class(logic_and,  AutoFoldExp, rec(
    ev := self >> ForAll(self.args, x->x.ev()),
    cfold := meth(self)
        local a;
        a := Filtered(self.args, x->x<>true);
        if ForAny(a, x->x=false) then return V_false;
        elif a=[] then return V_true;
        elif Length(a)=1 then return a[1];
        else self.args := a;
             return self;
        fi;
    end,
    computeType := self >> TBool
));

Class(logic_or,   AutoFoldExp, rec(
    ev := self >> ForAny(self.args, x->x.ev()),
    cfold := meth(self)
        local a;
        a := Filtered(self.args, x->x<>false);
        if ForAny(a, x->x=true) then return V_true;
        elif a=[] then return V_false;
        elif Length(a)=1 then return a[1];
        else self.args := a;
             return self;
        fi;
    end,
    computeType := self >> TBool
));

Class(logic_neg,  AutoFoldExp, rec(
    ev := self >> When(self.args[1].ev(), false, true),
    computeType := self >> TBool));


# Mixin for comparision operations.
# Users should define method _ev_op(a, b) only, which defines operation.
_logic_mixin := rec(
    computeType := self >> let(
        types := List(self.args, e->e.t),
        # There is no TPtr unification defined at the moment but it's legal
        # to compare pointers (at least with TPtr(TVoid)) so here is this stupid hack
        t     := Cond( ForAny(types, IsPtrT), TBool, UnifyTypes(types)),
        When( IsVecT(t),
            TVect(TBool, t.size),
        # else
            TBool)),
    ev := self >> let(
        a := self.args,
        l := Length(a),
        Checked( l>1,
            ApplyFunc(logic_and, List([2..l],
                i -> _ListElmOp(a[i-1], a[i], self._ev_op)
            )).ev()
        )
    ),
);

#F eq(a, b, c, ...) symbolic representation of a = b = c = ...
Class(eq,  _logic_mixin, AutoFoldExp, rec( _ev_op := (a, b) -> Checked(not AnySyms(a,b), a=b  )));

#F neq(a, b) symbolic representation of a<>b
Class(neq, _logic_mixin, AutoFoldExp, rec( _ev_op := (a, b) -> Checked(not AnySyms(a,b), a<>b )));

#F leq(a, b, c, ...) symbolic representation of a <= b <= c <= ...
Class(leq, _logic_mixin, AutoFoldExp, rec( _ev_op := (a, b) -> Checked(not AnySyms(a,b), a<=b )));

#F lt(a, b, c, ...) symbolic representation of a < b < c < ...
Class(lt,  _logic_mixin, AutoFoldExp, rec( _ev_op := (a, b) -> Checked(not AnySyms(a,b), a<b  )));

#F geq(a, b, c, ...) symbolic representation of a >= b >= c >= ...
Class(geq, _logic_mixin, AutoFoldExp, rec( _ev_op := (a, b) -> Checked(not AnySyms(a,b), a>=b )));

#F gt(a, b, c, ...) symbolic representation of a > b > c > ...
Class(gt,  _logic_mixin, AutoFoldExp, rec( _ev_op := (a, b) -> Checked(not AnySyms(a,b), a>b  )));

_logic_mask_mixin := rec(
    computeType := self >> let(
        t  := UnifyTypes(List(self.args, e->e.t)),
        b  := t.base_t(),
        nb := Cond(
            ObjId(b) in [T_Real, T_Int, T_UInt], T_Int(b.params[1]),
            b        in [TReal, TInt, TUInt],    TInt,
            Error("unexpected data type")
        ),
        Cond( IsVecT(t), TVect(nb, t.size), nb)
    ),

    ev := self >> let( b := Inherited(), _ListElmOp( b, b, (a, b) -> Checked(not IsSymbolic(a), When(a=true, -1, 0))))
);

#F mask_gt(a, b) symbolic representation of a > b where result is an integer mask: -1 (true) or 0 (false)
Class(mask_gt, _logic_mask_mixin, gt);

#F mask_eq(a, b) symbolic representation of a = b where result is an integer mask: -1 (true) or 0 (false)
Class(mask_eq, _logic_mask_mixin, eq);

#F mask_lt(a, b) symbolic representation of a < b where result is an integer mask: -1 (true) or 0 (false)
Class(mask_lt, _logic_mask_mixin, lt);


Class(cond, Exp, rec(
    eval := meth(self)
        local i, cc;
	i := 0;
        for i in [1..QuoInt(Length(self.args),2)] do
            cc := self.args[2*i-1].eval();
            if not IsValue(cc) then return self; # unevaluatable cond
            elif ((IsBool(cc.v) and cc.v) or (IsInt(cc.v) and cc.v<>0)) then # true clause found
                return self.args[2*i].eval();
            fi;
        od;
	if 2*i+1 > Length(self.args) then
            # in the case of nested conds, this particular cond might be unreachable,
	    # so it can be invalid, we generate errExp() in this case, instead of crashing
	    # return errExp(self.t);
            return Error("Else clause missing in 'cond' object <self>");
        else
	    return self.args[2*i+1].eval();
	fi;
    end,
    computeType := self >> UnifyTypes(List([1..QuoInt(Length(self.args),2)], i->self.args[2*i].t)),
    ev := self >> let(ev:=self.eval(), When(IsValue(ev), ev.v, ev))
));

#F _map_cond(<cexp>, <pred_func>, <exp_func>)
#F   Maps cond(...) expression <cexp> by applying <pred_func> to predicates and <exp_func> to expressions.

_map_cond := (cexp, pred_func, exp_func) -> ApplyFunc(cond, List([1..Length(cexp.args)], i ->
    Cond( i mod 2 = 1 and i<>Length(cexp.args), pred_func, exp_func)(cexp.args[i])));

#F maybe() --  "magic" boolean function, that satisfies logic_not(maybe()) = maybe()
#F
#F  maybe() behaves as 'true' inside 'and'/'or' operators,
#F  but also satisfies the uncertainty rule logic_not(maybe()) = maybe()
#F
Class(maybe, Exp, rec(
    computeType := self >> TBool
));

#F ----------------------------------------------------------------------------------------------
#F Expressions: Bit manipulation
#F

Class(bin_parity, Exp, rec(
    ev := self >> BinParity(self.args[1].ev()),
    computeType := self >> self.args[1].t
));
Class(bin_and, Exp, rec(
    ev := self >> _ListElmOp(self.args[1], self.args[2], BinAnd),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));
Class(bin_or, Exp, rec(
    ev := self >> _ListElmOp(self.args[1], self.args[2], BinOr),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));
Class(bin_andnot, Exp, rec(
    ev := self >> BinAnd(BinNot(self.args[1].ev()), self.args[2].ev()),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));
Class(bin_xor,   Exp, rec(
    ev := self >> _ListElmOp(self.args[1], self.args[2], BinXor),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));
Class(adrgen, Exp, rec(
    ev := self >> (2^self.args[1].ev()-1) - (2^self.args[2].ev()-1),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));
Class(concat, Exp, rec(
    ev := self >> (self.args[1].ev() * (2^self.args[3].ev()) + self.args[2].ev()),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t)
));
Class(truncate, Exp, rec(
    ev := self >> BinAnd(self.args[1].ev(), 2^self.args[2].ev()-1),
    computeType := self >> self.args[1].t
));

Class(bin_shr, Exp, rec(
    ev := self >> let(
        a:=self.args[1].ev(), b:=self.args[2].ev(), bits:=When(IsBound(self.args[3]), self.args[3].ev(), false),
        Cond(bits=false, When( IsList(a), ShiftList(a, -b, 0), Int(a * 2^(-b))), Int(a * 2^(-b)) mod 2^bits)),
    computeType := self >> self.args[1].t
));

Class(bin_shl,  Exp, rec(
    ev := self >> let(
        a:=self.args[1].ev(), b:=self.args[2].ev(), bits:=When(IsBound(self.args[3]), self.args[3].ev(), false),
        Cond( bits=false, When( IsList(a), ShiftList(a, b, 0), Int(a * 2^b) ),
              Int(a * 2^b) mod 2^bits)),
    computeType := self >> self.args[1].t
));

Class(arith_shr, Exp, rec(
    ev := self >> let(
        a:=self.args[1].ev(), b:=self.args[2].ev(), bits:=When(IsBound(self.args[3]), self.args[3].ev(), false),
        Cond(bits=false, When( IsList(a), ShiftList(a, -b, Last(a)), Int(a * 2^(-b))), Int(a * 2^(-b)) mod 2^bits)),
    computeType := self >> self.args[1].t
));

Class(arith_shl,  Exp, rec(
    ev := self >> let(
        a:=self.args[1].ev(), b:=self.args[2].ev(), bits:=When(IsBound(self.args[3]), self.args[3].ev(), false),
        Cond(bits=false, When( IsList(a), ShiftList(a, b, 0), Int(a * 2^b)), Int(a * 2^b) mod 2^bits)),
    computeType := self >> self.args[1].t
));

Class(rCyclicShift, Exp, rec(
    ev := self >> let(a := self.args[1].ev(), shift := self.args[2].ev(),
        c := 2^shift, bits := self.args[3].ev(),
        BinAnd(a, c-1) * 2^(bits-shift) + Int(a / c)),
    computeType := self >> self.args[1].t
));

Class(bit_sel, Exp, rec(
    ev := self >> let(a := self.args[1].ev(), bit := self.args[2].ev(),
                      bin_and(arith_shr(a, bit), 1).ev()
    )
));

Class(xor, Exp, rec(
    ev := self >> Xor    (self.args, e -> e.ev()),
    computeType := self >> UnifyTypes(List(self.args, x->x.t))
));

#F ----------------------------------------------------------------------------------------------
#F Expressions: Irrational functions
#F
Class(omega, AutoFoldExp, rec(
    ev := self >> E(self.args[1].ev()) ^ self.args[2].ev(),
    computeType := self >> TComplex));
Class(exp, AutoFoldRealExp, rec(
    ev := self >> d_exp(self.args[1].ev())));
Class(log, AutoFoldRealExp, rec(
    ev := self >> let( l := d_log(self.args[1].ev()),
         Cond(Length(self.args)=2, l/d_log(self.args[2].ev()), l))));
Class(cospi, AutoFoldExp, rec(
    ev := self >> CosPi(self.args[1].ev()),
    computeType := self >> TReal));
Class(sinpi, AutoFoldExp, rec(
    ev := self >> SinPi(self.args[1].ev()),
    computeType := self >> TReal));
Class(omegapi, AutoFoldExp, rec(
    ev := self >> ExpIPi(self.args[1].ev()),
    computeType := self >> TComplex));
Class(sqrt, AutoFoldRealExp, rec(
    ev := self >> Sqrt(self.args[1].ev())));
Class(rsqrt, AutoFoldRealExp, rec(
    ev := self >> 1/Sqrt(self.args[1].ev())));

#F ----------------------------------------------------------------------------------------------
#F Expressions: Specials and Wrappers
#F
Class(PlaceholderExp, Exp, rec(
    ev := self >> self.args[1].ev(),
    computeType := self >> self.args[1].t
));

Class(no_mod,  PlaceholderExp);
Class(small_mod, imod);
Class(accu,    PlaceholderExp);
Class(depends, PlaceholderExp);
Class(depends_memory, depends);

Class(virtual_var, Exp, rec(
    __call__ := (self, vars, idx) >>
       let(idx2 := toExpArg(idx),
           When(IsValue(idx2),
               Cond(idx2.v >= Length(vars), errExp(self.t), vars[idx2.v+1]),
               WithBases(self,
                   rec(args := [vars, idx2],
                       operations := ExpOps,
                       t := Last(vars).t)))),

    computeType := self >> Last(self.args[1]).t,

    ev := self >> let(
            vars := self.args[1],
            idx := self.args[2].eval(),
        Cond(not IsValue(idx),
                 self,
             idx.v < 0,
                 errExp(self.t),
             IsList(vars),
                 Cond(idx.v >= Length(vars), errExp(self.t), vars[idx.v+1]))),
));

#F castizx(<exp>) - cast signed <exp> to twice larger data type with zero extension
Class(castizx, Exp, rec(
    __call__ := (self, expr) >>
        WithBases(self, rec(
	    args       := [toExpArg(expr)],
	    operations := ExpOps,
	)).setType(),

    computeType := self >> self.args[1].t.double().toSigned(),

    ev := self >> self.args[1].t.toUnsigned().value(self.args[1].ev()).ev(),
));

#F castuzx(<exp>) - cast unsigned <exp> to twice larger data type with zero extension
Class(castuzx, castizx, rec(
    computeType := self >> self.args[1].t.double().toUnsigned(),
));




# cmemo(<expr>, <target>, <prefix>)
Class(cmemo, Exp, rec(
    __call__ := (self, prefix, target, exp) >>
    Cond(IsValue(exp), exp.v,
         #IsVar(exp), exp,
         WithBases(self, rec(
         operations := ExpOps,
         prefix  := prefix,
         target  := target,
         args := [ var.fresh_t(prefix, TInt) ],
         mapping := toExpArg(exp).eval() ))),
    eval := self >> self.mapping.eval()
));

ExprFuncs := rec(
    T_SUM := add,
    T_DIFF := sub,
    T_PROD := mul,
    T_QUO  := div,
    T_MOD  := imod,
    T_POW  := pow,
    nth    := nth,
    Int    := floor,
    QuoInt := idiv,
    LogMod := ilogmod,
    CosPi := cospi,
    SinPi := sinpi,
    Sqrt  := sqrt,
    ReComplex := re,
    ImComplex := im,
    Cond := cond
);

#F noneExp(<t>) -  represents an uninialized value of type <t>
#F
#F This handles the sitation with Scat * Diag * Scat(f)
#F Scat(f) should never write explicit 0's even though Diag scales them
#F
Class(noneExp, Exp, rec(
    computeType := self >> self.args[1]
));

#F errExp(<t>) -  represents an invalid result of type <t>
#F
#F   The reason this is used is to support the following strange rewrite:
#F      0 * nth(T, i) -> 0,   when i is out of bounds
#F   For example if i<0, normally the above might break, but using errExp:
#F      0 * nth(T, -1) -> 0 * errExp(TReal) -> 0
#F
Class(errExp, Exp, rec(
    computeType := self >> self.args[1]
));

#F funcExp(<i>) -- used to "hack" affine transformations out of Gath/Scat
#F
#F See Doc(Gath) for an explanation on how this works.
#F <i> must be an integer expression.
#F
#F Currently, this has the following semantics
#F
#F  nth(X, i)          == X[i]
#F  nth(X, funcExp(i)) == i
#F
#F The proper way of doing this would be instead (using h. coords, X[len(x)] = 1)
#F nth(X, funcExp(i)) -> i * nth(X, len(X)) = i * X[len(X)] = i
#F
#F See http://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
#F
Class(funcExp, Exp, rec(
    eval := self >> funcExp(self.args[1].eval()),
    computeType := self >> self.args[1].t,
    can_fold := False,
));

#F ----------------------------------------------------------------------------------------------
#F GAP Operations Records
#F

Class(ExpOps, PrintOps, rec(
   \+   := add,
   \-   := sub,
   \*   := (e1,e2) -> When(e1=-1, neg(e2), mul(e1,e2)),
   \/   := div,
   \^   := pow,
   \mod := imod,
   \=   := (e1,e2) -> Cond(
       ObjId(e1) <> ObjId(e2), false,
       e1.rChildren() = e2.rChildren()),
   \<   := (e1,e2) -> Cond(
       ObjId(e1) <> ObjId(e2), ObjId(e1) < ObjId(e2),
       e1.rChildren() < e2.rChildren())
));

Class(VarOps, ExpOps, rec(
    \= := (v1,v2) -> Same(v1,v2),
    \< := (v1,v2) -> Cond(not (IsVar(v1) and IsVar(v2)), ObjId(v1) < ObjId(v2),
                          BagAddr(v1) < BagAddr(v2))
));

Class(NthOps, ExpOps, rec(
    \= := (v1,v2) -> IsRec(v1) and IsRec(v2) and Same(ObjId(v1), ObjId(v2))
                     and v1.loc=v2.loc and v1.idx = v2.idx,
    \< := (v1,v2) -> Cond(
                      not Same(ObjId(v1), ObjId(v2)), ObjId(v1) < ObjId(v2),
                      v1.loc=v2.loc, v1.idx < v2.idx,
                      v1.loc < v2.loc)
));

_val := x->Cond(IsValue(x) or IsSymbolic(x), x, InferType(x).value(x));

ValueOps.\+ := (aa,bb) -> let(a:=_val(aa), b:=_val(bb), Cond(IsValue(a) and IsValue(b),
    let(t:=UnifyPair(a.t, b.t), t.value(t.sum(a.v, b.v))), add(a, b)));

ValueOps.\- := (aa,bb) -> let(a:=_val(aa), b:=_val(bb), Cond(IsValue(a) and IsValue(b),
    let(t:=UnifyPair(a.t, b.t), t.value(t.sum(a.v, -b.v))), sub(a, b)));

ValueOps.\* := (aa,bb) -> let(a:=_val(aa), b:=_val(bb), Cond(IsValue(a) and IsValue(b),
    let(t:=UnifyPair(a.t, b.t), t.value(t.product(a.v, b.v))), mul(a, b)));

ValueOps.\/ := div;
ValueOps.\^ := pow;
ValueOps.\mod := imod;

#----------------------------------------------------------------------------------------------
# Command : high level instructions
#
#   skip
#   assign
#   chain
#   decl
#   data
#   loop
#----------------------------------------------------------------------------------------------

CmdOps := rec(Print := s -> s.print(0,3));

Class(Command, AttrMixin, rec(
   isCommand := true,
   print := (self,i,si) >> Print(self.__name__),

   countedArithCost := (self, countrec) >> countrec.arithcost(self.countOps(countrec)),

   countOps := meth(self, countrec)
      local cmds, ops, i;
      ops := List([1..Length(countrec.ops)], i->0);

      if self.__name__ = "func" and self.id = "init" then return(ops); fi;

      if IsBound(self.cmds) then cmds := self.cmds;
      else if IsBound(self.cmd) then cmds := [self.cmd]; else return(0); fi;
      fi;

      for i in cmds do
        if IsBound(i.countOps) then
           ops := ops + i.countOps(countrec);
        else
           Error(i.__name__, "doesn't have countOps");
        fi;
      od;
      return(ops);
   end,

   free := self >> Set(ConcatList(self.rChildren(), FreeVars)),
   from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch).takeA(self.a)
));

Class(ExpCommand, Command, rec(
    isExpCommand := true,
    __call__ := arg >> let(self := arg[1], args := Drop(arg,1),
    WithBases(self, rec(
       operations := CmdOps,
        args := List(args, toExpArg)))),
    rChildren := self >> self.args,
    rSetChild := meth(self, i, newC) self.args[i] := newC; return newC; end,
    print := (self,i,si) >>
        Print(self.__name__, "(", PrintCS(self.args), ")")
));

IsCommand := x -> IsRec(x) and IsBound(x.isCommand) and x.isCommand;
IsExpCommand := x -> IsRec(x) and IsBound(x.isExpCommand) and x.isExpCommand;

#F throw(<arg>) - symbolic representation of exception throw
#F
Class(throw, ExpCommand);

#F call(<func>, <arg1>, <arg2>, ...) - symbolic representation of an external function call
#F
Class(call, ExpCommand);

Class(skip, Command, rec(
   __call__ := self >> WithBases(self, rec(operations:=CmdOps)),
   print := (self,i,si) >> Print(self.__name__, "()"),
   rChildren := self >> [],
   free := self >> [],
   op_in := self >> Set([])
));

Class(break, skip);

Class(const, Command, rec(
  __call__ := (self, value) >> WithBases(self, rec(operations:=CmdOps, val:=value)),
   print := (self,i,si) >> Print(self.__name__, "(", self.val ,")"),
   rChildren := self >> [],
   free := self >> []
));

Class(dma_barrier, skip, rec(
));

Class(dist_barrier, skip, rec(
));

Class(noUnparse, Command, rec(
   __call__ := (self, string) >> WithBases(self, rec(operations:=CmdOps, str:=string)),
   print := (self,i,si) >> Print(self.__name__, "(\"", self.str, "\")"),
   rChildren := self >> [],
   rSetChild := (self, n, c) >> Error("no children")
));

Class(assign, Command, rec(
   isAssign := true,
   __call__ := (self, loc, exp) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       exp := toExpArg(exp))),

   rChildren := self >> [self.loc, self.exp],
   rSetChild := rSetChildFields("loc", "exp"),
   unroll := self >> self,

   #Must do a collect so that nested assigns work
   countOps := (self, countrec) >> List([1..Length(countrec.ops)],
        i->Length(Collect(self, @(1, countrec.ops[i], e->IsRec(e) and
            ((IsBound(e.t) and (ObjId(e.t)=TVect) or (IsBound(e.countAsVectOp) and e.countAsVectOp())))))) ),

   print := (self,i,si) >> let(name := Cond(IsBound(self.isCompute) and self.isCompute,
                                            gap.colors.DarkYellow(self.__name__),
                                            IsBound(self.isLoad) and self.isLoad,
                                            gap.colors.DarkRed(self.__name__),
                                            IsBound(self.isStore) and self.isStore,
                                            gap.colors.DarkGreen(self.__name__),
                                            self.__name__),
                                Print(name, "(", self.loc, ", ", self.exp, ")"))


));

Class(regassign, assign, rec(
   isAssign := true,
   __call__ := (self, loc, exp) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       exp := toExpArg(exp))),

   rChildren := self >> [self.loc, self.exp],
   rSetChild := rSetChildFields("loc", "exp"),
   unroll := self >> self,

   #Must do a collect so that nested assigns work
   countOps := (self, countrec) >> List([1..Length(countrec.ops)],
        i->Length(Collect(self, @(1, countrec.ops[i], e->IsRec(e) and
            ((IsBound(e.t) and (ObjId(e.t)=TVect) or (IsBound(e.countAsVectOp) and e.countAsVectOp())))))) ),

   print := (self,i,is) >> let(name := Cond(IsBound(self.isCompute) and self.isCompute,
                                            gap.colors.DarkYellow(self.__name__),
                                            IsBound(self.isLoad) and self.isLoad,
                                            gap.colors.DarkRed(self.__name__),
                                            IsBound(self.isStore) and self.isStore,
                                            gap.colors.DarkGreen(self.__name__),
                                            self.__name__),
                                Print(name, "(", self.loc, ", ", self.exp, ")"))


));

IsAssign := x -> IsRec(x) and IsBound(x.isAssign) and x.isAssign;

Class(assign_acc, assign);

# syntax like a printf, prints out something in the output code.

Class(PRINT, Command, rec(
    __call__ := (arg) >> let(
        self := arg[1],
        fmt := arg[2],
        vars := Drop(arg, 2),
        Checked(
            IsString(fmt),
            IsList(vars),
            WithBases(self, rec(
                operations := CmdOps,
                fmt := fmt,
                vars := vars
            ))
        )
    ),

    rChildren := self >> Concat([self.fmt], self.vars),

    rSetChild := meth(self, n, val)
        if n = 1 then
            self.fmt := val;
        else
            self.vars[n-1] := val;
        fi;
    end,

    print := (self, i, si) >> Print(self.__name__, "(\"", self.fmt, When(Length(self.vars) <> 0, "\", ", "\""), PrintCS(self.vars), ")")
));

#NOTE THAT HAO
# Class(PRINT, Exp);

# inserts a comment into the output code
Class(comment, Command, rec(
    isComment := true,
    __call__ := (self, exp) >> Checked(
        IsString(exp),
        WithBases(self,
            rec(operations := CmdOps,
                exp := exp
            )
        )
    ),

    rChildren := self >> [self.exp],
    rSetChild := rSetChildFields("exp"),

    print := (self, i, si) >> Print(self.__name__, "(\"", self.exp, "\")")
));

Class(quote, Command, rec(
    isQuote := true,
    __call__ := (self, cmd) >> Checked(
        IsCommand(cmd),
        WithBases(self, rec(
            operations := CmdOps,
            cmd := cmd
        ))
    ),

    rChildren := self >> [self.cmd],
    rSetChild := rSetChildFields("cmd"),

    print := (self, i, si) >> Print(self.__name__, "(", self.cmd.print(i+si,si), ")")
));

Class(wrap, Command, rec(
   rChildren := self >> [ self.cmd ],
   rSetChild := rSetChildFields("cmd"),

   __call__ := (self, cmd) >> WithBases(self,
       rec(operations := CmdOps,
           cmd        := Checked(IsCommand(cmd), cmd))),

   print := meth(self,i,si)
         Print(self.__name__, "(\n");
         Print(Blanks(i+si), self.cmd.print(i+si, si), "\n");
     Print(Blanks(i), ")");
   end
));

Class(multiwrap, Command, rec(
   rChildren := self >> self.cmds,
   rSetChild := meth(self, n, newChild) self.cmds[n] := newChild; end,

   __call__ := meth(arg)
       local self, cmds;
       self := arg[1];
       cmds := Flat(Drop(arg, 1));
       return WithBases(self,
       rec(operations := CmdOps,
           cmds       := Checked(ForAll(cmds, IsCommand), cmds)));
   end,

   printCmds := meth(self, i, si)
       local c;
       for c in Take(self.cmds, Length(self.cmds)-1) do
           Print(Blanks(i));
       c.print(i, si);
       Print(",\n");
       od;
       Print(Blanks(i));
       Last(self.cmds).print(i, si);
       Print("\n");
   end,

   print := (self,i,si) >> When(Length(self.cmds)=0,
       Print(self.__name__, "()"),
       Print(self.__name__, "(\n", self.printCmds(i+si, si), Blanks(i), ")"))
));

IsChain :=  x -> IsRec(x) and IsBound(x.isChain) and x.isChain;

Class(chain, multiwrap, rec(
   isChain := true,

   flatten := self >> let(cls := self.__bases__[1],
       CopyFields(self, rec(cmds := ConcatList(self.cmds,
           c -> Cond(IsChain(c) and not IsBound(c.doNotFlatten), c.cmds,
                 ObjId(c) = skip, [],
             [c]))))),

   __call__ := meth(arg)
       local self, cmds;
       [self, cmds] := [arg[1], Flat(Drop(arg, 1))];
       return WithBases(self, rec(
           operations := CmdOps,
           cmds       := Checked(ForAll(cmds, IsCommand), cmds)));
   end

));

Class(unroll_cmd, multiwrap, rec(
    flatten := self >> chain(self.cmds).flatten()
));

Class(kern, Command, rec(
    __call__  := (self, bbnum, cmd) >> WithBases(self, rec(
        bbnum := bbnum,
        cmd := Checked(IsCommand(cmd), cmd),
        operations := CmdOps
    )),

    rChildren := self >> [self.bbnum, self.cmd],
    rSetChild := rSetChildFields("bbnum", "cmd"),

    print := (self, i, si) >> Print(self.__name__, "(", self.bbnum, ", ", self.cmd.print(i+si,si), ")")
));

Class(unparseChain, multiwrap, rec(
   #rChildren := self >> [],
   #rSetChild := self >> Error("Not implemented"),

   __call__ := meth(arg)
       local self, cmds;
       [self, cmds] := [arg[1], Flat(Drop(arg, 1))];
       return WithBases(self, rec(
           operations := CmdOps,
           cmds       := cmds));
   end,

   print := (self,i,si) >> Print(self.__name__)
));


Class(decl, Command, rec(
   __call__ := meth(self, vars, cmd)
       local tvars;
       if IsList(vars) and Length(vars)=0 then return cmd; fi;
       tvars := When(IsList(vars), Checked(ForAll(vars, IsLoc), vars),
                                   Checked(IsLoc(vars), [vars]));
       Sort(tvars, (a,b) -> a.id < b.id);
       return WithBases(self,
           rec(operations := CmdOps,
           cmd        := Checked(IsCommand(cmd), cmd),
           vars       := tvars));
   end,

   rChildren := self >> [self.vars, self.cmd],
   rSetChild := rSetChildFields("vars", "cmd"),

   print := (self, i, si) >> Print(self.__name__, "(", self.vars, ",\n",
       Blanks(i+si),
       self.cmd.print(i+si, si),
       "\n", Blanks(i), ")"),

   free := self >> Difference(self.cmd.free(), Set(self.vars)),
));

Class(data, Command, rec(
   __call__ := (self, var, value, cmd) >> WithBases(self,
           rec(operations := CmdOps,
           cmd        := Checked(IsCommand(cmd), cmd),
           var        := Checked(ObjId(var) in [code.var,code.param], var),
           value      := value)), #Checked(IsValue(value), value))),

   rChildren := self >> [self.var, self.value, self.cmd],
   rSetChild := rSetChildFields("var", "value", "cmd"),

   free := self >> Difference(Union(FreeVars(self.cmd), FreeVars(self.value)), [self.var]),

   print := (self, i, si) >> Print(self.__name__, "(", self.var, ", ",
       self.value, ",\n", Blanks(i+si),
       self.cmd.print(i+si, si),
       "\n", Blanks(i), ")"),

));

#F rdepth_marker(<depth>, <cmd>) - Autolib's recursion depth marker,
#F    BCRDepth turnes into this marker, <depth> >= 1, one is the deepest (recursion) level.
#F
Class(rdepth_marker, Command, rec(
    __call__  := (self, depth, cmd) >> WithBases(self, rec(
        depth      := depth,
        cmd        := Checked(IsCommand(cmd), cmd),
        operations := CmdOps
    )),

    rChildren := self >> [self.depth, self.cmd],
    rSetChild := rSetChildFields("depth", "cmd"),

    print := (self, i, si) >> Print(self.__name__, "(", self.depth, ", ", self.cmd.print(i+si,si), ")")
));

Declare(SubstVars);

Class(asmvolatile, Command, rec(
   __call__ := meth(self, asm)
       return WithBases(self,
           rec(operations := CmdOps,
           asm       := asm));
   end,

   rChildren := self >> [self.asm],
   rSetChild := rSetChildFields("asm"),
  print := (self, i, si) >> Print("asmvolatile(\n",self.asm,")\n"))
);

Class(loop_base, Command, rec(
   isLoop := true,

   countOps := (self, countrec) >> self.cmd.countOps(countrec) * Length(listRange(self.range)),

   rChildren := self >> [self.var, self.range, self.cmd],
   rSetChild := rSetChildFields("var", "range", "cmd"),

   print := (self, i, si) >> Print(self.__name__, "(", self.var, ", ",
       self.range, ",\n", Blanks(i+si),
       self.cmd.print(i+si, si),
       Print("\n", Blanks(i), ")")),

   free := meth(self) local c;
       c := self.cmd.free();
       if IsExp(self.range) then c:=Set(Concat(c, self.range.free())); fi;
       SubtractSet(c, Set([self.var]));
       return c;
   end
));

Declare(loopn);


FreshVars := function (code, map)
    local v;
    for v in Filtered(Difference(Collect(code, var), code.free()), e -> IsArrayT(e.t)) do
        map.(v.id) := var.fresh_t(String(Filtered(v.id, c -> not c in "0123456789")), v.t);
    od;
    return SubstTopDownNR(code, @(1, var, x -> IsBound(map.(x.id))), e -> map.(e.id));
end;

Class(loop, loop_base, rec(

   __call__ := meth(self, loopvar, range, cmd)
       local result;
#       Constraint(IsVar(loopvar)); YSV: could be a param
       Constraint(IsCommand(cmd));
       if IsSymbolic(range) then return loopn(loopvar, range, cmd); fi;
       range := toRange(range);
       if range = 1 then
           return SubstBottomUp(Copy(cmd), @(1, var, e->e=loopvar), e->V(0));
       elif range = 0 then
           return skip();
       else
           loopvar.setRange(range);
           range := listRange(range);
           result := WithBases(self,
               rec(operations := CmdOps, cmd := cmd, var := loopvar, range := range));
           loopvar.isLoopIndex := true;
           #loopvar.loop := result;
           return result;
       fi;
   end,

   unroll := self >>
      chain( List(self.range,
              index_value -> FreshVars(Copy(self.cmd),
                                   tab((self.var.id) := V(index_value)))))
));


Class(multibuffer_loop, loop_base, rec(

   __call__ := meth(self, loopvar, range, y, x, gathmem, twiddles, bufs, cmd, scatmem)
       local result;
#       Constraint(IsVar(loopvar));  YSV: could be a param
       Constraint(IsCommand(cmd));
       #if IsSymbolic(range) then return loopn(loopvar, range, cmd); fi;
       range := toRange(range);
       #if range = 1 then
       #    return SubstBottomUp(Copy(cmd), @(1, var, e->e=loopvar), e->V(0));
       #elif range = 0 then
       #    return skip();
       #else
           loopvar.setRange(range);
           range := listRange(range);
           result := WithBases(self,
               rec(operations := CmdOps,
               gathmem := gathmem,
               twiddles := twiddles,
               bufs := bufs,
               cmd := cmd,
               y := y,
               x := x,
               scatmem := scatmem,
               var := loopvar,
               range := range));
           loopvar.isLoopIndex := true;
           #loopvar.loop := result;
           return result;
       #fi;
   end,

   rChildren := self >> [self.var, self.range, self.y, self.x, self.gathmem, self.twiddles, self.bufs, self.cmd, self.scatmem],
   rSetChild := rSetChildFields("var", "range", "y", "x", "gathmem", "twiddles", "bufs", "cmd", "scatmem"),

   unroll := self >>
      chain( List(self.range,
              index_value -> SubstVars(Copy(self.cmd),
                                   tab((self.var.id) := V(index_value)))))
));


Class(mem_loop, multibuffer_loop);


Class(loop_sw, loop, rec(
   unroll := self >>
      chain( List(self.range,
              index_value -> SubstVars(Copy(self.cmd),
                                   tab((self.var.id) := V(index_value)))))
));


Class(loopn, loop_base, rec(

   __call__ := meth(self, loopvar, range, cmd)
       local result;
#       Constraint(IsVar(loopvar)); # YSV: could be a param
       Constraint(IsCommand(cmd));
       range := toExpArg(range);
       if IsValue(range) then return loop(loopvar, range.v, cmd);
       else
           loopvar.setRange(range);
           result := WithBases(self,
               rec(operations := CmdOps, cmd := cmd, var := loopvar, range := range));
           loopvar.isLoopIndex := true;
           return result;
       fi;
   end,

   unroll := self >> let(res:=loop(self.var, self.range.ev(), self.cmd),
       When(ObjId(res)=loop, res.unroll(), res)), # res if loop has single iteration it returns just the body
));

Class(doloop, loop_base, rec(
   __call__ := (self, loopvar, range, cmd) >> WithBases(self,
               rec(operations := CmdOps, cmd := cmd, var := loopvar, range := range))
));


# IF(<cond>, <then_cmd>, <else_cmd>)  -  symbolic representation of a conditional
#
Class(IF, Command, rec(
   __call__ := (self, cond, then_cmd, else_cmd) >>
       Cond( cond = true,  Checked(IsCommand(then_cmd), then_cmd),
             cond = false, Checked(IsCommand(else_cmd), else_cmd),
             WithBases( self, rec(
                 operations := CmdOps,
                 then_cmd   := Checked(IsCommand(then_cmd), then_cmd),
                 else_cmd   := Checked(IsCommand(else_cmd), else_cmd),
                 cond       := toExpArg(cond))) ),

   rChildren := self >> [self.cond, self.then_cmd, self.else_cmd],
   rSetChild := rSetChildFields("cond", "then_cmd", "else_cmd"),

   free := self >> Union(self.cond.free(), self.then_cmd.free(), self.else_cmd.free()),

   print := (self, i, si) >>
       Print(self.__name__, "(", self.cond, ",\n",
         Blanks(i+si), self.then_cmd.print(i+si, si), ",\n",
         Blanks(i+si), self.else_cmd.print(i+si, si), "\n",
         Blanks(i), ")")
));

Class(DOWHILE, Command, rec(
  __call__ := (self, cond, then_cmd) >> WithBases(self,
           rec(operations := CmdOps,
           then_cmd   := Checked(IsCommand(then_cmd), then_cmd),
           cond       := toExpArg(cond))),

   rChildren := self >> [self.cond, self.then_cmd],
   rSetChild := rSetChildFields("cond", "then_cmd"),

   free := self >> Union(self.cond.free(), self.then_cmd.free()),

   print := (self, i, si) >>
       Print(self.__name__, "(", self.cond, ",\n",
         Blanks(i+si), self.then_cmd.print(i+si, si), "\n",
         Blanks(i), ")")
));

Class(WHILE, Command, rec(
  __call__ := (self, cond, then_cmd) >> WithBases(self,
           rec(operations := CmdOps,
           then_cmd   := Checked(IsCommand(then_cmd), then_cmd),
           cond       := toExpArg(cond))),

   rChildren := self >> [self.cond, self.then_cmd],
   rSetChild := rSetChildFields("cond", "then_cmd"),

   free := self >> Union(self.cond.free(), self.then_cmd.free()),

   print := (self, i, si) >>
       Print(self.__name__, "(", self.cond, ",\n",
         Blanks(i+si), self.then_cmd.print(i+si, si), "\n",
         Blanks(i), ")")
));


Class(multi_if, ExpCommand, rec(
    __call__ := arg >> let(
        self := arg[1],
        args := Cond( Length(arg)=2 and IsList(arg[2]), arg[2], Drop(arg,1)),
        Cond( # Length(args)=0, skip(), #NOTE: code in autolib's _genPlan relies on this and messing with 'args' directly
              Length(args)=1, toExpArg(args[1]),
              WithBases(self, rec(
                  operations := CmdOps,
                  args := List(args, toExpArg))))),

   print := (self, i, si) >> Print(self.__name__, "(\n",
       DoForAll([1..Length(self.args)], j ->
           Cond( IsOddInt(j), Print(Blanks(i+si), self.args[j]),
                              Print(", ", self.args[j].print(i+si+si, si), When(j<>Length(self.args), ",\n")))),
       Print("\n", Blanks(i), ")"))


));


#F program(<cmd1>, <cmd2>, ...) - top level collection of commands,
#F   usually each cmd is decl, func or struct
#F
Class(program, multiwrap);

Class(trycatch, multiwrap);
Class(tryfinally, multiwrap);

#F func(<ret>, <id>, <params>, <cmd>)
#F   ret    - return type
#F   id     - function name string
#F   params - list of parameters (typed vars)
#F   cmd    - function body
#F
Class(func, Command, rec(
    __call__ := (self, ret, id, params, cmd) >> WithBases(self, rec(
            ret    := Checked(IsType(ret), ret),
            id     := Checked(IsString(id), id),
            params := Checked(IsList(params), params),
            cmd    := Checked(IsCommand(cmd), cmd),
            operations := CmdOps)),

    free := self >> Difference(self.cmd.free(), self.params),
    rChildren := self >> [self.ret, self.id, self.params, self.cmd],
    rSetChild := rSetChildFields("ret", "id", "params", "cmd"),

    print := (self, i, si) >> Print(self.__name__, "(", self.ret, ", \"", self.id, "\", ", self.params, ", \n",
        Blanks(i+si), self.cmd.print(i+si, si), "\n", Blanks(i), ")", self.printA()),

    # we have to handle vector values differently, since we only want to count unique ones, but they may be masqueraded by vparam or as integer constants in fpmuls
    countOps := meth(self, countrec)
        local count, reccountrec, vals, vparams, fpmuls;
        if self.id = "transform" and Last(countrec.ops) = Value then
            reccountrec := Copy(countrec);
            reccountrec.ops := DropLast(countrec.ops, 1);
            count := self.cmd.countOps(reccountrec);
            vals := Set(List(Collect(self.cmd, @@(1, Value, (e,cx)->ObjId(e.t)=TVect or
                (IsBound(cx.Value) and cx.Value=[] and ObjId(e.t)=AtomicTyp and e.t.name="TReal" and IsFloat(e.v)))), i->i.v));
            vparams := Set(List(Collect(self.cmd, @(1, spiral.platforms.vparam, e->IsList(e.p) and ForAll(e.p, IsString))), i->i.p));
            fpmuls := Set(Filtered(List(Collect(self.cmd, fpmul), i-> i.args[2]), IsValue));
            Add(count, Length(vals)+Length(vparams)+Length(fpmuls));
            return count;
        else
            return self.cmd.countOps(countrec);
        fi;
    end

));

Class(func_ppe, func);

#
## define
#
# this command is used to define new types in the unparsed code.
# if you need a
#
# typedef struct { ... }
#
# this is it.
#
Class(define, Command, rec(
    __call__ := (self, types) >> WithBases(self, rec(
        types := types,
        operations := CmdOps
    )),

    rChildren := self >> [self.types],
    rSetChild := rSetChildFields("types"),

    print := (self, i, si) >> Print(
        self.__name__, "(", self.types, ")"
    )
));

#
## IfDef
#
# A #if statement to control execution in code. Used for debugging -- getting counts for only
# certain kernels, etc.
#
Class(IfDef, Command, rec(
    __call__ := (self, cond, cmd) >> WithBases(self, rec(
        cond := Checked(IsString(cond), cond),
        operations := CmdOps,
        cmd := Checked(IsCommand(cmd), cmd)
    )),

    rChildren := self >> [self.cond, self.cmd],
    rSetChild := rSetChildFields("cond", "cmd"),

    print := (self, i, si) >> Print(
        self.__name__, "(\"", self.cond, "\", ", self.cmd, ")"
    ),
));

Class(Define, Command, rec(
    __call__ := (self, var, exp) >> WithBases(self, rec(
        operations := CmdOps,
        var := var,
        exp := exp
    )),
    rChildren := self >> [self.exp, self.var],
    rSetChild := rSetChildFields("exp", "var"),
    print := (self, i, si) >> Print(
        self.__name__, "(", self.var, ", ", self.exp, ")")
));

#-----------------------------------------------------------------------------
#F Ind()
#F Ind(<range>)  -- <range> must be an integer or symbolic integer that will
#F                  imply the range of variable of [0 .. <range>-1]
#F NB: we do not set isLoopIndex attribute below, because Ind() is now
#F     used in Lambda's and some other places which are not loops
#F     NOTE?
#F
Ind := arg -> Cond(
   Length(arg)=0, var.fresh_t("i", TInt),
   Length(arg)=1,
       Cond(arg[1]=TInt, var.fresh_t("ii", TInt),
                         var.fresh("i", TInt, toRange(arg[1]))),
   Error("Usage: Ind() | Ind(<range>)")
);

IndNR := () -> var.fresh_t("i", TInt);
IntVar := pfx -> var.fresh_t(pfx, TInt);
DataInd := (type, range) -> var.fresh("k", type, toRange(range));
TempVec := type -> var.fresh_t("T", type);
Dat := type -> var.fresh_t("D", type);
Dat1d := (type,nentries) -> var.fresh_t("D", TArray(type, nentries));
Dat2d := (type,rows,cols) -> var.fresh_t("D", TArray(TArray(type, cols), rows));
Dat3d := (type,planes,rows,cols) -> var.fresh_t("D", TArray(TArray(TArray(type, cols), rows), planes));
TempVar := type -> var.fresh_t("t", type);


IsLoop := x->IsRec(x) and IsBound(x.isLoop) and x.isLoop;
IsUnrollableLoop := x->IsRec(x) and IsBound(x.isLoop) and x.isLoop and x.__name__ <> "dist_loop";
IsChain := x->IsRec(x) and IsBound(x.isChain) and x.isChain;


IsLoopIndex := v -> IsRec(v) and IsBound(v.isLoopIndex) and v.isLoopIndex;
IsParallelLoopIndex := v -> IsRec(v) and IsBound(v.isParallelLoopIndex) and v.isParallelLoopIndex;
IndPar := idx -> Ind(idx).setAttr("isParallelLoopIndex");


#F FlattenCode(<code>) . . . . . . . . . . . flattens nested chain commands
#F
FlattenCode := c -> SubstBottomUp(c, @.cond(x->IsChain(x) or ObjId(x)=unroll_cmd), e -> e.flatten());

#F FlattenCode2(<code>) . . . same as FlattenCode, but also replaces chain(c) by c
#F
FlattenCode2 := c -> SubstBottomUpRules(c, [
    [@.cond(x->IsChain(x) or ObjId(x)=unroll_cmd), e -> e.flatten(), "flatten1"],
    [[chain, @(1)], e->e.cmds[1], "flatten2"]
]);

#F FlattenCode0(<code>) . . same as FlattenCode, but avoids unroll_cmd
#F
FlattenCode0 := c -> SubstBottomUp(c, @.cond(x->IsChain(x)), e -> e.flatten());


#F UnrollCode(<code>) . . . . . . fully unrolls <code> without optimization
#F   SubstBottomUp works faster than SubstTopDown as it unrolls innermost loops first
#F   but it doesn't work when loop domain depends from outer loop variable.
UnrollCode := c -> let(
    buc := SubstBottomUp(c, @.cond(x -> IsLoop(x) and not IsSymbolic(x.range)), e->e.unroll()),
    tdc := SubstTopDown(buc, @.cond(IsLoop), e->e.unroll()),
    SubstBottomUp(tdc, virtual_var, e->e.ev())
);


#F ArithCostCode(<code>)  . . . . . . . returns a list [num_adds, num_muls]
#F
ArithCostCode := c -> let(
    ops := List([Collect(c, add), Collect(c, sub), Collect(c, mul)],
	        lst -> Sum(lst, x->Length(x.args)-1)),
    [ops[1]+ops[2], ops[3]]);


#F SubstVars(<expr>, <bindings>)
#F
#F Evaluates several variables in <expr> to their values given in <bindings>.
#F <bindings> should be a record or a table of the form:
#F   rec( var1 := value1, ...)  OR
#F   tab( var1 := value1, ...)
#F
SubstVars := function (expr, bindings)
    return SubstLeaves(expr, @(200, var, e -> IsBound(bindings.(e.id))),
    e -> bindings.(e.id));
end;

SubstVarsEval := function (expr, bindings)
    return SubstBottomUp(expr, @,
    e -> Cond(IsVar(e), bindings.(e.id), IsExp(e), e.eval(), V(e)));
end;
