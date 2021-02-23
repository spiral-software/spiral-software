# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(Function, rec(
   isFunction := true,

   range := self >> self._range,
   tolist := self >> self.lambda().tolist(),

   lambda := self >> Error("Not implemented"),

   free   := self >> self.lambda().free(),
   domain := self >> self.lambda().domain(),
   at := (self, pos) >> self.lambda().at(pos),

   from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),

   _domain := false,
   _range := false,
   setRange := meth(self, range)
       self._range := range;
       return self;
   end,
   setDomain := meth(self, domain)
       self._domain := spiral.code.RulesStrengthReduce(domain);
       return self;
   end,

   # NOTE: below is a hack that already causes problems if we compute w/ symbolic higher order functions
   # 2 methods below allow to use constant functions (ie domain=1) as expressions

   # NOTE: don't remember where the hack above is used, so I removed it, now need to test what happens
   eval := self >> self, #When(self.domain() = 1 and self.rank() = 0, self.at(0), self),
   ev := self >> self,   #Checked(self.domain() = 1, self.at(0).ev()),
));

IsFunction := x -> IsRec(x) and IsBound(x.isFunction) and x.isFunction;

Declare(Lambda, LambdaOps, FList, FData);

#F Lambda(<vars>, <expr>)  - symbolic representation of a function mapping vars -> expr
#F
#F <vars> must be a list of variables or a single variable
#F <expr> is a symbolic expression using variables from <vars>
#F
Class(Lambda, Function, rec(
   rChildren := self >> [self.vars, self.expr, self._domain, self._range],
   rSetChild := rSetChildFields("vars", "expr", "_domain", "_range"),
   isLambda := true,
   lambda := self >> self,

   from_rChildren := (self, rch) >> ObjId(self)(rch[1], rch[2]).setDomain(rch[3]).setRange(rch[4]),

   domain := self >> When(self._domain <> false,
       self._domain,
       let(v := Last(self.vars), # ignore rank-associated vars
           When(IsBound(v.range) and not IsMeth(v.range), v.range, v.t))),

   # FF: need to be able to override range
   range := self >> When(self._range <> false, self._range, self.expr.t),

   tolist := self >> Checked(Length(self.vars)=1, let(dom := _unwrapV(self.domain()),
       Cond(IsType(dom), Error("Can't convert <self> to a list, ",
	       "because its .domain() is not a bounded interval"),
	    IsSymbolic(dom), Error("Can't convert <self> to a list, ",
	       "because its .domain() interval size is not known"),
	    List([0 .. dom-1], r -> self.at(r).eval())))),

   free := self >> Difference(self.expr.free(), self.vars),

   # FF: need to be able to override range
   print := self >> Print(self.name, "(", self.vars, ", ", self.expr, ")",
        When(self._range <> false, Print(".setRange(", self._range, ")"), ""),
        When(self._domain <> false, Print(".setDomain(", self._domain, ")"), "")
   ),

   # at(args), strict version, must have Length(args) = Length(vars)
   at := meth(arg)
       local self, args, i, bind;
       self := arg[1];
       args := Drop(arg, 1);
       # support variable-length call through lists; required by OL
       if IsList(args[1]) then args := args[1]; fi;

       Constraint(Length(args) = Length(self.vars));
       bind := tab();
       for i in [1..Length(self.vars)] do
           if IsBound(args[i]) then
               bind.(self.vars[i].id) := toExpArg(args[i]);
           fi;
       od;
       return SubstVars(Copy(self.expr), bind);
    end,

    # relaxed_at(lvars, v), relaxed version of at() for multi-ranks, allows lvars to be of any length
    # equivalent to at(Concatenation(lvars, [v])), if Length(lvars)=Length(self.vars)-1
    # v is the "standard" variable, everything in lvars is a loop index
    relaxed_at := meth(self, lvars, v)
       local i, bind, la, lv, args, vars;
       args := Concatenation(lvars, [v]);

       # for multi-rank functions loop indices can stay unbound
       vars := self.vars;
       [la, lv] := [Length(args), Length(vars)];
       if   la > lv then args := Drop(args, la - lv);
       elif lv > la then vars := Drop(vars, lv - la);
       fi;

       bind := tab();
       for i in [1..Length(vars)] do
           bind.(vars[i].id) := toExpArg(args[i]);
       od;
       return SubstVars(Copy(self.expr), bind);
    end,

#    computeType := self >> ApplyFunc(TFunc, Concatenation(
#           List(self.vars, x ->
#               When(IsBound(x.range) and not IsMeth(x.range), x.range, x.t)),
#           [self.expr.t])),

    computeType := self >> ApplyFunc(TFunc,
           List(self.vars, x ->
                Cond(IsBound(x.range) and not IsMeth(x.range), x.range,
                     Collect(self.expr, x)=[], TDummy,
                     x.t)
           ) :: [self.expr.t]),

    normalize := self >> let(
	t := self.computeType(),
	first_non_dummy := PositionProperty(DropLast(t.params,2), x->x<>TDummy),
	k := Cond(first_non_dummy=false, Length(self.vars), first_non_dummy),
	Lambda(self.vars{ [k .. Length(self.vars)]}, 
	       self.expr)),

    forCmp := meth(self)
        local map, i, t;
        map := tab();
        for i in [1..Length(self.vars)] do
            t := self.vars[i].t;
            map.(self.vars[i].id) := param(When(IsType(t), t, InferType(t)), "v"::StringInt(i));
        od;
        return SubstParamsCustom([self.vars, self.expr], map, [var]);
    end, 

    __call__ := meth(self, vars, expr)
        local usage, res, rank, loopvars;
        usage := Concatenation("Usage: Lambda(<argList>, <expr>)\n",
                            "Usage: Lambda(<arg>, <expr>)");
            if not IsList(vars) then vars := [vars]; fi;
        if not ForAll(vars, x -> IsVar(x))
            then Error("all elements in <argList> must be variables"); fi;
        if Length(vars) <> Length(Set(vars))
            then Error("<argList> should not contain any duplicates"); fi;

        expr := toExpArg(expr);
            rank := _rank(expr);
            # if <expr> is ranked, add extra arguments to lambda
            if rank > 0 and Length(vars)=1 then
                loopvars := List([1..rank], i->var.fresh_t("q", TInt));
                vars := Concatenation(loopvars, vars);
                expr := _downRankFull(expr, loopvars);
            fi;
        res := WithBases(self, rec(expr := expr, vars := vars, operations := LambdaOps));
        res.t := res.computeType();
	# NOTE: the below line should not be necessary, not sure what the purpose was
        # if IsExp(Last(res.vars).range) then res:=res.setDomain(Last(res.vars).range); fi;
        return res;
    end,

    downRank := meth(self, loopid, ind)
        local v, vars, exp;
        if Length(self.vars) <= loopid then
            return self;
        else
            v    := self.vars[ Length(self.vars) - loopid ];
            vars := ListWithout(self.vars, Length(self.vars)-loopid);
            exp  := SubstVars(Copy(self.expr), tab((v.id) := ind));
            return ObjId(self)(vars, exp);
        fi;
    end,
));

IsLambda := obj -> IsRec(obj) and IsBound(obj.isLambda) and obj.isLambda = true;

CompatLambdas := (x, y) -> Length(x.vars) = Length(y.vars);

LambdaCompose := function(l1, l2)
    if ObjId(l1)<>Lambda then l1 := l1.lambda(); fi;
    if ObjId(l2)<>Lambda then l2 := l2.lambda(); fi;
    Constraint(Length(l1.vars)=1); Constraint(Length(l2.vars)=1);
    return let(v:=l2.vars[1].clone(), Lambda(v, l1.at(l2.at(v))));
end;

Class(LambdaOps, rec(
    Print := x->x.print(),
    \+ := (l1, l2) -> let(ll1:=l1.lambda(), ll2:=l2.lambda(), v:=l1.vars[1].clone(),
    Lambda(v, ll1.at(v) + ll2.at(v))),
    \- := (l1, l2) -> let(ll1:=l1.lambda(), ll2:=l2.lambda(), v:=l1.vars[1].clone(),
    Lambda(v, ll1.at(v) - ll2.at(v))),
    \* := (l1, l2) -> let(ll1:=l1.lambda(), ll2:=l2.lambda(), v:=l1.vars[1].clone(),
    Lambda(v, ll1.at(v) * ll2.at(v))),
    \/ := (l1, l2) -> let(ll1:=l1.lambda(), ll2:=l2.lambda(), v:=l1.vars[1].clone(),
    Lambda(v, ll1.at(v) / ll2.at(v))),
    \^ := (l1, l2) -> let(ll1:=l1.lambda(), ll2:=l2.lambda(), v:=l1.vars[1].clone(),
    Lambda(v, ll1.at(v) ^ ll2.at(v))),
    \= := (l1, l2) -> IsLambda(l1) and IsLambda(l2) and l1.lambda().forCmp() = l2.lambda().forCmp(),
    \< := (l1, l2) -> When( IsLambda(l1) and IsLambda(l2),
                              l1.lambda().forCmp() < l2.lambda().forCmp(),
                              BagAddr(l1) < BagAddr(l2) ),
));


#F FDataOfs(<datavar>, <len>, <ofs>)
#F
Class(FDataOfs, Function, rec(
    __call__ := (self, datavar, len, ofs) >> WithBases(self, rec(
    var := datavar,
    operations := PrintOps,
    ofs := toExpArg(ofs),
    len := Checked(IsPosIntSym(len), len)
    )),

# <-Daniele's changes
#    rChildren := self >> [ self.var, self.len, self.ofs],
#    rSetChild := rSetChildFields("var", "len", "ofs"),

    rChildren := self >> [ self.var, self.len, self.ofs, self._domain, self._range],
    rSetChild := rSetChildFields("var", "len", "ofs", "_domain", "_range"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], rch[2], rch[3]).setDomain(rch[4]).setRange(rch[5]),
# ->

    domain := self >> self.len,
    print := self >> Print(self.name,"(",self.var,", ",self.len,", ",self.ofs,")"),

    at := (self, n) >> When(IsInt(n) and IsValue(self.ofs) and IsBound(self.var.value),
        self.var.value.v[n + self.ofs.v + 1],
        nth(self.var, n + self.ofs)),

    tolist := self >> List([0..EvalScalar(self.len-1)], i -> nth(self.var, self.ofs+i)),
    lambda := self >> let(x := Ind(self.domain()), Lambda(x, nth(self.var, self.ofs+x))),

    domain := self >> self.len,
    range := self >> When(self._range=false, self.var.t.t, self._range),
    inline := true,
    free := self >> self.ofs.free()
));

#F FData(<datavar>) -- symbolic function i -> datavar[i],
#F
#F domain = datavar.range
#F range = datavar.t
#F
#F
Class(FData, Function, rec(
   __call__ := arg >> let(
       self := arg[1],
       _val := Cond(Length(arg)=2, When(IsList(arg[2]), arg[2], [arg[2]]),
                               Drop(arg, 1)),
       val := When(Length(_val)=1 and IsLoc(_val[1]), _val[1], V(_val)),
       datavar := Cond(IsLoc(val), val,
                   Dat(val.t).setValue(val)),
       WithBases(self, rec(var := datavar, operations := PrintOps))),

   print := self >> Print(self.name, "(", self.var, ")"),
   rChildren := self >> [ self.var ],
   rSetChild := rSetChildFields("var"),

   at := (self, n) >> When(IsInt(n), self.var.value.v[n+1], self.lambda().at(n)),
   tolist := self >> When(IsBound(self.var.value), self.var.value.v, self.lambda().tolist()),
   lambda := self >> let(x := Ind(self.domain()), Lambda(x, nth(self.var, x))),

   domain := self >> self.var.t.size,
   range := self >> self.var.t.t,

   inline := true,
   free := self >> Set([]),
   part := (self, len, ofs) >> FDataOfs(self.var, len, ofs),
));

Class(RCData, Function, rec(
   __call__ := (self, func) >> WithBases(self, rec(
       func := Checked(IsFunction(func) or IsFuncExp(func), func),
       operations := PrintOps)),

   print := self >> Print(self.name, "(", self.func, ")"),
   rChildren := self >> [ self.func ],
   rSetChild := rSetChildFields("func"),

   at := (self, n) >> cond(n mod 2, im(self.func.at(idiv(n,2))), re(self.func.at(idiv(n,2)))),
   tolist := self >> ConcatList(self.func.tolist(), x->[re(x), im(x)]),

   domain := self >> let(d:=self.func.domain(),
       Cond(not IsType(d), d * 2,
            d=TInt,        TInt,
            Error("RCData: Invalid self.func.domain()"))),

   range := self >> self.func.range().realType(),
   # this .lambda corrently handles the case of multirank self.func
   lambda := self >> let(
       x := Ind(self.domain()),
       f := self.func.lambda(),
       jv := DropLast(f.vars, 1),
       Lambda(Concatenation(jv, [x]),
              cond(neq(imod(x, 2),0), im(f.relaxed_at(jv, idiv(x,2))), re(f.relaxed_at(jv, idiv(x,2)))))),

   inline := true,
   free := self >> self.func.free()
));

#F
#F CRData is the opposite of RCData: takes real function which assumed to be interleaved complex data
#F   and packs it into complex function: 
#F   [re0, im0, re1, im1, ... ] -> [ (re0, im0), (re1, im1), ... ]
#F

Class(CRData, RCData, rec(
   
   at     := (self, n) >> cxpack(self.func.at(2*n), self.func.at(2*n+1)),
   tolist := self >> let( l := self.func.tolist(), List([1..Length(l)/2], i->cxpack(l[2*i-1], l[2*i]))),
   
   lambda := self >> let(
       x  := Ind(self.domain()),
       f  := self.func.lambda(),
       jv := DropLast(f.vars, 1),
       Lambda(Concatenation(jv, [x]), cxpack(f.relaxed_at(jv, 2*x), f.relaxed_at(jv, 2*x+1)))),


   domain := self >> let(d:=self.func.domain(),
       Cond(not IsType(d), div(d, 2),
            d=TInt,        TInt,
            Error("CRData: Invalid self.func.domain()"))),

   range := self >> self.func.range().complexType(),
));

Declare(FConj, FInv, FRConj);

Class(FConj, Function, rec(
   __call__ := (self, func) >> When(ObjId(func)=FConj, func.func, WithBases(self, rec(
       func := Checked(IsFunction(func) or IsFuncExp(func), func),
       operations := PrintOps))),

   print := self >> Print(self.name, "(", self.func, ")"),
   rChildren := self >> [ self.func ],
   rSetChild := rSetChildFields("func"),

   at := (self, n) >> conj(self.func.at(n)),
   tolist := self >> List(self.func.tolist(), conj),

   domain := self >> self.func.domain(),
   range := self >> self.func.range(),
   lambda := self >> let(x := Ind(self.domain()), Lambda(x, self.at(x))),

   inline := true,
   free := self >> self.func.free()
));

Class(FInv, Function, rec(
   __call__ := (self, func) >> When(ObjId(func)=FInv, func.func, WithBases(self, rec(
       func := Checked(IsFunction(func) or IsFuncExp(func), func),
       operations := PrintOps))),

   print := self >> Print(self.name, "(", self.func, ")"),
   rChildren := self >> [ self.func ],
   rSetChild := rSetChildFields("func"),

   at := (self, n) >> 1/(self.func.at(n)),
   tolist := self >> List(self.func.tolist(), x -> 1/x),

   domain := self >> self.func.domain(),
   range := self >> self.func.range(),
   lambda := self >> let(x := Ind(self.domain()), Lambda(x, self.at(x))),

   inline := true,
   free := self >> self.func.free()
));

# conjugation of N complex entries stored as 2*N reals r,i,r,i,...
Class(FRConj, Function, rec(
   __call__ := (self, func) >> When(ObjId(func)=FRConj, func.func, WithBases(self, rec(
       func := Checked(IsFunction(func) or IsFuncExp(func), func),
       operations := PrintOps))),

   print := self >> Print(self.name, "(", self.func, ")"),
   rChildren := self >> [ self.func ],
   rSetChild := rSetChildFields("func"),

   at := (self, n) >> cond(n mod 2, neg(self.func.at(n)), self.func.at(n)),
   tolist := self >> let(ll := self.func.tolist(),
       List([0..Length(ll)-1], i -> When(i mod 2 = 1, neg(ll[1+i]), ll[1+i]))),

   domain := self >> self.func.domain(),
   range := self >> self.func.range(),
   lambda := self >> let(x := Ind(self.domain()), Lambda(x, self.at(x))),

   inline := true,
   free := self >> self.func.free()
));

#F FList(<type>, <list>) -- symbolic function i -> list[i],
#F
#F Example: FList(TInt, [1,2,3])
#F See also: FData
Class(FList, Function, rec(
   __call__ := (self, t, list) >> Checked(IsList(list), WithBases(self, rec(
       list := list,
           t := t, operations := PrintOps))),

   print := self >> Print(self.name, "(", self.t, ", ", self.list, ")"),
   rChildren := self >> [self.t, self.list],
   rSetChild := rSetChildFields("t", "list"),

   domain := self >> Length(self.list),
   range  := self >> self.t,

   at := (self, n) >> When(IsInt(n), self.list[n+1], self.lambda().at(n)),

   _typ := self >> When(IsInt(self.t), TInt, self.t),
   tolist := self >> List(self.list, e->When(not IsSymbolic(e), Value(self._typ(), e), e)),
   tolistval := self >> Value(TArray(self._typ(), Length(self.list)), self.tolist()),
   lambda := self >> let(x := Ind(self.domain()), Lambda(x, nth(self.tolistval(), x))),

   todata := self >> FData(self.list),
   inline := false,
   free := self >> Union(List(self.list, FreeVars))
));



FF := x -> Cond(IsRec(x) and IsBound(x.isFunction), x,
                IsList(x), FList(x),
                Error("<x> must be a function or a list"));

#F FPerm(<perm>) -- function representing permutation
#F
#F Example: FPerm(Perm((1,2,3),4))
Class(FPerm, Function, rec(
   __call__ := (self, perm) >> WithBases(self, rec(
           perm := perm,
           operations := PrintOps)),

   print := self >> Print(self.name, "(", self.perm, ")"),
   rChildren := self >> [self.perm],
   rSetChild := rSetChildFields("perm"),

   domain := self >> self.perm.dims()[1],
   range  := self >> self.perm.dims()[1],

   at := (self, n) >> When(IsInt(n), self.list[n+1], self.lambda().at(n)),

   _typ := self >> TInt,
   tolist := self >> List(self.list, e->When(not IsSymbolic(e), Value(self._typ(), e), e)),
   tolistval := self >> Value(TArray(self._typ(), Length(self.list)), self.tolist()),
   lambda := self >> self.perm.sums().func.lambda(),

   inline := true,
   free := self >> FreeVars
));