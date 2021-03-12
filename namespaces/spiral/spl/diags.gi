
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F DiagFunc - base class for diagonal generating functions
#F Required fields in subclasses:
#F    .abbrevs = [ (p1)     -> [parlist1],
#F                 (p1,...) -> [parlist2] ]
#F       Mapping from various lengths of parameter lists to a
#F       fixed signature .def
#F
#F    .lambda(self)  - must return Lambda() object representing the actual
#F                     function. One should use .params field here
#F
#F    .domain(self) -- must return the domain of the function (integer or type)
#F    .range(self)  -- must return the range of the function (integer or type)
#F
#F Optional:
#F    .def(self, p1, ...)  - this method is called from the constructor 
#F                           before the instance object is created. 
#F                           <self> = class in this case
#F                           .def() must return a record, all fields of the record
#F                           will be copied to the resulting instance object,
#F                           which will also have the field .params := [p1, ...].
#F                           additional error checking can be performed in .def
#F                           NB: parameter lists are normalized here, using .abbrevs
Class(DiagFunc, Function, Sym, rec(
    isSPL := false,

    range  := self >> Error(ObjId(self), " does not implement .range()"), 
    domain := self >> Error(ObjId(self), " does not implement .domain()"), 

    tensor_op := (v,rows,cols,fl,gl) ->
        Lambda(v, fl.at(idiv(v, cols)) * gl.at(imod(v, cols))),

    sums := self >> self,
    rSetChild := meth(self, n, newChild)
        self.params[n] := newChild;
    end,

    print := meth(self, i, is)
        local params;
        params := let(p:=self.params, When(IsList(p), p, [p]));
        Print(self.name, "(",
          DoForAllButLast(params, x -> Print(x, ", ")),
          Last(params), ")");
    end,

    def := arg -> rec(),

    fromDef := meth(arg)
        local result, self, params, h, lkup;
        self := arg[1];
        params := arg{[2..Length(arg)]};
        params := self.canonizeParams(params);
        self.checkParams(params);
        params := When(IsList(params), params, [params]);

        h := self.hash;
        if h<>false then
            lkup := h.objLookup(self, params);
            if lkup[1] <> false then return lkup[1]; fi;
        fi;

        result := SPL(WithBases(self, CopyFields(
                    rec(params := params),
                    ApplyFunc(self.def,params))));

        if h<>false then return h.objAdd(result, lkup[2]);
        else return result;
        fi;
    end,

    free := self >> Union(List(self.params, FreeVars)),
    __call__ := ~.fromDef,

    isReal := self >> not IsComplexT(self.range())
));

#F II - interval diagonal function
#F
#F II(<N>) - diagonal of N 1's
#F II(<N>, <to>) - diagonal with 1's from 0..<to>, and 0's after
#F II(<N>, <from>, <to>) - diagonal with 1's in <from>..<to>, 0's elsewhere
#F
Class(II, DiagFunc, rec(
    abbrevs := [ (N)         -> [N,0,N],
                 (N,to)      -> [N,0,to],
                 (N,from,to) -> [N,from,to] ],

    def := (N, from, to) -> Checked(
	IsPosInt0Sym(N), IsPosInt0Sym(from), IsPosInt0Sym(to), rec()),

    lambda := self >> let(
        i := Ind(self.params[1]), from := self.params[2], to := self.params[3],
        Lambda(i, cond(leq(from, i, to-1), 1, 0))),

    domain := self >> self.params[1],
    range := self >> TInt, 
));

#F fConst(<t>, <N>, <c>) - constant diagonal function, <N> values of <c> of type <t>
#F
Class(fConst, DiagFunc, rec(
    abbrevs := [   (N,c) -> Checked(           IsPosInt0Sym(N), [TReal, N,c]),
                 (t,N,c) -> Checked(IsType(t), IsPosInt0Sym(N), [t, N, c]) ],
    range := self >> self.params[1],
    domain := self >> self.params[2], 
    lambda := self >> let(i := Ind(self.params[2]), Lambda(i, self.params[3]))
));

#F fUnk(<t>, <N>) - dummy diagonal that denotes an "unknown" function, <N> values of type <t>
#F   To be used for .hashAs methods of transforms
#F
Class(fUnk, DiagFunc, rec(
    abbrevs := [ N -> Checked(IsIntSym(N), [TReal, N]), 
	        (t, N) -> Checked(IsType(t), IsIntSym(N), [t, N]) ],
    range := self >> self.params[1],
    domain := self >> self.params[2],
    lambda := self >> let(N:=self.params[2], i:=Ind(N), 
	Cond(IsIntT(self.params[1]), Lambda(i, i+1), 
	                             Lambda(i, self.params[1].one()/N*(i+1))))
));

# FUnk is for backwards compatibility, use fUnk
FUnk := n -> fUnk(TReal, n);

Declare(LD);

# UD(<N>, <shift>) - upper diagonal
#
Class(UD, Sym, rec(
    def := (n,k) -> Checked(k < n, Z(n,k) * Diag(II(n, k, n))),
    transpose := self >> ApplyFunc(LD, self.params)
));

# LD(<N>, <shift>) - lower diagonal
#
Class(LD, Sym, rec(
    def := (n,k) -> Checked(k < n, Z(n,n-k) * Diag(II(n, 0, n-k))),
    transpose := self >> ApplyFunc(UD, self.params)
));

# [ 1 1
#     1 1
#       ... ]
Class(S, Sym, rec(
    def := n -> SUM(I(n), UD(n,1))));


# Class(fComputeOnline, PlaceholderExp);
Class(fComputeOnline, FuncClassOper, rec(
    range  := self >> self._children[1].range(),
    domain := self >> self._children[1].domain(),
    lambda := self >> self._children[1].lambda(),
    tolist := self >> self._children[1].tolist(),
    free   := self >> self._children[1].free(),
    ev     := self >> self._children[1].ev(),
    eval   := self >> self._children[1].eval(),
));

Declare(fPrecompute);

# fPrecompute(<func>)
#
# Marks functions that should be precomputed by either generating a
# table (opts.generateInitCode := true) or runtime initialization code
# (opts.generateInitCode := true).
#
# Actual precomputation is done by Process_fPrecompute. Which uses
# options record 'opts' to determine the precomputation strategy.
#
Class(fPrecompute, FuncClassOper, rec(
    def    := func -> rec(t := TPtr(func.range())),
    range  := self >> self._children[1].range(),
    domain := self >> self._children[1].domain(),
    lambda := self >> self._children[1].lambda(),
    tolist := self >> self._children[1].tolist(),
    free   := self >> self._children[1].free(),
    isReal := self >> self._children[1].isReal()
));


# diagAdd(<f1>, <f2>,...) : i -> f1(i) + f2(i) + ...
# Pointwise addition of several diagonal functions
Class(diagAdd, FuncClassOper, rec(
    domain := self >> self.child(1).domain(),
    range  := self >> UnifyTypes(List(self.children(), x->x.range())),
    lambda := self >> let(v := Ind(self.domain()), 
	Lambda(v, ApplyFunc(add, List(self.children(), i->i.at(v))))),
));

# diagMul(<f1>, <f2>,...) : i -> f1(i) * f2(i) * ...
# Pointwise multiplication of several diagonal functions
Class(diagMul, FuncClassOper, rec(
    domain := self >> self.child(1).domain(),
    range  := self >> UnifyTypes(List(self.children(), (x) -> x.range())),
    lambda := self >> let(v := Ind(self.domain()), 
	Lambda(v, ApplyFunc(mul, List(self.children(), i->i.at(v))))),
));

Class(diagCond, FuncClassOper, rec(
    domain := self >> self.child(2).domain(),
    range  := self >> UnifyTypes(List(Drop(self.children(), 1), x->x.range())),
    lambda := self >> let(v := Ind(self.domain()), 
	Lambda(v, cond(self.child(1), self.child(2).at(v), self.child(3).at(v))))
));

#F fCast(<toT>, <fromT>)  -- symbolic type cast function 
#F
#F Example: fCast(TReal, TInt)
#F
Class(fCast, DiagFunc, rec(
    abbrevs := [ (toT, fromT)  -> Checked(IsType(toT), IsType(fromT), [toT, fromT]) ], 
    range := self >> self.params[1],
    domain := self >> self.params[2],
    lambda := self >> let(i:=TempVar(self.domain()), Lambda(i, tcast(self.range(), i))), 
));
