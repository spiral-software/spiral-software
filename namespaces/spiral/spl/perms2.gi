
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


domain := f -> f.domain(); pdomain := p -> p.val.domain();
range  := f -> f.range();  prange := p -> p.val.range();
is_perm := f -> f.isPermutation();

Class(fId, PermClass, rec(
    domain := self >> self.params[1],
    range  := self >> self.params[1],
    def    := size -> Checked(IsPosInt0Sym(size), rec(size := size)),
    lambda := self >> let(i := Ind(self.params[1]), Lambda(i,i)),
    transpose := self >> self,
    isIdentity := True
));

Class(fBase, FuncClass, rec(
    abbrevs := [ var -> Checked(IsVar(var) or ObjId(var)=ind, [var.range, var]) ],
    def    := (N, pos) -> rec(),
    domain := self >> 1,
    range  := self >> self.params[1],
    print  := (self, i, is) >> Print(self.name, "(", 
        When(ObjId(self.params[2]) in [var, ind], Print(self.params[2]), PrintCS(self.params)), ")"),
    lambda := self >> let(i := Ind(1), Lambda(i, self.params[2]))
));

# used for I_n dirsum J_n
# below represent an alternator <I_n | J_n>_x o fBase(j)
# x = (pos+par) mod 2
Class(fXbase, FuncClass, rec(
    def := (n,j,par,pos) -> rec(),
    domain := self >> 1,
    range := self >> self.params[1],
    lambda := self >> let(
	n:=self.params[1], j:=self.params[2], par:=self.params[3],
	pos:=self.params[4], i := Ind(1),
	Lambda(i, j + imod(pos+par, 2)*(n - 1 - 2*j))
    )
));

# used for I_n dirsum (I_1 dirsum J_n-1)
# below represent an alternator <I_n | I_1 dirsum J_n_1>_x o fBase(j)
# x = (pos+par) mod 2
Class(fYbase, FuncClass, rec(
    def := (n,j,par,pos) -> rec(),
    lambda := self >> let(
	n := self.params[1], j := self.params[2], par := self.params[3],
	pos := self.params[4], i := Ind(1),
	Lambda(i, cond(eq(j, 0), 0, j + imod(pos+par, 2)*(n - 2*j)))
    )
));

Class(fDot, FuncClass, rec(
    def    := (v, func) -> Chain(v.setRange(func.domain()), rec()),
    range  := self >> self.params[2].range(),
    domain := self >> self.params[2].domain(),
    lambda := self >> let(
        v    := self.params[1],
        func := self.params[2],
        i    := Ind(func.domain()),
        Lambda(i, SubstVars(func.at(i), tab((v.id) := i)))) 
));

Class(fPlaceholder, FuncClass, rec(
    def := func -> rec(),
    range := self >> self.params[1].range(),
    domain := self >> self.params[1].domain(),
    lambda := self >> self.params[1].lambda(),
    at := (self, n) >> self.params[1].at(n)
));

#F fAdd is the "add constant" index-mapping function
#F fAdd(N, n, k): II_n -> II_N, i->i+k
#F References: "Formal Loop Merging for Signal Transforms"
Class(fAdd, FuncClass, rec(
    def := (N, n, val) -> rec(), 
    range  := self >> self.params[1],
    domain := self >> self.params[2],
    lambda := self >> let(i := Ind(self.params[2]), Lambda(i, i + self.params[3]))
));

#F fSub is the "subtract constant" index-mapping function
#F fSub(N, n, k): II_n -> II_N, i->i-k
#F References: "fAdd" function
Class(fSub, FuncClass, rec(
    def := (N, n, val) -> rec(N := N, n := n),
    lambda := self >> let(i := Ind(self.params[2]), Lambda(i, i - self.params[3]))
));

Function.hash := false;

Class(FuncClassOper, Function, rec(
    _perm := true,
    _children := [],

    child := (self, n) >> self._children[n],
    children := self >> self._children,
    numChildren := self >> Length(self._children),

    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),
    rChildren := self >> self._children,
    rSetChild := meth(self, n, what)
        self._children[n] := what;
    end,
    free := self >> Union(List(self.rChildren(), FreeVars)),

    skipOneChild := false,

    advdims   := self >> [ [[ self.range() ]], [[ self.domain() ]] ],
    advrange  := self >> [[ self.range() ]],
    advdomain := self >> [[ self.domain() ]],
    # dimensionality of range (1-d, 2-d, etc)
    advrangeDim := self >> Length(self.advrange()[1]),
    # dimensionality of domain (1-d, 2-d, etc)
    advdomainDim := self >> Length(self.advdomain()[1]),
  
    __call__ := meth(arg)
        local self, children, lkup, res, h;
        self := arg[1];
        children := Flat(Drop(arg, 1));
	Constraint(ForAll(children, c -> IsFunction(c) or IsFuncExp(c)));
        if self.skipOneChild and Length(children)=1 then return children[1]; fi;

        h := self.hash;
        if h<>false then
            lkup := h.objLookup(self, children);
            if lkup[1]<>false then return lkup[1]; fi;
        fi;
        res := WithBases(self, rec(operations := RewritableObjectOps, _children := children));
        if h<>false then return h.objAdd(res, lkup[2]);
        else return res;
        fi;
    end,

    print := self >> Print(self.name, "(", PrintCS(self._children), ")"),
    range := self >> Error("not implemented"), 
    domain := self >> Error("not implemented"), 

    rightBinary := self >> FoldR1(self._children, (p,x) -> let(base:=self.__bases__[1], base(x, p))),
    leftBinary  := self >> FoldL1(self._children, (p,x) -> let(base:=self.__bases__[1], base(p, x))),

    equals := (self, o) >> ObjId(o) = ObjId(self) and self.rChildren() = o.rChildren(),

    lessThan := (self, o) >> Cond(
        ObjId(self) <> ObjId(o), ObjId(self) < ObjId(o), 
        [ ObjId(self), self.rChildren() ] < [ ObjId(o), o.rChildren() ]
    ),
));

FuncClass.tensor_op := (v,range,domain,fl,gl) ->
    Lambda(v, range * fl.at(idiv(v, domain)) + gl.at(imod(v, domain)));

FuncClassOper.tensor_op := FuncClass.tensor_op;

Class(fTensorBase, SumsBase, FuncClassOper, rec(
    domain := self >> Product(self._children, x->x.domain()),

    range := self >> let(cht := List(self._children, x->x.range()),
        Cond(ForAll(cht, x->not IsType(x)), Product(cht),
             ForAll(cht, x->not IsType(x) or x=TInt), TInt,
             Error("self.range() can't handle this combination of self._children ranges"))),

    skipOneChild := true,

    split_op := self >> let(
        c := self.children(), v := Ind(self.domain()),
        c1dom := c[1].domain(), c2dom := c[2].domain(),
        Checked(Length(c)=2,
            [ Cond(c1dom=1, 0, c2dom=1, v, idiv(v, c2dom)),
              Cond(c1dom=1, v, c2dom=1, 0, imod(v, c2dom)),
              v ])),

    lambda := self >> Cond(
	self.numChildren() > 2, 
	    self.rightBinary().lambda(),
        let(split := self.split_op(), 
            f := self.child(1).lambda(), 
            g := self.child(2).lambda(),
            jv := DropLast(When(Length(f.vars) > Length(g.vars), f.vars, g.vars), 1),
            Lambda(Concatenation(jv, [split[3]]),
                   self.combine_op(jv, split, f, g)))),
    
    isIdentity := self >> ForAll(self._children, IsIdentity),
));

Class(fTensor, fTensorBase, rec(
    combine_op := (self, jv, split, f, g) >> 
        self.child(2).range() * f.relaxed_at(jv, split[1]) + g.relaxed_at(jv, split[2]),

    transpose := self >> self.__bases__[1](List(self.children(), c->c.transpose()))
));

Class(diagTensor, fTensorBase, rec(
    print := FuncClassOper.print,
    range := self >> UnifyTypes(List(self.children(), x->x.range())),
    combine_op := (self, jv, split, f, g) >> f.relaxed_at(jv, split[1]) * g.relaxed_at(jv, split[2])
));

Class(gammaTensor, fTensorBase, rec(
    isCyclic := true,
    print := FuncClassOper.print,
    combine_op := (self, jv, split, f, g) >> let(
        r := self.child(1).range(), s := self.child(2).range(),
        imod(s * no_mod(f.relaxed_at(jv,split[1])) + r * no_mod(g.relaxed_at(jv,split[2])), self.range()))
));

_rankedLambdaCompose := function(l1, l2) 
    local jv, v, vars;
    l1.expr := l1.expr.eval(); 
    l2.expr := l2.expr.eval(); 

    vars := When(Length(l1.vars) > Length(l2.vars), l1.vars, l2.vars);
    v := Last(vars);
    jv := vars{[1..Length(vars)-1]};

    return Lambda(vars, l1.relaxed_at(jv, l2.relaxed_at(jv, v)));
end;

Class(fCompose, FuncClassOper, rec(
    domain := self >> Last(self._children).domain(),
    range := self >> self._children[1].range(),
    advdomain := self >> Last(self._children).advdomain(),
    advrange := self >> self._children[1].advrange(),
    skipOneChild := true,

    lambda := self >>
        FoldL1(List(self.children(), z->z.lambda()), _rankedLambdaCompose),

    transpose := self >> self.__bases__[1](List(Reversed(self.children()), c->c.transpose())),

    isIdentity := self >> ForAll(self._children, IsIdentity),
));

Class(fDirsum, FuncClassOper, rec(
    domain := self >> Sum(self._children, x->x.domain()),
    range  := self >> Sum(self._children, x->x.range()),
    skipOneChild := true,

    lambda := meth(self)
        local dom_spans, row_spans, i, conds, v;
        # spans uses 1-based offsets
        dom_spans := BaseOverlap.spans(0, List(self.children(), x->x.domain())) - 1;
        row_spans := BaseOverlap.spans(0, List(self.children(), x->x.range())) - 1;
        conds := [];
        v := Ind(self.domain());
        for i in [1..self.numChildren()] do
            Add(conds, leq(dom_spans[i][1], v, dom_spans[i][2]));
            Add(conds, row_spans[i][1] + self.child(i).lambda().at(v - dom_spans[i][1]));
        od;
        return Lambda(v, ApplyFunc(cond, conds));
    end,

    transpose := self >> self.__bases__[1](List(self.children(), c->c.transpose()))
));

Class(fStack, FuncClassOper, rec(
    domain := self >> Sum(self._children, x->x.domain()),
    # NB: use lowercase max that handles symbolics
    range := self >> ApplyFunc(max, List(self._children, x->x.range())),  
    subdomainsDivisibleBy := (self, n) >> ForAll(self._children, x -> x.domain() mod n = 0),
    skipOneChild := true,

    lambda := meth(self)
        local dom_spans, i, conds, v;
        # spans uses 1-based offsets
        dom_spans := BaseOverlap.spans(0, List(self.children(), x->x.domain())) - 1;
        conds := [];
        v := Ind(self.domain());
        for i in [1..self.numChildren()] do
            Add(conds, leq(dom_spans[i][1], v, dom_spans[i][2]));
            Add(conds, self.child(i).lambda().at(v - dom_spans[i][1]));
        od;
	# Else branch needed below (=errExp), it ensures that cond(..).ev() does not crash 
	# with nested conds. For example:
	#   cond(c1, cond(...., errExp), b)
	# if c1 is false, our inner cond can be invalid, because then b is taken instead
	Add(conds, errExp(conds[2].t)); 
        return Lambda(v, ApplyFunc(cond, conds));
    end
));

Class(diagDirsum, fStack, rec(
    range := self >> UnifyTypes(List(self.children(), x->x.range()))
));

# Function for Pease twiddle access. Maps n->n to index into
#   Diag(TC(n,2,0)) based upon iteration l.
#   fAccPease(n, r, l)
Class(fAccPease, FuncClass, rec(
   def := (n, r, l) -> rec(N:=n, n:=n),
   lambda := self >> let(
       N := self.params[1], k := Ind(N),
       R := self.params[2], l := self.params[3],

       Lambda(k, let(
           bits:=Log2Int(N)-Log2Int(R),
	       rCyclicShift(
               bin_and(
                   k, 
                   concat(
                       bin_shl(
                           2^(bits)-1, 
                           l*Log2Int(R), 
                           bits
                       ),
		               R-1, 
                       Log2Int(R)
                   )
               ), 
               Log2Int(R), 
               Log2Int(N)
           )
	   )))
));
	   

#F fPad(<diag_func>)
#F
#F   Index function with 0 range and non zero domain. 
#F   Returns result of <diag_func> diagonal function wrapped into funcExp().
#F   Relies on rewriting nth(<loc>, funcExp(A)) -> A.

Class(fPad, FuncClass, rec(
    def       := (f) -> rec(), 
    
    domain    := self >> self.params[1].domain(),
    range     := self >> 0,
    
    lambda    := self >> let(
                    i := Ind(self.domain()), 
                    l := self.params[1].lambda(),
                    Lambda(i, funcExp(l.at(i)))),

    transpose := self >> Error("fPad is not transposable, needs inert form"),
));

#F fInsert(<map>, <value>)
#F
#F Index function with padding.
#F
#F   Ex: Gath(fInsert([1, 1, 0, 1, 0], 3))*X ==
#F
#F   [ [ 1, 0, 0 ],
#F     [ 0, 1, 0 ],
#F     [ 0, 0, 0 ],  * X + [ 0, 0, 3, 0, 3 ]^T 
#F     [ 0, 0, 1 ],
#F     [ 0, 0, 0 ] ]
#F
#F

Class(fInsert, FuncClass, rec(
    def := (map, val) -> rec(), 
    domain := self >> Length(self.params[1]),
    range  := self >> Sum(self.params[1]),
    lambda := self >> let(
        i := Ind(self.domain()), 
        map := self.params[1],
        v   := funcExp(self.params[2]),
        ctr := Counter(0),
        Lambda(i, ApplyFunc( cond, 
            ConcatList([1..Length(map)], j ->[eq(i, j-1), Cond(map[j]<>0, ctr.next(), v)])
            :: [errExp(TInt)]))),
    transpose := self >> Error("fInsert is not transposable, needs inert form"),
));

