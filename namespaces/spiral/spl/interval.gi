
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(TInterval, Typ, rec(
     __call__ := (self, size) >>
        WithBases(self, rec(
        size := Checked((IsInt(size) and size >= 0) or IsSymbolic(size), size),
        operations := TypOps)),

    hash := (self, val, size) >> (Sum(val, x->x.t.hash(x.v, size)) mod size)+1,

    equals := (self, o) >> IsBound(o.__bases__) and self.__bases__ = o.__bases__
                           and self.size = o.size,

    rChildren := self >> [self.size],
    rSetChild := rSetChildFields("size"),

    print := self >> When(IsBound(self.size), Print(self.name, "(", self.size, ")"), Print(self.name)),
    isIntervalT := true
));


#F IntervalFunc - base class for interval generating functions
Class(IntervalFunc, Function, Sym, rec(
    isSPL := false,
    range := self >> TInterval(self.size),
    domain := self >> TInterval(self.size),
#    tensor_op := (v,rows,cols,fl,gl) ->
#        Lambda(v, fl.at(idiv(v, cols)) * gl.at(imod(v, cols))),
#
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

    toset := self >> Filtered([0..self.size-1], i ->self.elOf(i).ev() = V(1)),

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

    result := SPL(WithBases(self,  Inherit(
        rec(params := params),
        ApplyFunc(self.def,params))));

    if h<>false then return h.objAdd(result, lkup[2]);
    else return result;
    fi;
    end,
    __call__ := ~.fromDef
));



#F ivII - interval function
#F
#F ivII(<N>) - diagonal of N 1's
#F ivII(<N>, <to>) - diagonal with 1's from 0..<to>, and 0's after
#F ivII(<N>, <from>, <to>) - diagonal with 1's in <from>..<to>, 0's elsewhere
#F
Class(ivII, IntervalFunc, rec(
    abbrevs := [ (N)         -> [N,0,N],
             (N,to)      -> [N,0,to],
             (N,from,to) -> [N,from,to] ],
    def := (N, from, to) -> Checked(IsPosInt0(N),IsPosInt0(from),IsPosInt0(to),
                                rec(size:=N)),
    elOf := (self, i) >> let(from:=self.params[2], to:=self.params[3], cond(leq(from, i, to-1), 1, 0))

));


Class(ivElOf, FuncClassOper, rec(
    updateDomain  := self >> TVoid,
    updateRange := self >> TInt,
    eval := self >> self.child(1).elOf(self.child(2))
));


Class(ivUnion, FuncClassOper, rec(
    updateDomain := self >> self.child(1).domain(),
    updateRange  := self >> self.child(1).range(),
    elOf := (self, i) >> When(ForAny(self.children(), j->j.elOf(i).ev() = V(1)), V(1), V(0)),
    toset := self >> Filtered([0..self.child(1).size-1], i ->self.elOf(i).ev() = V(1)),
));


Class(ivIntersect, FuncClassOper, rec(
    updateDomain := self >> self.child(1).domain(),
    updateRange  := self >> self.child(1).range(),
    elOf := (self, i) >> When(ForAll(self.children(), j->j.elOf(i).ev() = V(1)), V(1), V(0)),
    toset := self >> Filtered([0..self.child(1).size-1], i ->self.elOf(i).ev() = V(1)),
));


Class(ivInvert, FuncClassOper, rec(
    updateDomain := self >> self.child(1).domain(),
    updateRange  := self >> self.child(1).range(),
    elOf := (self, i) >> When(self.child(1).elOf(i).ev() = V(1), V(0), V(1)),
    toset := self >> Filtered([0..self.child(1).size-1], i ->self.elOf(i).ev() = V(0)),
));

Class(ivTensor, fTensorBase, rec(
    print := FuncClassOper.print,
    updateDomain := self >> TInterval(Product(List(self.children(), i->i.size))),
    updateRange := self >> TInterval(Product(List(self.children(), i->i.size))),
    split_op := self >> let(
        c := self.children(), v := Ind(self.domain().size),
        c1dom := c[1].domain().size, c2dom := c[2].domain().size,
        Checked(Length(c)=2,
            [ Cond(c1dom=1, 0, c2dom=1, v, idiv(v, c2dom)),
            Cond(c1dom=1, v, c2dom=1, 0, imod(v, c2dom)),
            v ])),
    combine_op := (self, split, F, G) >> F.elOf(split[1]) * G.elOf(split[2]),
    elOf := (self, i) >>
        Cond(self.numChildren() > 2, self.rightBinary().elOf(i),
         let(split := self.split_op(),
            Lambda(split[3],
             self.combine_op(split, self.child(1), self.child(2))).at(i))),
    toset := self >> Filtered([0..self.domain().size-1], i ->self.elOf(i).ev() = V(1)),
));
