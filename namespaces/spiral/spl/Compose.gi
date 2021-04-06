
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(Compose);
Declare(ComposeDists);
Declare(ComposeStreams);

ccFunc := function(list)
      local i, l;
      l := [];
      for i in list do
          Cond(IsBound(i.createCode), Append(l, [i.createCode()]), Append(l, [i]));
      od;
      return Flat(l);
end;


# ==========================================================================
# Compose
# ==========================================================================
Class(Compose, BaseOperation, rec(
    # flatten by associativity
    abbrevs := [ arg ->
    [ Flat(List(Flat(arg),
            s -> When(IsSPL(s) and ObjId(s)=Compose, s.children(), s))) ]
    ],

    withTags := (self, tags) >> ApplyFunc(
	ObjId(self),
	List(self.children(), x -> x.withTags(tags))).takeAobj(self),

    checkDims := children -> let(chdims := List(children, c->c.dims()),
        DoForAll([1..Length(chdims)-1], i ->
            DoForAll(Zip2(chdims[i][2],chdims[i+1][1]), x->
        When( not(IsSymbolic(x[1]) or IsSymbolic(x[2])) and (x[1] <> x[2]),
            Error("Dimensions of children ",i," and ",i+1," do not match (",x[1]," <> ",x[2],") in ", children), 0)))),

    new := meth(self, L)
        local dim;
        Constraint(IsList(L) and L<>[]);
        #ForAll(L, x -> Constraint(IsSPL(x)));
        if Length(L) = 1 then return L[1]; fi;

        dim := Rows(L[1]);
        self.checkDims(L);
        L := Filtered(L, x -> not IsIdentityObj(x));
        if L = [] then   # all factors are identities
            return I(dim);
        elif Length(L) = 1 then 
            return L[1]; 
        else
            return SPL(WithBases(self, rec(
                         _children := L,
                         dimensions := [L[1].dims()[1], L[Length(L)].dims()[2]])));
        fi;
    end,

     rng:= self>>let(c := self._children, c[1].rng()),
     dmn:= self>>let(c := self._children, Last(c).dmn()),
    #-----------------------------------------------------------------------
 #   dims := self >> let(c := self._children, [Rows(c[1]), Cols(Last(c))]),
     advdims := self >> [self._children[1].advdims()[1], Last(self._children).advdims()[2]],
    #-----------------------------------------------------------------------
    isPermutation := self >> ForAll(self._children, IsPermutationSPL),
    #-----------------------------------------------------------------------
    toAMat := self >> Product(List(self._children, AMatSPL)),
    #-----------------------------------------------------------------------
    transpose := self >>
        CopyFields(self, rec(
        _children := Reversed(List(self._children, x -> x.transpose())),
        dimensions := Reversed(self.dimensions))),
    conjTranspose := self >>
        CopyFields(self, rec(
        _children := Reversed(List(self._children, x -> x.conjTranspose())),
        dimensions := Reversed(self.dimensions))),
    inverse := self >>
        CopyFields(self, rec(
        _children := Reversed(List(self._children, x -> x.inverse())),
        dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    printSeparationChar := " * ",
    print := meth(self, indent, indentStep)
        local s, newline;

    s := self.children();
    if Length(s) = 2 and ((IsBound(s[1]._sym) and IsBound(s[2]._mat))
                       or (IsBound(s[2]._sym) and IsBound(s[1]._mat)))
       or ForAll(s, x->IsBound(x._sym))
        then newline := Ignore;
    else newline := self._newline; fi;

        DoForAllButLast(s, c->Chain(SPLOps.Print(c, indent, indentStep),
                                Print(self.printSeparationChar),
                    newline(indent)));

    Last(s).print(indent, indentStep);

    end,
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >>
        Sum(List(self.children(), x -> x.arithmeticCost(costMul, costAddMul))),

    createCode := self >> let (l := self._children, l2 := ccFunc(List(l)),
          Compose(l2)),

    normalizedArithCost := self >> Sum(List(self.children(), i->i.normalizedArithCost())),
    latexSymbol := "\\cdot"

 ));

Class(ComposeDists, Compose, rec(
    printSeparationChar := " |*| ",
    leftMostParScat := self >> self._children[1].leftMostParScat(),
    rightMostParGath := self >> self._children[Length(self._children)].rightMostParGath(),
));

Class(ComposeStreams, Compose, rec(
    printSeparationChar := " -*- ",
    leftMostParScat := self >> self._children[1].leftMostParScat(),
    rightMostParGath := self >> self._children[Length(self._children)].rightMostParGath(),
));
