
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



Class(OLBase, SumsBase, rec(
    isOL := true,
    visitAs := "OLBase",
));

## Multiplication operator
## OLMultiplication(1, n) is I(n)
## OLMultiplication(2, n) is a point-wise multiplication of two vectors of size n
Class(OLMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [self.rChildren()[2], StripList(Replicate(self.rChildren()[1], self.rChildren()[2]))],
));

## OLConjMultiplication(n) is a point-wise multiplication of two vectors of size n where second vector is complex conjugated
##
Class(OLConjMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [self.rChildren()[2], StripList(Replicate(self.rChildren()[1], self.rChildren()[2]))],
));

#F RCOLMultiplication is RC(OLMultiplication(..))
#F
Class(RCOLMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [2*self.rChildren()[2], StripList(Replicate(self.rChildren()[1], 2*self.rChildren()[2]))],
));

## RCOLConjMultiplication(n) is RC(OLConjMultiplication(n))
##
Class(RCOLConjMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [2*self.rChildren()[2], StripList(Replicate(self.rChildren()[1], 2*self.rChildren()[2]))],
));


Class( 2DI, SumsBase, RewritableObject, AttrMixin, rec(
   isSPL := true,
   isIdentity := True,
   advdims := self >> let(a:=self.rChildren(), [[a], [a]]),
   dims := self >> let(a:=Product(self.rChildren()), [a, a]),
   rng := self >> [ TArray(TUnknown, Product(self.rChildren()))],
   dmn := self >> [ TArray(TUnknown, Product(self.rChildren()))],
   children := self >> [],
   arity := ClassSPL.arity,
   updateParams := meth(self)
       self.func := fId(Product(self.params));
       Inherited();
   end,
));

Class(LeftOver, RewritableObject, ClassSPL, rec(
    isSums := true,
    isSPL := true,

    __call__ := (self, cond, spl) >> Cond(
	cond=true, spl,
	cond=false, let(d := spl.dims(), VirtualPad(d[1], d[2], I(0))),
	Inherited(cond, spl)),

    rng := self >> self.params[2].rng(),
    dmn := self >> self.params[2].dmn(),
    dims := self >> [ StripList(List(self.rng(), (l) -> l.size)), StripList(List(self.dmn(), (l) -> l.size)) ],
    advdims := self >> self.params[2].advdims(),

    isInplace := self >> self.params[2].isInplace(),
    transpose := self >> ObjId(self)(self.params[1], self.params[2].transpose()),
    conjTranspose := self >> ObjId(self)(self.params[1], self.params[2].conjTranspose()),

    isReal := self >> self.params[2].isReal(),

    a := rec(),

    print:= arg >> let(
	self := arg[1],
	i := Cond(Length(arg)>=3, arg[2], 0), 
	is := Cond(Length(arg)>=3, arg[3], 4), 
	Print(self.__name__, "(", self.params[1], "\n", 
	    Blanks(i+is), self.params[2].print(i+is, is), "\n", Blanks(i), ")"))
));


Class( OLDup, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [Replicate(self.params[1], self.params[2]), self.params[2]],
));

# ParSeqWrap is a helper class used dduring codegen stage to ease code generation
#

Class(ParSeqWrap, BaseContainer, rec(
    __call__ := (self, p, ci, y, x) >> 
        WithBases(self, rec(p := p, ci := ci, y := y, x := x, dimensions := [p.dimsCompL(p.child(ci)), p.dimsCompR(p.child(ci))])),
    dims := self >> self.dimensions,
    children := self >> [self.p.child(self.ci)],
));

#  ParSeq(<N>, <spl>, <spl>, ...) operator implements simultaneous sequentional (first <N> inputs/outputs) and 
#    parallel (all other inputs/outputs) data flow:
#  
#        +----+-<- +
#        |    |
#  * <- A <- B <- *
#        |    |
#  + <---+----+
#
# For example: ParSeq(1, Addition(2,1), Addition(2,1), Addition(2,1));
#              Y[0] := X1[0] + X2[0] + X2[0] + X2[0];

Class(ParSeq, SumsBase, BaseOperation, rec(
    area := self >> Sum(self.children(), x->x.area()),
    abbrevs := [ arg -> Checked( Length(arg)>1 and IsPosInt0(arg[1]),
                                 [ arg[1], Flat(Drop(arg, 1)) ] )
               ],

    # filter list leaving elements with positions of Compose inputs/outputs
    filtCompL := (self, lst) >> lst{[1..self.fb_cnt]},
    filtCompR := (self, lst) >> lst{[1..self.fb_cnt]},
    filtSUML  := (self, lst) >> lst{[self.fb_cnt+1..Length(lst)]},
    filtSUMR  := (self, lst) >> lst{[self.fb_cnt+1..Length(lst)]},

    dimsCompL := (self, child) >> self.filtCompL(Flat([self.dims()[1]])),
    dimsCompR := (self, child) >> self.filtCompR(Flat([self.dims()[2]])),
    dimsSUML  := (self, child) >> self.filtSUML(Flat([self.dims()[1]])),
    dimsSUMR  := (self, child) >> self.filtSUMR(Flat([self.dims()[2]])),

    checkDimsCompose := (self) >> let(chdims := List(self._children, c -> [self.dimsCompL(c), self.dimsCompR(c)]),
        DoForAll([1..Length(chdims)-1], i ->
            DoForAll(Zip2(chdims[i][2], chdims[i+1][1]), x->
                When( not(IsSymbolic(x[1]) or IsSymbolic(x[2])) and (x[1] <> x[2]),
                      Error("Dimensions of children ",i," and ",i+1," do not match (",x[1]," <> ",x[2],") in ", self._children), 0)))),

    checkDimsSUM := (self) >> let(
        chdims := TransposedMat(List(self._children, c -> [self.dimsSUML(c), self.dimsSUMR(c)])),
        dims := List(TransposedMat(chdims[1]) :: TransposedMat(chdims[2]), d -> Set(Filtered(d, e -> not IsSymbolic(e)))),
        When(not ForAll(dims, d -> Length(d) in [0,1]),
            Error("Dimensions of summands do not match"), 0)),

    #-----------------------------------------------------------------------
    new := meth(self, fb_cnt, C)
        local dims, obj, a;
        Constraint(Length(C) >= 1); Constraint(ForAll(C, IsSPL));
        a := C[1].arity();
        Constraint(ForAll(C, e -> e.arity()=a));
        Constraint(IsInt(fb_cnt) and fb_cnt>=0 and fb_cnt <= a[1] and fb_cnt <= a[2]);

        if Length(C) = 1 then return C[1]; fi;
        if fb_cnt=a[1] and a[1]=a[2] then return Compose(C); fi;
        if fb_cnt=0 then return SUM(C); fi;

        # check 'Compose' dims
        obj := SPL(WithBases(self, rec( _children := C, fb_cnt := fb_cnt)));
        obj.checkDimsCompose();
        obj.checkDimsSUM();
        return obj.setDims();
    end,
    
    rng:= self >> self._children[1].rng(),
    dmn:= self >> Last(self._children).dmn(),

    isPermutation := self >> false,

    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [self.fb_cnt] :: rch).appendAobj(self),

    print := (self, i, is) >> self._print([self.fb_cnt] :: self.rChildren(), i, is),

));


