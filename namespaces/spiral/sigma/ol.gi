
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


DimLength := (l) -> When(IsList(l), Length(l), 1); 
               
Class(CrossBase, BaseOperation, rec(
  abbrevs := [ arg -> [Flat(arg)] ],

   new := meth(self, L)
        if Length(L)=1 then return L[1]; fi;
        return SPL(WithBases(self, rec(_children:=L, dimensions := TransposedMat(List(L, e -> e.dims())))));
   end,

   area := self >> Sum(self.children(), x->x.area()),
   
   normalizedArithCost := self >> Sum(self.children(), x -> x.normalizedArithCost()),

));

Declare(Cross);

Class(Cross, CrossBase, rec(
   codeletName:="C",

   isPermutation := self >> false,

   dmn:=self>>let(li:=[],Flat(List(self._children, x-> x.dmn()))),

   rng:=self>>let(li:=[],Flat(List(self._children, x-> x.rng()))),

   sums:= self>>ObjId(self)(List(self.rChildren(),x->x.sums())),

   transpose := self >> self,

   isIdentity := self >> ForAll(self._children, IsIdentitySPL),

   isBlockTransitive := true,
));

Declare(fCross);
Class(fCross, FuncClassOper, rec(
   _perm := true,

   isFunction := true,
   skipOneChild := true,

   range := (self) >> StripList(List(self.rChildren(), (x) -> x.range())),

   advrange := (self) >> List(self.rChildren(), (x) -> x.advrange()[1]),

   domain := (self) >> StripList(List(self.rChildren(), (x) -> x.domain())),

   advdomain := (self) >> List(self.rChildren(), (x) -> x.advdomain()[1]),

   toSpl := (self, allinds, cols) >> 
       fCross(List([1..Length(self.rChildren())], 
               i -> self.rChildren()[i].toSpl(allinds,cols[i]))),

   downRank := (self, loopid, ind) >>
       fCross(List(self.rChildren(),
           c -> c.downRank(loopid, ind))),

   downRankFull := (self, allinds) >>
       fCross(List(self.rChildren(),
           c -> c.downRankFull(allinds))),

   isIdentity := self >> ForAll(self._children, IsIdentity),
   
   transpose  := self >> ObjId(self)(List(self._children, e -> e.transpose())),
));

Class(ExplicitGath, Gath);
Class(ExplicitCross, Cross);

Class(ExplicitOLRuleset, RuleSet);
RewriteRules(ExplicitOLRuleset, rec(
   Gath_f2DTrExpR := Rule([Gath, [@(2,fCompose), @(3,f2DTrExplicitR), ...]],
      e-> Gath(fCompose(Drop(@(2).val.rChildren(),1)))*ExplicitGath(ApplyFunc(f2DTr,@(3).val.rChildren()))),

   Gath_f2DTrExpLR := Rule([Gath, [@(2,fCompose), @(3,f2DTrExplicitL), @(4,f2DTrExplicitR), ...]],
      e-> Gath(fCompose([ApplyFunc(f2DTr,@(4).val.rChildren())]::Drop(@(2).val.rChildren(),2)))
      *ExplicitGath(ApplyFunc(f2DTr,@(3).val.rChildren()))),

   Cross_Compose_ExplicitGath_1 := Rule([Cross, [@(1,Compose), ..., @(2,ExplicitGath)], @(3) ],
       e -> 
       Compose(Cross(Compose(DropLast(@(1).val.rChildren(),1)), @(3).val),
       ExplicitCross(ApplyFunc(Gath, @(2).val.rChildren()), ApplyFunc(2DI, @(3).val.advdims()[2][1])))),

   Cross_Compose_ExplicitGath_2 := Rule([Cross, @(3), [@(1,Compose), ..., @(2,ExplicitGath)] ],
       e -> 
       Compose(Cross(@(3).val, Compose(DropLast(@(1).val.rChildren(),1))),
       ExplicitCross(ApplyFunc(2DI, @(3).val.advdims()[2][1]), ApplyFunc(Gath, @(2).val.rChildren())))),
));

Class(FinalOLRuleset, RuleSet);
RewriteRules(FinalOLRuleset, rec(
   ExplicitCross := Rule(@(1,ExplicitCross),
       e -> ApplyFunc(Cross, @(1).val.rChildren()))
));
