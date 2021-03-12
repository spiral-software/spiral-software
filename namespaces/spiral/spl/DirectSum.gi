
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(DirectSum);
# ==========================================================================
# DirectSum
# ==========================================================================
Class(DirectSum, BaseOperation, rec(
    _spl_name := "direct_sum",

    abbrevs := [ arg ->
    [ Filtered(
        Flat(List(Flat(arg),
            s -> When(IsSPL(s) and ObjId(s)=DirectSum, s.children(), s))), x->x.dimensions<>[0,0]) ]
    ],

    new := meth(self, L)
        #ForAll(L, x -> Constraint(IsSPL(x)));
    if ForAll(L, IsIdentitySPL)
        then return I(Sum(L, Rows));
    else
        return SPL(WithBases( self,
        rec( _children := L,
        dimensions := [ Sum(L, t -> t.dimensions[1]),
                        Sum(L, t -> t.dimensions[2]) ] )));
    fi;
    end,
    #-----------------------------------------------------------------------
    dims := self >> [ Sum(self._children, t -> t.dimensions[1]),
                  Sum(self._children, t -> t.dimensions[2]) ],
    area := self >> Sum(self._children, t -> t.area()),
    #-----------------------------------------------------------------------
    isPermutation := self >> ForAll(self._children, IsPermutationSPL),
    #-----------------------------------------------------------------------
    toAMat := self >> DirectSumAMat(List(self._children, AMatSPL)),
    #-----------------------------------------------------------------------
    transpose := self >> CopyFields(self, rec(
        _children := List(self._children, x->x.transpose()),
        dimensions := Reversed(self.dimensions))),
    conjTranspose := self >> CopyFields(self, rec(
        _children := List(self._children, x->x.conjTranspose()),
        dimensions := Reversed(self.dimensions))),
    inverse := self >> CopyFields(self, rec(
        _children := List(self._children, x->x.inverse()),
        dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >>
        Sum(List(self.children(), x -> x.arithmeticCost(costMul, costAddMul)))
));

Class(DelayedDirectSum, DirectSum, rec(sums := self >> self));
