
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ExpFunc - ExpDiag element function.
#F
Class(ExpFunc, Lambda, rec(
    _isExpFunc := true,
    # this is not a ranked index function, not related to GT,
    # GT should not mess with it in the first place
    downRank := (self, loopid, ind) >> self,
    downRankFull := (self, inds) >> self,
    upRank := self >> self,
    upRankBy := (self, n) >> self,
    rotate := (self, n) >> self,
    split := (self, loopid, inner_its, outer_its) >>  self,
));

IsExpFunc := (obj) -> IsRec(obj) and IsBound(obj._isExpFunc) and obj._isExpFunc;


#F ExpDiag(<func>, <d>)
#F
#F ExpDiag(ExpFunc( [a,b], sub(a,b) ), 3)
#F
#F corresponds to matrix:
#F   [ [ 1 -1  0  0  0  0 ],
#F     [ 0  0  1 -1  0  0 ],
#F     [ 0  0  0  0  1 -1 ] ]
#F

Class(ExpDiag, BaseMat, rec(
    _short_print := true,
    abbrevs := [ (L, d) -> Checked( IsExpFunc(L) and IsPosIntSym(d), [L, d] )],

    new := (self, L, d) >> SPL(WithBases(self, rec(element := L, TType := L.range(), d := d))).setDims(),

    #-----------------------------------------------------------------------
    dims := self >> [self.d, self.d*Length(self.element.vars)],
    #-----------------------------------------------------------------------
    isPermutation := self >> false,
    #-----------------------------------------------------------------------
    isReal := self >> let(
	t := self.element.range(), 
        tt := Cond(IsVecT(t), t.t, t),
	tt <> TComplex and ObjId(tt)<>T_Complex),
    #-----------------------------------------------------------------------
    toAMat         := abstract(),
    #-----------------------------------------------------------------------
    transpose      := abstract(),
    conjTranspose  := abstract(),
    inverse        := abstract(),
    #-----------------------------------------------------------------------
    arithmeticCost := abstract(),

    sums           := self >> self,
    rChildren      := (self) >> [self.element, self.d],
    rSetChild      := rSetChildFields("element", "d"),
    area           := self >> self.d
));




