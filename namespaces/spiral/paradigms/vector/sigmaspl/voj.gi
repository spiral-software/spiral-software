
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F VO1dsJ(<n>, <v>) -- vector construct for I(1) \dirsum J(n-1)
#F
Class(VO1dsJ, BaseMat, SumsBase, rec(
    new := (self, n, v) >> SPL(WithBases(self, rec(n := n, v := v))).setDims(),
    dims := self >> [self.n, self.n], 
    #-----------------------------------------------------------------------
    rChildren := self >> [self.n, self.v],
    rSetChild := rSetChildFields("n", "v"),
    #-----------------------------------------------------------------------
    transpose := self >> self,
    conjTranspose := self >> self, 
    area := self >> self.n * self.n,
    #-----------------------------------------------------------------------
    toAMat := self >> AMatMat(MatSPL(DirectSum(O(1), J(self.n-1))))
));

#F VIxJ2(<v>) -- vector construct for I(v/2) \tensor J(2)
#F
Class(VIxJ2, BaseMat, SumsBase, rec(
    new := (self, v) >> SPL(WithBases(self, rec(v := v))).setDims(),
    dims := self >> [self.v, self.v],
    #-----------------------------------------------------------------------
    isReal := True,
    rChildren := self >> [self.v],
    rSetChild := rSetChildFields("v"),
    #-----------------------------------------------------------------------
    transpose := self >> self, 
    conjTranspose := self >> self, 
    area := self >> self.v * self.v,
    #-----------------------------------------------------------------------
    toAMat := self >> AMatMat(MatSPL(Tensor(I(self.v/2), J(2))))
));

#F For v > 2
#F    RCVIxJ2(v) == RC(VIxJ2(v/2))
#F               == I(v/4) \tensor J(2) \tensor I(2)
#F For v=2
#F    RCVIxJ2(2) == J(2)
#F
Class(RCVIxJ2, BaseMat, SumsBase, rec(
    new := (self, v) >> SPL(WithBases(self, rec(v := v))).setDims(),
    dims := self >> [self.v, self.v],
    #-----------------------------------------------------------------------
    isReal := True,
    rChildren := self >> [self.v],
    rSetChild := rSetChildFields("v"),
    #-----------------------------------------------------------------------
    transpose := self >> self, 
    conjTranspose := self >> self, 
    area := self >> self.v * self.v,
    #-----------------------------------------------------------------------
    toAMat := self >> Cond(self.v=2, J(2).toAMat(), # YSV: NOTE: check if this is correct
	                             Tensor(I(self.v/4), J(2), I(2)).toAMat())
));

#F VJxI(<m>, <v>) -- vector construct for J(m) \tensor I(v/m)
#F
Class(VJxI, BaseMat, SumsBase, rec(
    new := (self, m, v) >> SPL(WithBases(self, rec(m := m, v := v))).setDims(),
    dims := self >> [self.v, self.v],
    #-----------------------------------------------------------------------
    isReal := True,
    rChildren := self >> [self.m, self.v],
    rSetChild := rSetChildFields("m", "v"),
    #-----------------------------------------------------------------------
    transpose := self >> self, 
    conjTranspose := self >> self, 
    area := self >> self.v * self.v,
    #-----------------------------------------------------------------------
    toAMat := self >> AMatMat(MatSPL(Tensor(J(self.m), I(self.v/self.m))))
));

