
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# Conjugate(<spl>, <perm>)
#    perm^-1 * spl * perm
# ==========================================================================
Class(Conjugate, BaseOperation, rec(
    new := (self, spl, conj) >> Checked(IsSPL(spl), IsSPL(conj),
	When(Dimensions(conj) = [1,1], spl,
	     SPL(WithBases( self, 
		     rec( _children := [spl, conj],
			 dimensions := spl.dimensions ))))),
    #-----------------------------------------------------------------------
    isPermutation := False, 
    #-----------------------------------------------------------------------
    dims := self >> self.dimensions,
    #-----------------------------------------------------------------------
    toAMat := self >> 
        ConjugateAMat(AMatSPL(self._children[1]), AMatSPL(self._children[2])),
    #-----------------------------------------------------------------------
    print := (self, i, is) >>
        Print("(", self.child(1).print(i, is), ") ^ (", self.child(2).print(i+is, is), ")", 
              self.printA()),
    #-----------------------------------------------------------------------
    transpose := self >> 
       Inherit(self, rec(_children := [ TransposedSPL(self._children[1]),
					self._children[2] ], 
	                dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >>
        self._children[1].arithmeticCost(costMul, costAddMul)
));
