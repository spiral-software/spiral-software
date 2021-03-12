
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# Mat
# ==========================================================================
Class(Mat, BaseMat, rec(
    new := (self, M) >> Checked(IsMat(M), 
	SPL(WithBases(self, 
		rec( element    := M,
		     dimensions := DimensionsMat(M) )))),
    #-----------------------------------------------------------------------
    dims := self >> DimensionsMat(self.element),
    #-----------------------------------------------------------------------
    isPermutation := False, 
    #-----------------------------------------------------------------------
    isReal := self >> ForAll(self.element, r -> ForAll(r, x -> IsRealNumber(x))),
    #-----------------------------------------------------------------------
    toAMat := self >> AMatMat(List(self.element, r -> List(r, EvalScalar))),
    #-----------------------------------------------------------------------
    transpose := self >>  # we use CopyFields to copy all fields of self
        CopyFields(self, rec(element := TransposedMat(self.element),
		          dimensions := Reversed(self.dimensions))),
    transpose := self >>  # we use CopyFields to copy all fields of self
        CopyFields(self, rec(element := self.element ^ -1,
		          dimensions := Reversed(self.dimensions))),
    conjTranspose := self >>  # we use CopyFields to copy all fields of self
        CopyFields(self, rec(element := MapMat(TransposedMat(self.element), conj),
		          dimensions := Reversed(self.dimensions))),
	
    #-----------------------------------------------------------------------
    arithmeticCost := meth(self, costMul, costAddMul)
        local cost, row, nz;
	cost := costMul(0) - costMul(0); # will work even when costMul(0) <> 0
	for row in self.element do
	    nz := Filtered(row, x -> x<>0);
	    if Length(nz) > 0 then
		cost := cost + costMul(nz[1]) 
		             + Sum(nz{[2..Length(nz)]}, x -> costAddMul(x));
	    fi;
	od;
	return cost;
    end,
));

