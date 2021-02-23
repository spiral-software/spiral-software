
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# Perm
# ==========================================================================
Class(Perm, BaseMat, rec(
    new := (self, P, size) >>
	Checked(IsPerm(P), IsSymbolic(size) or IsValue(size) or (IsInt(size) and SPLOps.IsDegree(P, size)),
	    When(P = (), 
		    SPL(WithBases(self, rec(isIdentity:=True, 
				            element:=P, size := size, 
					    dimensions := [size,size]))),
		    SPL(WithBases(self, rec(element:=P, size := size, 
					    dimensions := [size,size]))))),

    rChildren := self >> [self.element, self.size],
    rSetChild := rSetChildFields("element", "size"),
    #-----------------------------------------------------------------------
    dims := self >> [self.size, self.size],
    #-----------------------------------------------------------------------
    isPermutation := True,
    #-----------------------------------------------------------------------
    isReal := True,
    #-----------------------------------------------------------------------
    toAMat := self >> AMatPerm(self.element, EvalScalar(self.size)), 
    #-----------------------------------------------------------------------
    transpose := self >>  # we use inherit to copy all fields of self
        Inherit(self, rec(element := self.element^-1)),
    #-----------------------------------------------------------------------
    print := meth(self, indent, indentStep) 
        Print(self.name, "(",  self.element, ", ", self.size, ")");
    end,
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >> costMul(0) - costMul(0)
));
