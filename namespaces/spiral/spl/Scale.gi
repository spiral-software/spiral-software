
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# Scale
# ==========================================================================
Class(Scale, BaseContainer, rec(
    new := meth(self, s, spl)
        Constraint(IsSPL(spl));
	if s = 1 then return spl; fi;
        Constraint(IsScalarOrNum(s));
        return SPL(WithBases( self, 
	    rec( _children := [spl],
		scalar := s,
		dimensions := spl.dims() )));
    end,

    rChildren := self >> [self.scalar, self._children[1]],
    rSetChild := meth(self, n, what)
        if n=1 then self.scalar := what;
        elif n=2 then self._children[1] := what;
        else Error("<n> must be in [1..2]");
        fi;
    end,
    #from_rChildren  -- inherited
    #-----------------------------------------------------------------------
    isPermutation := False, 
    #-----------------------------------------------------------------------
    isReal := self >> IsRealNumber(self.scalar) and IsRealSPL(self._children[1]),
    #-----------------------------------------------------------------------
    toAMat := self >> EvalScalar(self.scalar) * AMatSPL(self._children[1]),
    #-----------------------------------------------------------------------
    arithmeticCost := meth(self, costMul, costAddMul)
        return costMul(self.scalar) * Minimum(self.dimensions) 
	       + self._children[1].arithmeticCost(costMul, costAddMul);
    end
));
