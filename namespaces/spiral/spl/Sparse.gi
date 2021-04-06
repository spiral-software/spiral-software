
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# Sparse
# ==========================================================================
Class(Sparse, BaseMat, rec(
    new := meth(self, L)
        Constraint(IsList(L));
	if not ForAll(L, t -> IsList(t) and Length(t)=3 and 
		              IsInt(t[1]) and IsInt(t[2])) then
	    Error("<L> must be a list of triples [i, j, a_(i,j)]");
	fi;	
        return SPL(WithBases( self, rec( element := L,
				       dimensions := [ Maximum(List(L, t->t[1])),
				                       Maximum(List(L, t->t[2])) ]
                       )));
    end,
    #-----------------------------------------------------------------------
    dims := self >> [ Maximum(List(self.element, t->t[1])), 
	              Maximum(List(self.element, t->t[2])) ],
    #-----------------------------------------------------------------------
    isPermutation := False, 
    #-----------------------------------------------------------------------
    isReal := self >> ForAll(self.element, t -> IsRealNumber(t[3])),
    #-----------------------------------------------------------------------
    toAMat := self >> AMatMat(List(MatSparseSPL(self), r -> List(r, EvalScalar))),
    #-----------------------------------------------------------------------
    transpose := meth(self)  # we use CopyFields to copy all fields of self
        local L, t;
	L := [ ];
	for t in self.element do
	    Add(L, [t[2], t[1], t[3]]);
	od;
	return CopyFields(self, rec(element := L, 
		                 dimensions := Reversed(self.dims())));
    end,
    conjTranspose := meth(self)  # we use CopyFields to copy all fields of self
        local L, t;
	L := [ ];
	for t in self.element do
	    Add(L, [t[2], t[1], Global.Conjugate(t[3])]);
	od;
	return CopyFields(self, rec(element := L, 
		                 dimensions := Reversed(self.dims())));
    end,
    #-----------------------------------------------------------------------
    arithmeticCost := meth(self, costMul, costAddMul)
        local cost, row, elms;
	cost := costMul(0) - costMul(0); # will work even when costMul(0) <> 0
	for row in [1..self.dimensions[1]] do
	    elms := Filtered(self.element, e -> e[1]=row);
	    if Length(elms) > 0 then
		cost := cost + costMul(elms[1][3]) 
		             + Sum(elms{[2..Length(elms)]}, e -> costAddMul(e[3]));
	    fi;
	od;
	return cost;
    end
));

