
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


IsContainer := x -> IsRec(x) and IsBound(x.isContainer) and x.isContainer;

Class(BaseContainer, BaseOperation, rec(
    
    isContainer := true,

    new := meth(self, spl) 
        local res;
	res := SPL(WithBases(self, rec(_children:=[spl])));
	res.dimensions := res.dims();
	return res;
    end,

    dims := meth(self) return self._children[1].dims(); end,
    area := meth(self) return self._children[1].area(); end,

    isPermutation := meth(self) return IsPermutationSPL(self._children[1]); end,
    isReal := meth(self) return IsRealSPL(self._children[1]); end,
    toAMat := meth(self) return AMatSPL(self._children[1]); end,

    transpose := self >> 
        CopyFields(self, rec(_children:=[self._children[1].transpose()], 
	                     dimensions:=Reversed(self.dimensions))),
    conjTranspose := self >> 
        CopyFields(self, rec(_children:=[self._children[1].conjTranspose()], 
	                     dimensions:=Reversed(self.dimensions))),

    normalizedArithCost := self >> self._children[1].normalizedArithCost(),   
));
