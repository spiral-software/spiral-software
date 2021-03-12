
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# ColDirectSum
# ==========================================================================
Class(ColDirectSum, BaseOverlap, rec(
    #-----------------------------------------------------------------------
    abbrevs := [ 
        function(arg) 
          arg:=Flat(arg); 
          return [arg[1], arg{[2..Length(arg)]}];
        end ],
    #-----------------------------------------------------------------------
    new := meth(self, overlap, spls)
       return self._new(0, overlap, spls);
    end,
    #-----------------------------------------------------------------------
    dims := meth(self)
        local ovdim;
	ovdim := self.ovDim(self.overlap,
	                    List(self._children, t -> t.dimensions[1]));
	return [ ovdim[3] - ovdim[2] + 1, # max - min + 1
	         Sum(self._children, t -> t.dimensions[2]) ];
    end,
    #-----------------------------------------------------------------------
    toAMat := meth(self) 
        return 
	   AMatSPL(
	       Sparse(
		   self._coloverlap(
		       List(self._children, t -> t.dimensions[1]),
		       self.overlap
		   )
	       )
	   ) *
	   DirectSumAMat(List(self._children, AMatSPL));
    end,
));

ColDirectSum._transpose_class := RowDirectSum;
RowDirectSum._transpose_class := ColDirectSum;
