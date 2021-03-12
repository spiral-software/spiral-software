
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# RowDirectSum
# ==========================================================================
Class(RowDirectSum, BaseOverlap, rec(
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
    _dims := meth(self)
        local ovdim;
    ovdim := self.ovDim(self.overlap,
                        List(self._children, t -> t.dimensions[2]));
    return [ Sum(self._children, t -> t.dimensions[1]),
             ovdim[3] - ovdim[2] + 1 # max - min + 1
    ];
    end,
    dims := self >> When(IsBound(self._setDims), self._setDims, let(d := Try(self._dims()), 
	    When(d[1], d[2], [errExp(TInt), errExp(TInt)]))),
    #-----------------------------------------------------------------------
    toAMat := meth(self)
        return
      DirectSumAMat(List(self._children, AMatSPL)) *
      AMatSPL(
          Sparse(
          self._rowoverlap(
              List(self._children, t -> t.dimensions[2]),
              self.overlap
          )
          )
      );
    end,
    #-----------------------------------------------------------------------
    arithmeticCost := meth(self, costMul, costAddMul)
        return Sum(List(self.children(), x -> x.arithmeticCost(costMul, costAddMul)));
    end
));
