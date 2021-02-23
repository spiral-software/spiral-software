
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# RowTensor
# ==========================================================================
Class(RowTensor, BaseOverlap, rec(
    #-----------------------------------------------------------------------
    dims := self >> let(D := self._children[1].dimensions, 
        [ self.isize * D[1],
	  D[2] + AbsInt((D[2] - self.overlap) * (self.isize - 1)) ]),
    #-----------------------------------------------------------------------
    toAMat := self >> let(n := EvalScalar(self.isize),
        TensorProductAMat(AMatSPL(I(n)), AMatSPL(self._children[1])) *
	AMatSPL(Sparse(
		self._rowoverlap(
		    List([1..n], i-> EvalScalar(Cols(self._children[1]))),
		    EvalScalar(self.overlap)
		)))
    ),
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >>
        self.isize * self._children[1].arithmeticCost(costMul, costAddMul)
));

# ==========================================================================
# ColTensor
# ==========================================================================
Class(ColTensor, BaseOverlap, rec(
    #-----------------------------------------------------------------------
    dims := self >> let(D := self._children[1].dimensions,
        [ D[1] + AbsInt((D[1] - self.overlap) * (self.isize - 1)),
	  self.isize * D[2] ]),
    #-----------------------------------------------------------------------
    toAMat := self >> let(n := _unwrap(self.isize),
        AMatSPL(Sparse(
		self._coloverlap(
		    List([1..n], i->self._children[1].dimensions[1]),
		    self.overlap
		))) * 
	TensorProductAMat(AMatSPL(I(n)), AMatSPL(self._children[1]))
    )
));

ColTensor._transpose_class := RowTensor;
RowTensor._transpose_class := ColTensor;
