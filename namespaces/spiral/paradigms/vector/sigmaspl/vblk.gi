
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ==========================================================================
#F VBlk(<M>, <v>)
#F
#F Matrix of <v> element diagonals -- each matrix entry is represented as a 
#F list vector elements, or as vector values, e.g., TVect(t, v).value(..) 
#F
Class(VBlk, SumsBase, Mat, rec(
    #-----------------------------------------------------------------------
    new := (self, M, v) >> SPL(WithBases(self, rec(
        element := M,
	v       := v, 
        TType   := Cond( # NOTE: add checks to M
	    IsList(M),     TVect(UnifyTypes(List(Concatenation(M), InferType)).t, v),
	    IsSymbolic(M), M.t.t.t),
    ))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> Dimensions(self.element) * self.v,
    #-----------------------------------------------------------------------
    rChildren := self >> [self.element, self.v],
    rSetChild := rSetChildFields("element", "v"),
    #-----------------------------------------------------------------------
    toAMat   := self >> BlockMat(MapMat(self.element, 
        x -> Diag(FList(self.TType.t, _unwrap(When(IsList(x), x, x.ev())))))).toAMat(),
    #-----------------------------------------------------------------------
    velement := self >> Cond(
	IsList(self.element), let(
	    d := Dimensions(self.element),
	    t := self.TType,
	    TArray(TArray(t, d[2]), d[1]).value(
		MapMat(self.element, x -> self.TType.value(x)))),
	IsSymbolic(self.element), self.element),
    #-----------------------------------------------------------------------
    transpose := self >> Cond(self.isSymmetric(), self,
        CopyFields(self, rec(element := TransposedMat(self.element),
                             dimensions := Reversed(self.dimensions)))),
    conjTranspose := self >> self.transpose(), # NOTE: implement this properly
    #-----------------------------------------------------------------------
    isSymmetric := False,
    setSymmetric := self >> Inherit(self, rec(isSymmetric := True)),
    #-----------------------------------------------------------------------
    isReal := self >> Cond(
	IsList(self.element),  ForAll(self.element, r -> ForAll(r, IsRealNumber)),
	not IsComplexT(self.element.t.t.t)),
));

#F ==========================================================================
#F RCVBlk(<M>, <v>)
#F
#F Matrix of complex diagonals --
#F     each matrix entry is represented as a list of v/2 complex numbers
#F
#F <v> here represents the *real* vector length 
#F (YSV: it used to be complex vector length in older versions)
#F
Class(RCVBlk, VBlk, rec(
    #-----------------------------------------------------------------------
    toAMat := self >> BlockMat(MapMat(self.element, 
        x -> RCDiag(RCData(FList(TComplex, x))))).toAMat(),
    #-----------------------------------------------------------------------
    velement := self >> Cond(
        IsList(self.element),
            apack.fromMat(self.element, cel -> ApplyFunc(vpack, 
                ConcatList( Flat([cel]), e -> [re(e), im(e)] ))),
        IsSymbolic(self.element),
            self.element),
    #-----------------------------------------------------------------------
    isReal := self >> true,
    #-----------------------------------------------------------------------
    # transposition will conjugate complex numbers, similarly to RCDiag
    transpose := self >> ObjId(self)(
	MapMat(TransposedMat(self.element), x -> List(x, conj)), self.v),
    #-----------------------------------------------------------------------
    conjTranspose := self >> ObjId(self)(TransposedMat(self.element), self.v)
));

#F ==========================================================================
#F VReplicate(<v>) - 1 -> v vector,  each component is splatted.
#F
#F Takes 1 vector and produces <v> vectors, by vdup'ping (splatting) each component
#F
Class(VReplicate, BaseContainer, rec(
    #-----------------------------------------------------------------------
    new := (self, v) >> SPL(WithBases(self, rec(_children := [], v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.v^2, self.v],
    #-----------------------------------------------------------------------
    toAMat := self >> Tensor(
        I(self.v),
        Mat(TransposedMat([Replicate(self.v, 1)]))).toAMat(),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.v, ")"),
    #-----------------------------------------------------------------------
    sums  := self >> self,
    #-----------------------------------------------------------------------
    isPermutation := False, 
    #-----------------------------------------------------------------------
    rChildren := self >> [],
    rSetChild := (self, n, what) >> Error("no children"),
    #-----------------------------------------------------------------------    
    transpose     := self >> InertTranspose(self),
    conjTranspose := self >> InertConjTranspose(self)
));

#F ==========================================================================
#F VHAdd(<v>) - takes a <v> vectors and produces 1 vector
#F                         - each vector is added horizontally
#F
Class(VHAdd, BaseContainer, rec(
    #-----------------------------------------------------------------------    
    new := (self, v) >> SPL(WithBases(self, rec(v := v))).setDims(),
    #-----------------------------------------------------------------------    
    dims := self >> [self.v, self.v^2],
    #-----------------------------------------------------------------------    
    toAMat := self >> Tensor(
        I(self.v),
        Mat([Replicate(self.v,1)])).toAMat(),
    #-----------------------------------------------------------------------    
    print := (self,i,is) >> Print(self.name, "(", self.v, ")"),
    #-----------------------------------------------------------------------    
    sums := self >> self,
    #-----------------------------------------------------------------------    
    isPermutation := False, 
    #-----------------------------------------------------------------------    
    rChildren := self >> [],
    _children := self >> [],
    #-----------------------------------------------------------------------    
    transpose     := self >> Error("Not implemented"),
    conjTranspose := self >> Error("Not implemented"),
));


