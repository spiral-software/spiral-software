
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(BaseVGath, BaseMat, SumsBase, rec(
    isVGath := true,
    sums := self >> self,
    isReal := self >> true,
    dims := self >> Error(ObjId(self), ".dims() is undefined"),
    needInterleavedLeft := False,
    needInterleavedRight := False,
    conjTranspose := self >> self.transpose(), 
));

Class(BaseSVGath, BaseMat, SumsBase, rec(
    isVGath := true,
    isSVGath := true,
    #-----------------------------------------------------------------------
    abbrevs := [
	(func, v, sv) -> Checked(
	    IsFunction(func) or IsFuncExp(func), IsInt(v), IsInt(sv), 
	    [func, v, sv, Cond(CanBeFullyEvaluated(func.domain()), EvalScalar((func.domain()*sv mod v) / sv), Unk(TInt))]),
	(func, v, sv, rem) -> Checked(
	    IsFunction(func) or IsFuncExp(func), IsInt(v), IsInt(sv), rem _is Unk or IsInt(_unwrap(rem)),
	    [func, v, sv, rem]),
    ],
    #-----------------------------------------------------------------------
    new := (self, func, v, sv, rem) >> SPL(WithBases(self, 
        rec(func := func, v := v, sv := sv, rem := rem))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> Error(ObjId(self), ".dims() is undefined"),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.func, ", ",
        self.v, ", ", self.sv, ", ", self.rem, ")", self.printA()),
    #-----------------------------------------------------------------------
    getConstantRem := self >> Cond(
	self.rem _is Unk and not IsSymbolic(EvalScalar(self.func.domain())),
	    EvalScalar(imod(self.func.domain()*self.sv, self.v) / self.sv), 
	self.rem _is Unk,
	    Error("Can't generate sub-vector gather/scatter code for <self>, because its dimensions are unknown"),
	self.rem
    ),
    #-----------------------------------------------------------------------
    conjTranspose := self >> self.transpose(), 

    sums := self >> self,
    isReal := self >> true,
    needInterleavedLeft := False,
    needInterleavedRight := True
));

IsVGath := x -> IsRec(x) and IsBound(x.isVGath) and x.isVGath;
IsSVGath := x -> IsRec(x) and IsBound(x.isSVGath) and x.isSVGath;

#F ==========================================================================
#F VGath(<func>, <v>) - vector gather (read) matrix, assumes aligned input
#F 
#F <func> must be a vector index function returning vector indices
#F
#F VGath(f, v) == Gath(fTensor(f, fId(v)))
#F ==========================================================================
Class(VGath, BaseVGath, SumsBase, rec(
    #-----------------------------------------------------------------------
    #F NOTE: rChildren: does not expose .v!
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v),
    #-----------------------------------------------------------------------
    new := (self, func, v) >> SPL(WithBases(self,
        rec(func := func, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.v*self.func.domain(), self.v*self.func.range()],
    #-----------------------------------------------------------------------
    area := self >> self.func.domain() * self.v,
    transpose := self >> VScat(self.func, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.func, ", ", self.v,")", self.printA()),
    #-----------------------------------------------------------------------
    toAMat := self >> let(v:=self.v, n := EvalScalar(v*self.func.domain()),
        N := EvalScalar(v*self.func.range()),
        func := fTensor(self.func, fId(v)).lambda(),
        AMatMat(List([0..n-1], row -> BasisVec(N, EvalScalar(func.at(row).ev()))))),
));


#F ==========================================================================
#F VGath_u(<func>, <v>) -- assumes input is unaligned, and uses scalar <func>
#F
#F VGath_u performs an unaligned gather operations. Thus <func> must be an
#F index function returning scalar, not vector, addresses (unlike VGath).
#F
#F In particular:
#F   VGath(func, vlen) = VGath_u(fTensor(func, fBase(vlen, 0)), vlen)
#F ==========================================================================
Class(VGath_u, VGath, rec(
    new := (self, func, v) >> SPL(WithBases(self,
        rec(func := func, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.v*self.func.domain(), self.func.range()],
    #-----------------------------------------------------------------------
    toAMat := self >> let(v:=self.v, 
        n := EvalScalar(v*self.func.domain()),
        N := EvalScalar(self.func.range()),
        func := self.func.lambda(),
        AMatMat(List([0..n-1], row -> BasisVec(N, EvalScalar(add(func.at(idiv(row,v)), imod(row, v)).ev()))))),
    #-----------------------------------------------------------------------
    transpose := self >> VScat_u(self.func, self.v)
));

#F ==========================================================================
#F VGath_zero(N, n, <v>) - vector gather matrix
#F ==========================================================================
Class(VGath_zero, BaseVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.N, self.n, self.v],
    rSetChild := rSetChildFields("N", "n", "v"),
    #-----------------------------------------------------------------------
    new := (self, N, n, v) >> SPL(WithBases(self,
        rec(n := n, N := N, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.v * self.n, self.v * self.N], 
    #-----------------------------------------------------------------------
    transpose := self >> VScat_zero(self.N, self.n, self.v),
    toAMat := self >> VGath(fAdd(self.N, self.n, 0), self.v).toAMat(),
 ));


#F ==========================================================================
#F VGath_dup(<func>, <v>) - gathers scalar values and replicates <v> times to
#F                          obtain a vector values (read) matrix
#F Example:
#F spiral> PrintMat(MatSPL(VGath_dup(fId(4), 2)));
#F [ [ 1,  ,  ,   ],
#F   [ 1,  ,  ,   ],
#F   [  , 1,  ,   ],
#F   [  , 1,  ,   ],
#F   [  ,  , 1,   ],
#F   [  ,  , 1,   ],
#F   [  ,  ,  , 1 ],
#F   [  ,  ,  , 1 ] ]
#F
#F This is performed using shuffles in SSE/SSE2, or movddup in SSE4.
#F
#F The transpose of VGath_dup is VScat_red (red = reduce) incurs
#F arithmetic ops.
#F
#F spiral> PrintMat(MatSPL(VScat_red(fId(4), 2)));
#F [ [ 1, 1,  ,  ,  ,  ,  ,   ],
#F   [  ,  , 1, 1,  ,  ,  ,   ],
#F   [  ,  ,  ,  , 1, 1,  ,   ],
#F   [  ,  ,  ,  ,  ,  , 1, 1 ] ]
#F
#F Some architectures have special instructions for this operation. 
#F For example, SSE4 has 'hadd' == horizontal add.
#F
#F ==========================================================================
Declare(VGath_dup);
Class(VGath_dup, VGath, rec(
    #-----------------------------------------------------------------------
    new := (self, func, v) >> SPL(WithBases(self,
        rec(func := func, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.v*self.func.domain(), self.func.range()],
    #-----------------------------------------------------------------------
    transpose := self >> Error("not yet supported"), #VScat_red(self.func, self.v),
    #-----------------------------------------------------------------------
    toAMat := self >> Tensor(
        Gath(self.func),
        Mat(TransposedMat([Replicate(self.v, 1)]))).toAMat(),
    #-----------------------------------------------------------------------
    toloop := (self, bksize) >> let(i := Ind(self.func.domain() / bksize),
	ISum(i, i.range,
            VScat(fTensor(fBase(i), fId(bksize)),self.v) *
            VGath_dup(fCompose(self.func, fTensor(fBase(i), fId(bksize))),self.v)))
    #-----------------------------------------------------------------------
));

#F ==========================================================================
#F VGath_dup_x_I(<func>, <v>) - gathers vector values and replicates <v> times
#F NOTE: EXPLAIN THIS
#F ==========================================================================
VGath_dup_x_I := (func, v) -> Cond(
    func.domain() = 1,
        VReplicate(v),
    # else
    let(i := Ind(func.domain()), 
	ISum(i, i.range,
	    VScat(fTensor(fBase(i),fId(v)), v) *
	    VReplicate(v) *
	    VGath(fBase(i),v)) *
        VGath(func,v)
    )
);

#F ==========================================================================
#F VGath_sv(<func>, <v>, <sv>) - vector gather (read) matrix on subvectors
#F NOTE: EXPLAIN THIS
#F ==========================================================================
Class(VGath_sv, BaseSVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],    # self >> [self.func, self.v, self.sv],
    rSetChild := rSetChildFields("func"), # rSetChildFields("func", "v", "sv"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v, self.sv, self.rem),
    #-----------------------------------------------------------------------
    dims := self >> [_roundup(self.sv * self.func.domain(), self.v),
                     self.sv * self.func.range() ],
    #-----------------------------------------------------------------------
    area := self >> 2 * self.func.domain() * self.sv,
    #-----------------------------------------------------------------------
    transpose := self >> VScat_sv(self.func, self.v, self.sv, self.rem),
    #-----------------------------------------------------------------------
    toAMat := self >> let(sv:=self.sv,
        n := EvalScalar(sv * self.func.domain()), N := EvalScalar(Cols(self)),
        nv := EvalScalar(Rows(self)), 
	func := fTensor(self.func, fId(sv)).lambda(),
        Scat(fAdd(nv, n, 0)).toAMat() *
        AMatMat(List([0..n-1], row -> BasisVec(N, EvalScalar(func.at(row)))))),
));

#F ==========================================================================
#F RCVGath_sv(<func>, <v>, <sv>) - vector gather (read) matrix on subvectors
#F NOTE: EXPLAIN THIS
#F ==========================================================================
Class(RCVGath_sv, BaseSVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],    # self >> [self.func, self.v, self.sv],
    rSetChild := rSetChildFields("func"), # rSetChildFields("func", "v", "sv"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v, self.sv, self.rem),
    #-----------------------------------------------------------------------
    dims := self >> let(
        n := self.func.domain() * self.sv, nv := _roundup(n, self.v),
	N := self.func.range() * self.sv,  
        2 * [nv, N]),
    #-----------------------------------------------------------------------
    area := self >> 2 * self.func.domain() * self.sv,
    #-----------------------------------------------------------------------
    transpose := self >> RCVScat_sv(self.func, self.v, self.sv, self.rem),
    #-----------------------------------------------------------------------
    toAMat := self >> let(
	sv := EvalScalar(self.sv), n := EvalScalar(self.func.domain()), 
	N  := EvalScalar(self.func.range()),
        func := fTensor(self.func, fId(sv), fId(2)).lambda(),
        Scat(fTensor(fAdd(_roundup(n, self.v), n, 0), fId(2))).toAMat() * 
	AMatMat(List([0..2*n-1], row -> BasisVec(2*N, EvalScalar(func.at(row).ev()))))),
));

#F ==========================================================================
#F VStretchGath(<func>, <part>, <v>) -
#F   vector gather (read) matrix on subvectors
#F   reads according to func but fills up entire vectors
#F NOTE: EXPLAIN THIS
#F ==========================================================================
Class(VStretchGath, BaseSVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.part, self.v),
    #-----------------------------------------------------------------------
    dims := self >> let(
	n  := self.func.domain(), N := self.func.range(), 
	nv := self.part * _roundup(n/self.part, self.v),
        [nv, N]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, func, part, v) >> SPL(WithBases(self,
        rec(func := func, part := part, v := v))).setDims(),
    #-----------------------------------------------------------------------
    area := self >> self.func.domain(),
    #-----------------------------------------------------------------------
    transpose := self >> VStretchScat(self.func, self.part, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.func, ", ",
        self.part, ", ", self.v, ")", self.printA()),
    #-----------------------------------------------------------------------
    toAMat := self >> let(
	n  := EvalScalar(self.func.domain()), 
	N  := EvalScalar(self.func.range()), 
	nv := self.part * _roundup(n/self.part, self.v),
        Tensor(I(self.part), Scat(fAdd(nv/self.part, n/self.part, 0))).toAMat() *
	AMatMat(List([0..n-1], row -> 
		    BasisVec(N, EvalScalar(self.func.at(row).ev()))))),
));


#F ==========================================================================
#F vRCStretchGath(<func>, <parts>, <v>) -
#F   vector gather (read) matrix on subvectors
#F   reads according to func but fills up entire vectors
#F NOTE: EXPLAIN THIS
#F ==========================================================================
Class(vRCStretchGath, BaseSVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.part, self.v),
    #-----------------------------------------------------------------------
    dims := self >> let(
	n  := self.func.domain(), N := self.func.range(), 
	nv := self.part * _roundup(n/self.part, self.v/2),
	2*[nv, N]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, func, part, v) >> SPL(WithBases(self, 
        rec(func := func, part := part, v := v))).setDims(),
    #-----------------------------------------------------------------------
    area := self >> 2 * self.func.domain(),
    #-----------------------------------------------------------------------
    transpose := self >> vRCStretchScat(self.func, self.part, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.func, ", ",
        self.part, ", ", self.v, ")", self.printA()),
    #-----------------------------------------------------------------------
    toAMat := self >> let(
	n  := self.func.domain(), N := self.func.range(), 
	nv := self.part * _roundup(n/self.part, self.v/2),
        func := fTensor(self.func, fId(2)),
        Tensor(I(self.part), 
	       Scat(fAdd(nv/self.part, n/self.part, 0)), I(2)).toAMat() *
	AMatMat(List([0..2*n-1], row -> BasisVec(2*N, EvalScalar(func.at(row).ev()))))),
));


#F ==========================================================================
#F RCVStretchGath(<func>, <parts>, <v>) -
#F   vector gather (read) matrix on subvectors
#F   reads according to func but fills up entire vectors
#F NOTE: EXPLAIN THIS
#F ==========================================================================
Class(RCVStretchGath, BaseSVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.part, self.v),
    #-----------------------------------------------------------------------
    dims := self >> let(
	n  := self.func.domain(), N := self.func.range(), 
	nv := self.part * _roundup(n/self.part, self.v),
	2*[nv, N]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, func, part, v) >> SPL(WithBases(self, 
        rec(func := func, part := part, v := v))).setDims(),
    #-----------------------------------------------------------------------
    area := self >> 2 * self.func.domain() * self.v,
    #-----------------------------------------------------------------------
    transpose := self >> RCVStretchScat(self.func, self.part, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.func, ", ",
        self.part, ", ", self.v, ")", self.printA()),
    #-----------------------------------------------------------------------
    toAMat := self >> Tensor(VStretchGath(self.func, self.part, self.v),
                             I(2)).toAMat(),
));


#F ==========================================================================
#F VGath_pc(N, n, ofs, v) - gather for unaligned partial contiguous vectors
#F   reads <n> contiguous values out of <N> at an offset <ofs>
#F   outputs ceil(<n> / <v>) vectors
#F
#F   .N   -- input size, only affects dimensions of VGath_pc matrix
#F   .n   -- how many contiguous points we are reading
#F   .ofs -- starting offset 
#F   .v   -- vector length
#F
#F Example:
#F
#F spiral> PrintMat(MatSPL(VGath_pc(5, 3, 0, 4)));
#F [ [ 1,  ,  ,  ,   ], 
#F   [  , 1,  ,  ,   ], 
#F   [  ,  , 1,  ,   ], 
#F   [  ,  ,  ,  ,   ] ]
#F
#F ==========================================================================
Class(VGath_pc, BaseSVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.N, self.n, self.ofs, self.v],
    rSetChild := rSetChildFields("N", "n", "ofs", "v"),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, N, n, ofs, v) >> SPL(WithBases(self, rec(
        N := N, n := n, ofs := ofs, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [ _roundup(self.n, self.v), self.N ],
    #-----------------------------------------------------------------------
    area := self >> self.n,
    #-----------------------------------------------------------------------
    transpose := self >> VScat_pc(self.N, self.n, self.ofs, self.v),
    #-----------------------------------------------------------------------
    toAMat := self >> let(n := self.n, N := self.N, nv := _roundup(n, self.v), 
        Scat(fAdd(nv, n, 0)).toAMat() * Gath(fAdd(N, n, self.ofs)).toAMat()),
    #-----------------------------------------------------------------------
    print :=  (self, i, is) >> Print(self.__name__, "(", self.N, 
        ", ", self.n, ", ", self.ofs, ", ", self.v, ")", self.printA()),
));

#F ==========================================================================
#F IxVGath_pc(k, N, n, ofs, v)
#F 
#F Performs multiple stacked VGath_pc operations
#F   
#F IxVGath_pc(k, N, n, ofs, v) = I(k) tensor VGath_pc(N, n, ofs, v)
#F
#F See also: VGath_pc
#F
#F Example: 
#F
#F spiral> PrintMat(MatSPL(VGath_pc(5, 3, 0, 4)));
#F [ [ 1,  ,  ,  ,   ], 
#F   [  , 1,  ,  ,   ], 
#F   [  ,  , 1,  ,   ], 
#F   [  ,  ,  ,  ,   ] ]
#F
#F spiral> PrintMat(MatSPL(IxVGath_pc(2, 5, 3, 0, 4)));
#F [ [ 1,  ,  ,  ,  ,  ,  ,  ,  ,   ], 
#F   [  , 1,  ,  ,  ,  ,  ,  ,  ,   ], 
#F   [  ,  , 1,  ,  ,  ,  ,  ,  ,   ], 
#F   [  ,  ,  ,  ,  ,  ,  ,  ,  ,   ], 
#F   [  ,  ,  ,  ,  , 1,  ,  ,  ,   ], 
#F   [  ,  ,  ,  ,  ,  , 1,  ,  ,   ], 
#F   [  ,  ,  ,  ,  ,  ,  , 1,  ,   ], 
#F   [  ,  ,  ,  ,  ,  ,  ,  ,  ,   ] ]
#F
#F ==========================================================================
Class(IxVGath_pc, BaseSVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.k, self.N, self.n, self.ofs, self.v],
    rSetChild := rSetChildFields("k", "N", "n", "ofs", "v"),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, k, N, n, ofs, v) >> SPL(WithBases(self, rec(
        k := k, N := N, n:= n, ofs := ofs, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [ self.k*_roundup(self.n, self.v), self.k*self.N ],
    #-----------------------------------------------------------------------
    area := self >> self.k * self.n, 
    #-----------------------------------------------------------------------
    transpose := self >> IxVScat_pc(self.k, self.N, self.n, self.ofs, self.v),
    #-----------------------------------------------------------------------
    toloop := (self, bksize) >> let(
	j := Ind(self.k),
	gath := VGath_pc(self.k*self.N, self.n, self.ofs + j*self.N, self.v),
	nvecs := Rows(gath) / self.v,
	ISum(j, j.range, 
             VScat(HH(self.k*nvecs, nvecs, j*nvecs, [1]), self.v) *
	     gath).split(bksize)),
    #-----------------------------------------------------------------------
    toAMat := self >> 
        Tensor(I(self.k), VGath_pc(self.N, self.n, self.ofs, self.v)).toAMat(),
    #-----------------------------------------------------------------------
    print :=  (self, i, is) >> Print(self.__name__, "(", self.k, ", ", self.N, 
        ", ", self.n, ", ", self.ofs, ", ", self.v, ")", self.printA()),
));

#F ==========================================================================
#F IxRCVGath_pc(k, N, n, ofs, v)
#F   vector gather (read) matrix reading unaligned
#F   partial contiguous COMPLEX vectors
#F
#F   IxRCVGath_pc(..params..) == IxVGath_pc(..params..) tensor I(2)
#F
#F  Note: IxRCVGath_pc(k, N, n, ofs, v) == IxVGath_pc(k, 2*N, 2*n, 2*ofs, 2*v)
#F        observe 2*v inside IxVGath_pc.
#F
#F        IxVGath_pc(j, 2*N, 2*n, 2*ofs, v), however, is not exactly equal
#F        it has a different # of rows, if n < v/2
#F ==========================================================================
Class(IxRCVGath_pc, BaseSVGath, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.k, self.N, self.n, self.ofs, self.v],
    rSetChild := rSetChildFields("k", "N", "n", "ofs", "v"),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, k, N, n, ofs, v) >> SPL(WithBases(self, rec(
        k := k, N := N, n := n, ofs := ofs, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [ 2*self.k*_roundup(self.n, self.v),  2*self.k*self.N ],
    #-----------------------------------------------------------------------
    area := self >> 2 * self.k * self.n, 
    #-----------------------------------------------------------------------
    transpose := self >> IxRCVScat_pc(self.k, self.N, self.n, self.ofs, self.v),
    #-----------------------------------------------------------------------
    toAMat := self >> 
        Tensor(IxVGath_pc(self.k, self.N, self.n, self.ofs, self.v), I(2)).toAMat(),
    #-----------------------------------------------------------------------
    print :=  (self, i, is) >> Print(self.__name__, "(", self.k, ", ", self.N, 
        ", ", self.n, ", ", self.ofs, ", ", self.v, ")", self.printA()),
));
