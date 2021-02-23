
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# NOTE: MISSING (YV)
Class(VScat_red);

Class(BaseVScat, BaseMat, SumsBase, rec(
    isVScat := true,
    sums := self >> self,
    dims := self >> self.dimensions,
    isReal := self >> true,
    needInterleavedLeft := False,
    needInterleavedRight := False,
    area := self >> self.transpose().area(),
    toAMat := self >> TransposedAMat(self.transpose().toAMat()),
    conjTranspose := self >> self.transpose(), 
));

Class(BaseSVScat, BaseMat, SumsBase, rec(
    isVScat := true,
    isSVScat := true,
    #-----------------------------------------------------------------------
    abbrevs        := BaseSVGath.abbrevs, 
    new            := BaseSVGath.new, 
    print          := BaseSVGath.print,
    getConstantRem := BaseSVGath.getConstantRem,
    #-----------------------------------------------------------------------
    dims := self >> Error(ObjId(self), ".dims() is undefined"),
    #-----------------------------------------------------------------------
    sums := self >> self,
    isReal := self >> true,
    needInterleavedLeft := True,
    needInterleavedRight := False,
    #-----------------------------------------------------------------------
    conjTranspose := self >> self.transpose(), 
    area := self >> self.transpose().area(),
    toAMat := self >> TransposedAMat(self.transpose().toAMat())
));

IsVScat := x -> IsRec(x) and IsBound(x.isVScat) and x.isVScat;
IsSVScat := x -> IsRec(x) and IsBound(x.isSVScat) and x.isSVScat;

# ==========================================================================
# VScat(<func>, <v>) - vector scatter (write) matrix, assumes aligned output
# ==========================================================================
Class(VScat, BaseVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v),
    #-----------------------------------------------------------------------
    new := (self, func, v) >> SPL(WithBases(self, 
        rec(func := func, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.v * self.func.range(), self.v * self.func.domain()],
    #-----------------------------------------------------------------------
    transpose := self >> VGath(self.func, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.func, ", ", self.v, ")", self.printA()),
 ));

# ==========================================================================
# VScat_u(<func>, <v>) -- same as VScat but assumes that output is unaligned
# ==========================================================================
Class(VScat_u, VScat, rec(
    dims := self >> [ self.func.range(), self.v * self.func.domain()],
    transpose := self >> VGath_u(self.func, self.v)
));

# ==========================================================================
# VScat_zero(N, n, <v>) - vector scatter (write) matrix
# ==========================================================================
Class(VScat_zero, BaseVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [],
    rSetChild := rSetChildFields(),
    from_rChildren := (self, rch) >> ObjId(self)(self.N, self.n, self.v),
    #-----------------------------------------------------------------------
    new := (self, N, n, v) >> SPL(WithBases(self,
        rec(n := n, N := N, v := v))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.v * self.N, self.v * self.n],
    #-----------------------------------------------------------------------
    transpose := self >> VGath_zero(self.N, self.n, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.N, ", ",
        self.n, ", ", self.v,")", self.printA()),
 ));

# ==========================================================================
# VScat_sv(<func>, <v>, <sv>) - vector scatter (write) matrix on subvectors
# ==========================================================================
Class(VScat_sv, BaseSVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],     # self >> [self.func, self.v, self.sv],
    rSetChild := rSetChildFields("func"), # rSetChildFields("func", "v", "sv"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v, self.sv, self.rem),
    #-----------------------------------------------------------------------
    dims := self >> [self.sv * self.func.range(),
                     _roundup(self.sv * self.func.domain(), self.v)],
    #-----------------------------------------------------------------------
    transpose := self >> VGath_sv(self.func, self.v, self.sv, self.rem),
));

# ==========================================================================
# RCVScat_sv(<func>, <v>, <sv>) - vector scatter (write) matrix on subvectors
# ==========================================================================
Class(RCVScat_sv, BaseSVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],     # self >> [self.func, self.v, self.sv],
    rSetChild := rSetChildFields("func"), # rSetChildFields("func", "v", "sv"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v, self.sv, self.rem),
    #-----------------------------------------------------------------------
    dims := self >> let(
	n := self.func.domain() * self.sv,  nv := _roundup(n, self.v), 
	N := self.func.range() * self.sv, 
	[2*N, 2*nv]),
    #-----------------------------------------------------------------------
    transpose := self >> RCVGath_sv(self.func, self.v, self.sv, self.rem),
));

# ==========================================================================
# VStretchScat(<func>, part, <v>) - vector scatter (write) matrix on subvectors
# ==========================================================================
Class(VStretchScat, BaseSVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.part, self.v),
    #-----------------------------------------------------------------------
    dims := self >> let(
	n  := self.func.domain(), N := self.func.range(), 
	nv := self.part * _roundup(n / self.part, self.v),
        [N, nv]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, func, part, v) >> SPL(WithBases(self, 
	rec(func := func, part := part, v := v))).setDims(),
    #-----------------------------------------------------------------------
    transpose := self >> VStretchGath(self.func, self.part, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.func, ", ",
        self.part, ", ", self.v, ")", self.printA()),
));


# ==========================================================================
# vRCStretchScat(<func>, part, <v>) - vector scatter (write) matrix on subvectors
# ==========================================================================
Class(vRCStretchScat, BaseSVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.part, self.v),
    #-----------------------------------------------------------------------
    dims := self >> let(
	n  := self.func.domain(), N := self.func.range(), 
	nv := self.part * _roundup(n / self.part, self.v/2),
        2*[N, nv]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, func, part, v) >> SPL(WithBases(self, 
        rec(func := func, part := part, v := v))).setDims(),
    #-----------------------------------------------------------------------
    transpose := self >> vRCStretchGath(self.func, self.part, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.func, ", ",
        self.part, ", ", self.v, ")", self.printA()),
));

# ==========================================================================
# RCVStretchScat(<func>, part, <v>) - vector scatter (write) matrix on subvectors
# ==========================================================================
Class(RCVStretchScat, BaseSVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.part, self.v),
    #-----------------------------------------------------------------------
    dims := self >> let(
	n  := self.func.domain(), N := self.func.range(), 
	nv := self.part * _roundup(n / self.part, self.v),
        2*[N, nv]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, func, part, v) >> SPL(WithBases(self, rec(
        func := func, part := part, v := v))).setDims(),
    #-----------------------------------------------------------------------
    transpose := self >> RCVStretchGath(self.func, self.part, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.func, ", ",
        self.part, ", ", self.v, ")", self.printA()),
 ));

# ==========================================================================
# VScat_pc(N, n, ofs, v) - vector scatter (write) matrix writing unaligned
#                         partial contiguous vectors
# ==========================================================================
Class(VScat_pc, BaseSVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.N, self.n, self.ofs, self.v],
    rSetChild := rSetChildFields("N", "n", "ofs", "v"),
    #-----------------------------------------------------------------------
    dims := self >> let(nv := _roundup(self.n, self.v), [self.N, nv]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, N, n, ofs, v) >> SPL(WithBases(self, rec(
        N := N, n:= n, ofs := ofs, v := v))).setDims(),
    #-----------------------------------------------------------------------
    transpose := self >> VGath_pc(self.N, self.n, self.ofs, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.N, ", ", self.n, ", ",
        self.ofs, ", ", self.v,")", self.printA()),
));

# ==========================================================================
# IxVScat_pc(k, N, n, ofs, v) - vector gather (read)
#   matrix reading unaligned
#   partial REAL contiguous vectors
# ==========================================================================
Class(IxVScat_pc, BaseSVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.k, self.N, self.n, self.ofs, self.v],
    rSetChild := rSetChildFields("k", "N", "n", "ofs", "v"),

    from_rChildren := (self, rch) >> ObjId(self)(self.k, self.N, self.n, self.ofs, self.v),
    #-----------------------------------------------------------------------
    dims := self >> let(nv := _roundup(self.n, self.v), [self.k*self.N, self.k*nv]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, k, N, n, ofs, v) >> SPL(WithBases(self, rec(
        k := k, N := N, n:= n, ofs := ofs, v := v))).setDims(),
    #-----------------------------------------------------------------------
    transpose := self >> IxVGath_pc(self.k, self.N, self.n, self.ofs, self.v),
    #-----------------------------------------------------------------------
    toloop := (self, bksize) >> self.transpose().toloop(bksize).transpose(),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.k, ", ", self.N, ", ",
        self.n, ", ", self.ofs, ", ", self.v,")", self.printA()),
));

# ==========================================================================
# IxRCVScat_pc(k, N, n, ofs, v) - vector gather (read)
#   matrix reading unaligned
#   partial REAL contiguous vectors
# ==========================================================================
Class(IxRCVScat_pc, BaseSVScat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [],         # self >> [self.ofs],  # SEEMS NOT TO WORK?!
    rSetChild := rSetChildFields(),  #  rSetChildFields("ofs"),
    from_rChildren := (self, rch) >> ObjId(self)(self.k, self.N, self.n, self.ofs, self.v),
    #-----------------------------------------------------------------------
    dims := self >> let(nv := _roundup(self.n, self.v), 2 * [self.k*self.N, self.k*nv]),
    #-----------------------------------------------------------------------
    abbrevs := [],
    new := (self, k, N, n, ofs, v) >> SPL(WithBases(self, rec(
        k := k, N := N, n := n, ofs := ofs, v := v))).setDims(),
    #-----------------------------------------------------------------------
    transpose := self >> IxRCVGath_pc(self.k, self.N, self.n, self.ofs, self.v),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.__name__, "(", self.k, ", ", self.N, ", ",
        self.n, ", ", self.ofs, ", ", self.v,")", self.printA()),
));


# ==========================================================================
# Accumulative scatter versions
# NOTE: to be replaced by a Sigma-SPL lowering step, CodeSumsAcc, and 
#        a specialized CodegenAcc
#
# NOTE(!!!): .transpose().transpose() on below classes will lead to non-accumulative
#             versions, and thus invalid code!
Class(VScatAcc,    VScat,    rec());
Class(VScatAcc_u,  VScat_u,  rec());
Class(VScat_svAcc, VScat_sv, rec());
Class(VScat_pcAcc, VScat_pc, rec());


# ==========================================================================
# Methods for vectorization  of ScatGath constructs
# NOTE: This is a hack

# BB needed, as basic block size is not known due to varying function domain
ScatGath.toloopRCVec := (self, _bksize, v) >> let(
    bksize := When(_bksize = 1, v, _bksize),
    dom    := RulesMergedStrengthReduce(_divideFunc(self.gfunc.domain(), bksize)), 
    i      := Ind(dom), #ok := Error("caught"),
    sfunc  := RulesFuncSimp(self.sfunc),
    ISum(i, dom,
        BB(
            Compose(When(ObjId(sfunc)=fId, [], [RCVScat_sv(sfunc, v, 1)]) ::
		    [VScat(fTensor(fBase(i), fId(2*bksize/v)), v)]) *
            RCVGath_sv(fCompose(self.gfunc, fTensor(fBase(i), fId(bksize))).setDomain(bksize), v, 1)
        )
    )
);

ScatGath.toloopVec := (self, _bksize, v) >> let(
    bksize := When(_bksize = 1, v, _bksize),
    dom    := RulesMergedStrengthReduce(_divideFunc(self.gfunc.domain(), bksize)), 
    i      := Ind(dom),
    sfunc  := RulesFuncSimp(self.sfunc),
    ISum(i, dom,
        BB(
            Compose(When(ObjId(sfunc)=fId, [], [VScat_sv(sfunc, v, 1)]) :: 
		    [ VScat(fTensor(fBase(i), fId(bksize/v)), v)]) *
            VGath_sv(fCompose(self.gfunc, fTensor(fBase(i), fId(bksize))).setDomain(bksize), v, 1)
        )
    )
);
