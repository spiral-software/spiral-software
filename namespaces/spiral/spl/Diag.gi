
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ==========================================================================
#F Diag(<diag-func>)
#F ==========================================================================
Class(Diag, BaseMat, rec(
    _short_print := true,
    abbrevs := [ arg -> When(Length(arg)=1,
                         When(IsFunction(arg[1]) or IsFuncExp(arg[1]) or (IsRec(arg[1]) and not IsScalar(arg[1])) or IsList(arg[1]), [arg[1]], [[arg[1]]]),
                 [Flat(arg)]) ],

    new := meth(self, L)
        if IsList(L) then 
            L := List(L, toExpArg); 
            L := FList(UnifyTypes(List(L, x->x.t)), L);
        elif not IsRec(L) or not IsBound(L.lambda) then
            Error("List of numbers or LambdaList expected");
        fi;
        return SPL(WithBases(self, rec(element := L, TType := L.range()))).setDims();
    end,
    #-----------------------------------------------------------------------
    dims := self >> Replicate(2, self.element.domain()),
    #-----------------------------------------------------------------------
    isPermutation := self >> false,
    #-----------------------------------------------------------------------
    isReal := self >> let(
	t := self.element.range(), 
        tt := Cond(IsVecT(t), t.t, t),
	tt <> TComplex and ObjId(tt)<>T_Complex),
    #-----------------------------------------------------------------------
    toAMat := self >> DiagonalAMat(List(EvalScalar(self.element.tolist()), EvalScalar)),
    #-----------------------------------------------------------------------
    transpose := self >> self,
    conjTranspose := self >> Diag(FConj(self.element)),
    inverse := self >> Diag(FInv(self.element)),
    #-----------------------------------------------------------------------
    normalizedArithCost := self >> 0,
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >> Sum(self.element.tolist(), costMul),
    #-----------------------------------------------------------------------
    toloop := (self, bksize) >> let(
	j := Ind(self.element.domain()),
	ISum(j, 
	    Scat(fTensor(fBase(j), fId(1))) *
	    Diag(fCompose(self.element, fTensor(fBase(j), fId(1)))) *
	    Gath(fTensor(fBase(j), fId(1)))
	).split(bksize)
    )

));


#F ==========================================================================
#F DiagCpxSplit(<diag-func>) - complex diagonal specified in split format 
#F NOTE: maybe this should be somehow replaced by VRCDiag (?)
#F  this object is helpful for complex code, because it improves performance
#   when using SSE
#F ==========================================================================
Class(DiagCpxSplit, Diag, rec(
    #-----------------------------------------------------------------------
    dims := self >> Replicate(2, div(self.element.domain(), 2)),
    #-----------------------------------------------------------------------
    isReal := self >> false, 
    #-----------------------------------------------------------------------
    toAMat := meth(self)
        local lst, i, cpxlst;
        lst := List(EvalScalar(self.element.tolist()), EvalScalar);
        cpxlst := [];
        for i in [0..Length(lst)/2-1] do
            Add(cpxlst, lst[2*i+1]+E(4)*lst[2*i+2]);
        od;
        return DiagonalAMat(cpxlst);
    end,
    #-----------------------------------------------------------------------
    transpose := self >> self,
    conjTranspose := self >> ObjId(self)(FRConj(self.element)),
    #-----------------------------------------------------------------------
    normalizedArithCost := self >> 0,
    #-----------------------------------------------------------------------
    #arithmeticCost := (self, costMul, costAddMul) >> Sum(self.element.tolist(), costMul)
));

Declare(RowVec, ColVec);

Class(RowVec, Diag, rec(
    dims := self >> [1, self.element.domain()],
    toAMat := self >> AMatMat([List(self.element.tolist(), EvalScalar)]),
    transpose := self >> ColVec(self.element),
    conjTranspose := self >> ColVec(FConj(self.element)),
    arithmeticCost := (self, costMul, costAddMul) >> Sum(self.element.tolist(), costAddMul),

    toBlk := self >> Checked(not IsSymbolic(self.element.domain()), Blk([self.element.tolist()])),
    toDiagBlk := self >> Checked(not IsSymbolic(self.element.domain()), 
	Blk([Replicate(EvalScalar(self.element.domain()), 1)]) * Diag(self.element)),
));

Class(ColVec, Diag, rec(
    dims := self >> [self.element.domain(), 1],
    toAMat := self >> AMatMat(TransposedMat([List(self.element.tolist(), EvalScalar)])),
    transpose := self >> RowVec(self.element),
    conjTranspose := self >> RowVec(FConj(self.element)),

    toBlk := self >> self.transpose().toBlk().transpose(),
    toDiagBlk := self >> self.transpose().toDiagBlk().transpose(),
));

Declare(RCDiag);

#F RCDiagonalAMat(<elts>, <post>)
#F
#F Works similarly to RCDiag, but creates an AMat object instead of SPL.
#F See Doc(RCDiag).
#F
#F <elts> must be a list of size 2*n, which consists of interleaved real/imaginary data. 
#F (not a symbolic function as in RCDiag)
#F
#F <post> must be an <amat>, and not spl
#F
RCDiagonalAMat := (elts, post) -> 
    DirectSumAMat(List([1..Length(elts)/2], x -> 
            let(r := EvalScalar(elts[2*x-1]), i := EvalScalar(elts[2*x]),
                post * AMatMat([[r,-i],[i,r]]))));

#F RCDiag(<diagfunc>)
#F RCDiag(<diagfunc>, <2x2 spl>)
#F
#F  Complex diagonal operating on real data in (r,i) format.
#F  Represents a block diagonal with 2x2 blocks (complex rotations),
#F  with post-processing matrix <post-spl> applied to each block.
#F
#F  For n complex numbers, <diagfunc> must be a diagonal function of size 2*n,
#F  which consists of interleaved real/imaginary data. 
#F
#F  Example: RCDiag(RCData(Tw1(8,4,1))              normal twiddle diagonal
#F           RCDiag(RCData(Tw1(8,4,1)), 1/2*F(2))   twiddles for Hartley transform
#F
Class(RCDiag, BaseMat, rec(
    dims := self >> Replicate(2, self.element.domain()),
    area := self >> 2*self.element.domain(),
    transpose := self >> RCDiag(FRConj(self.element), self.post),

    conjTranspose := self >> Cond(self.element.range()=TComplex,
        RCDiag(FRConj(FConj(self.element)), self.post),
        RCDiag(FRConj(self.element), self.post)),

    toAMat := self >> let(elts := self.element.tolist(), # r,i,r,i,...
        Checked(Length(elts) mod 2 = 0,
            DirectSumAMat(List([1..Length(elts)/2], x -> 
                    let(r := EvalScalar(elts[2*x-1]), i := EvalScalar(elts[2*x]),
                        self.post.toAMat() * AMatMat([[r,-i],[i,r]])))))),

    toloop := self >> let(
        func := self.element, # r,i,r,i,...
        j    := Ind(idiv(func.domain(),2)),
        re   := func.at(2*j),
        im   := func.at(2*j+1),
        IDirSum(j, j.range,
                self.post * Mat([[re, -im],
                                 [im,  re]]))),

    abbrevs := [
        () -> Error("Usage: \n  RCDiag(<diagfunc>, <2x2 spl>)\n",
                    "  RCDiag(<diagfunc>)\n",
                    "  RCDiag(<e0>, <e1>, ...)\n"),
        (L)    -> Checked(IsList(L) or IsFunction(L), [L, I(2)]),
        (L, P) -> Checked(IsList(L) or IsFunction(L), IsSPL(P), [L, P])
    ],

    new := meth(self, L, P)
        local inline, res;
        if IsList(L) then L := FList(L);
        elif not IsRec(L) or not IsBound(L.lambda) then
            Error("List of numbers or LambdaList expected");
        fi;
        res := SPL(WithBases(self, rec(element := L, post := P, TType := L.range())));
        res.dimensions := res.dims();
        return res;
    end,

    isReal := self >> self.element.range() <> TComplex and self.post.isReal(),
    rChildren := self >> [self.element, self.post],
    rSetChild := rSetChildFields("element", "post"),

    normalizedArithCost := self >> 0,
));

# Won't work for now because divisions by zero need to be handled.
# Declare(RCDiagLiftingSteps);
#
# #F RCDiagLiftingSteps(<diagfunc>)
# #F RCDiagLiftingSteps(<diagfunc>, <2x2 spl>)
# #F
# #F  Complex diagonal operating on real data in (r,i) format.
# #F  Represents a block diagonal with 2x2 blocks (complex rotations),
# #F  with post-processing matrix <post-spl> applied to each block.
# #F
# #F  Example: RCDiagLiftingSteps(T(8,4))   normal twiddle diagonal
# #F           RCDiagLiftingSteps(T(8,4), 1/2*F(2))   twiddles for Hartley transform
# #F
#
# Class(RCDiagLiftingSteps, BaseMat, rec(
#     dims := self >> Replicate(2, self.element.domain()),
#
#     transpose := self >> RCDiagLiftingSteps(FRConj(self.element), self.post),
#     conjTranspose := self >> self.transpose(),
#
#     toAMat := self >> let(elts := self.element.tolist(), # r,i,r,i,...
#   Checked(Length(elts) mod 2 = 0,
#       DirectSumAMat(List([1..Length(elts)/2],
#       x -> let(r := EvalScalar(elts[2*x-1]), i := EvalScalar(elts[2*x]),
#           self.post.toAMat() * AMatMat([[r,-i],[i,r]])))))),
#
#     abbrevs := [
#   () -> Error("Usage: \n  RCDiagLiftingSteps(<diagfunc>, <2x2 spl>)\n",
#          "  RCDiagLiftingSteps(<diagfunc>)\n",
#          "  RCDiagLiftingSteps(<e0>, <e1>, ...)\n"),
#   (L)    -> Checked(IsList(L) or IsFunction(L), [L, I(2)]),
#   (L, P) -> Checked(IsList(L) or IsFunction(L), IsSPL(P), [L, P])
#     ],
#
#     new := meth(self, L, P)
#         local inline, res;
#         if IsList(L) then L := FList(L);
#   elif not IsRec(L) or not IsBound(L.lambda) then
#       Error("List of numbers or LambdaList expected");
#   fi;
#   inline := IsBound(L.inline) and L.inline;
#         res := SPL(WithBases(self, rec(element := L, post := P, inline := inline)));
#   res.dimensions := res.dims();
#   return res;
#     end,
#
#     isReal := True,
#     rChildren := self >> [self.element, self.post],
#     rSetChild := rSetChildFields("element", "post"),
#     print := (self, i, is) >> Print(self.name, "(", self.element,
#         When(ObjId(self.post)<>I, Print(", ", self.post)), ")")
# ));
