
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# 2-way complex implementation of DFT via DST1 using new method
# (gives native 4-way DFT for any size)
# Also 2-way real implementation of RDFT via DST1, again for any size
#
ImportAll(paradigms.vector);

bc := function(n)
    local l, r, top, k, odd, even;
    k := Int(n/2);
    odd := When(IsOddInt(n), 1, 0);
    even := When(IsEvenInt(n), 1, 0);

    l := Diag(List([1..2*(k-even)], i->(-1)^i)) *
         Gath(fCompose( fAdd(2*k+odd, k+odd, 1), 
                        fDirsum(fId(1), fTensor(fId(k-1-even), fConst(1,2,0)), fId(1)), 
                        fTensor(fId(k-even), J(2))));

    r := Diag(List(1+[1..2*(k-even)], i->(-1)^i)) *
         Gath(fCompose( OS(2*k+odd, -1), 
                        fAdd(2*k+odd, k+odd, 0), 
                        fDirsum(fId(1), fTensor(fId(k-1-even), fConst(1,2,0)), fId(1)) )); 

    top := Cond(even=1, VBase(F(2), 2) * Tensor(RowVec(fConst(TReal, k, 1)), I(2)),
                        Tensor(Mat([[1,1]]), I(2)) * 
                        DirectSum(Tensor(RowVec(fConst(TReal, k, 1)), I(2)), VGath_sv(fId(1), 2, 1) ));

    return VStack(top, Tensor(Mat([[1, 1]]), I(2*(k-even))) * VStack(l, r));
end;

icomb := k -> let(j:=Ind(Int((k-1)/2)), a:=fdiv((j+1), k), c:=cospi(2*a), s:=sinpi(2*a), 
    IDirSum(j, 
            F(2) * Diag(1, E(4)) * Mat([[ 1/s, c/s ], 
                                        [   0,   1 ]])));

rcomb := k -> let(j:=Ind(Int((k-1)/2)), a:=fdiv((j+1), k), c:=cospi(2*a), s:=sinpi(2*a), 
    IDirSum(j, 
            Mat([[ 1/s, c/s ], 
                 [   0,   1 ]])));

mydft := k -> let(hf := Int((k-1)/2), even := IsEvenInt(k), dst := Cond(even, DST1(hf), DST5(hf)),
    Cond(even, Kp(k, 2), DirectSum(VScat_sv(fId(1), 2, 1), K(k-1, 2))) *
    DirectSum(Cond(even, I(2),  Tensor(Mat([[1, 1]]), I(2)) * VStack(I(2), VBase(J(2), 2))),    #Mat([[1, 1]])),
              icomb(k) * Tensor(dst, I(2))) *
    bc(k)
);

myrdft := k -> let(hf := Int((k-1)/2), even := IsEvenInt(k), dst := Cond(even, DST1(hf), DST5(hf)),
                               #Mat([[1, 1]])),
    DirectSum(Cond(even, I(2), VScat_sv(fId(1), 2, 1) * Tensor(Mat([[1, 1]]), I(2)) * VStack(I(2), VBase(J(2), 2))),
              rcomb(k) * Tensor(dst, I(2))) *
    bc(k)
);

Class(RulesAutoTag, RulesTag);
RewriteRules(RulesAutoTag, rec(
        # override TTag_drop to drop the tag only on constructs that are already vector constructs
    TTag_drop := Rule([TTag, @(1, Concatenation(_VConstructL, _VConstructR)), @, [ListClass, @(2, [AVecLib]) ]],
        e -> @(1).val),

    TTag_Gath := Rule([TTag, @(1, [Prm, Gath]), @, [ListClass, @(2, [AVecLib]) ]], e -> 
        Cond(_divides(e.tags[1].params[1], domain(@(1).val.func)), 
             VGath_sv(@(1).val.func, e.tags[1].params[1], 1),
             @(1).val)),

    TTag_Scat := Rule([TTag, @(1, [Scat]), @, [ListClass, @(2, [AVecLib])]], e -> 
        Cond(_divides(e.tags[1].params[1], domain(@(1).val.func)), 
             VScat_sv(@(1).val.func, e.tags[1].params[1], 1),
             @(1).val)),

    TTag_diag := Rule([TTag, @(1, Diag), @, [ListClass, @(2, [AVecLib])]], e -> 
        VDiag(@(1).val.element, e.tags[1].params[1])),

    GT_Vec := Rule(@(1, GT, e->e.firstTag().kind() = AVecLib), e ->
        Cond(# divisible by v
             GT_HVec.applicable(e), GT_HVec.apply(e, 0, 0), 
             # non-divisible
             GT_HVecND.apply(e, 0, 0))),

    GT_VJam_Stride1 := Rule(@(1, GT, e -> e.firstTag().kind() in [AVJamL, AVJamR] and 
                                          GT_VJam_Stride1_Franz.applicable(e)), 
        e -> GT_VJam_Stride1_Franz.apply(e, GT_VJam_Stride1_Franz.child(e, 0), 0)), 

    TL_Vec := Rule(@(1, TL, e->e.firstTag().kind()=AVecLib), e -> e.withoutFirstTag().withTags([
                Cond(e.firstTag().params[1]=2, AVecReg(SSE_2x64f),
                     e.firstTag().params[1]=4, AVecReg(SSE_4x32f)) ])),

    VJam_NT1 := Rule([@(1, VJam), @(2).cond(e->IsNonTerminal(e) and _rank(e)=0), @(3)],
        e -> VTensor(e.child(1), @(3).val))
));

RulesAutoVectorize := MergedRuleSet(RulesAutolibStandard, RulesAutoTag);

AutoLibDefaults.vectorizeSUM:=true; # NOTE

prep0 := s -> ApplyStrategy(s,
    [ RulesAutolibStandard,   toLambdaWrap, RulesAutoVectorize ], 
    UntilDone, AutoLibDefaults);

prep := s -> ApplyStrategy(s,
    [ RulesAutolibStandard,   toLambdaWrap,      RulesAutoVectorize, x->EliminateGT(x, []),
      RulesUnsafeVectorizeH,  RulesLibBCPrepare, _removeLambdaWrap ],
    UntilDone, AutoLibDefaults);

vdft  := k -> prep(TTag(AutoLibSumsGen_HH(mydft(k), SpiralDefaults)).withTags([AVecLib(2)]));
vdft0 := k -> prep0(TTag(AutoLibSumsGen_HH(mydft(k), SpiralDefaults)).withTags([AVecLib(2)]));

vrdft  := k -> prep(TTag(AutoLibSumsGen_HH(myrdft(k), SpiralDefaults)).withTags([AVecLib(2)]));
vrdft0 := k -> prep0(TTag(AutoLibSumsGen_HH(myrdft(k), SpiralDefaults)).withTags([AVecLib(2)]));

implement_dst := s -> SubstTopDownNR(s, @(1, [DST1,DST5]), e ->
    SPLAMat(PRealMatrixDecompositionByMonMonSymmetry(MatSPL(e))));


NonTerminal.getTags := self >> [];
vopts := SIMDGlobals.getOpts(SSE_2x64f);
Add(vopts.compileStrategy, fixup);
Add(vopts.includes, "<include/complex_gcc_sse.h>");
vopts.breakdownRules := CopyFields(vopts.breakdownRules, rec(
        DCT1:=SpiralDefaults.breakdownRules.DCT1,
        DST1:=SpiralDefaults.breakdownRules.DST1,
        DCT2:=SpiralDefaults.breakdownRules.DCT2,
        DST2:=SpiralDefaults.breakdownRules.DST2,
        DCT3:=SpiralDefaults.breakdownRules.DCT3,
        #DST3:=SpiralDefaults.breakdownRules.DST3,
        DCT4:=SpiralDefaults.breakdownRules.DCT4,
        DST4:=SpiralDefaults.breakdownRules.DST4,
        RDFT:=SpiralDefaults.breakdownRules.RDFT,
        DFT := [DFT_Base, DFT_CT, DFT_Rader, DFT_PD, DFT_GoodThomas],
        SRDFT1 := [SRDFT1_toPRDFT1],
        SRDFT3 := [SRDFT3_toPRDFT3]
));

vopts.XType := TComplex;
vopts.YType := TComplex;
vopts.unparser := CMacroUnparserProg;
Unbind(vopts.profile);
vopts.language := "c.icc.opt";
vopts.customDataType := "float_cplx";
vopts.includes := vopts.includes{[1,3,4,5,6,7,8,9]};


rvopts := CopyFields(vopts, rec(XType := TReal, YType := TReal, unparser := SSEUnparser,
                                customDataType := "double",
                                includes := DropLast(vopts.includes, 1), 
                                generateComplexCode :=false));



obj := AutoLibSumsGen_HH(Tensor(I(8), F(2)), opts).withTags([AVecLib(2)]);
obj := Descend(obj, GT_HVec, opts).child(1);
Descend(obj, GT_VJam_Stride1_Franz, opts);

obj := AutoLibSumsGen_HH(Tensor(I(2), F(2)), opts).withTags([AVJamL(2)]);
Descend(obj, GT_VJam_Stride1_Franz, opts);

s := vdft(6);

ss:=RandomRuleTree(implement_dst(s), SpiralDefaults);
ss:=SumsSPL(SPLRuleTree(ss), vopts);
ss:=ApplyStrategy(ss, [StandardVecRules, RulesUnsafeVectorizeH, RulesMergedStrengthReduce],
                  UntilDone, vopts); 
c:=CodeSums(ss, vopts);

ss:=0; c:=0;
measure:=function(what, n, num, vopts)
    local res, s, i, t;
    res := [];
    s := implement_dst(what(n));
    for i in [1..num] do 
        ss:=RandomRuleTree(s, SpiralDefaults);
        ss:=SumsSPL(SPLRuleTree(ss), vopts);
        ss:=ApplyStrategy(ss, [StandardVecRules, RulesUnsafeVectorizeH, RulesMergedStrengthReduce],
                  UntilDone, vopts); 
        c:=CodeSums(ss, vopts);
        t := CMeasure(c, vopts);
        Add(res, t);
        PrintLine("n ", n, "  i ", i, "  cyc ", t);
        AppendTo("dft_via_dst.txt", 
        PrintLine("n ", n, "  i ", i, "  cyc ", t));
    od;
    return res;
end;

allres := [];
#for n in [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40] do
for n in [5..56] do
    t:=Try(measure(vdft, n, 5, vopts));
    Add(allres, t);
od;
