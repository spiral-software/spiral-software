
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(paradigms.common);
ImportAll(paradigms.vector);
Import(formgen);

buildBench := function(sizes, isa, optrec, filter, name)
    local opts, goodsizes, dpbench, exp;

    opts := SIMDGlobals.getOpts(isa, optrec);
    goodsizes := Filtered(sizes, i->let(rt := RandomRuleTreeDP(TRC(DFT(i)).withTags(opts.tags), opts), IsRuleTree(rt) and rt.children[1].children[1].rule in filter));
    dpbench :=  doSimdDft(goodsizes, isa, optrec);
    exp := Filtered(RecFields(dpbench.exp), i-> not i in SystemRecFields)[1];

    dpbench.exp.(exp).name := Concat(exp, "_", name);

    dpbench.fileTransform := (exp, t, opts) -> Concat(isa.name, "_", name, "_", "dft_ic_", StringInt(Rows(t)/2), ".c");
    dpbench.funcTransform := (exp, t, opts) -> Concat("dft", StringInt(Rows(t)/2));
    dpbench.txtFileName   := (exp, runMethod) -> Concat(isa.name, "_", name, "_dft_ic_", SubString(runMethod, 5), ".txt");
    dpbench.exp.(exp).hashFile := Concat(isa.name, "_", name, "_dft_ic_", "DP", ".hash");
    dpbench.exp.(exp).outputVecStatistics := true;

    return dpbench;
end;


buildExperiments := function(isas, sizes, setups)
    local benches, isa, vect, vmode, exp, b;

    benches := [];
    for isa in isas do
        for vect in SIMDGlobals.experiments.vects do
            for vmode in SIMDGlobals.experiments.vmodes do
                for exp in setups do
                    b := buildBench(sizes, isa, ApplyFunc(CopyFields, [vmode, vect]::exp[1]), exp[2], Concat(vect.name, "_", vmode.name, "_", exp[3]));
                    Add(benches, b);
                    PrintLine(isa, ", ", vect.name, ", ", vmode.name, ", ", exp[3], "", ": ", List(b.exp.(Filtered(RecFields(b.exp), i-> not i in SystemRecFields)[1]).benchTransforms, i->Rows(i)/2));
                od;
            od;
        od;
    od;

    return benches;
end;

# preset for unrolled DFT code
Class(unrollDFT, rec(
    # scalar algorithms
    PD := true,
    PFA := true,
    PFA_maxSize := 10000,
    RealRader := true,
    Rader := true,
    Rader_maxSize := 10000,
    PRDFT:=true,
    URDFT:= true,
    URDFT_maxRadix := 10000,
    CT := true,
    minCost := true,
    Mincost_maxSize := 10000,
    splitRadix := false,
    SplitRadix_maxSize := 10000,
    # basic vector configuration
    stdTTensor := false,
    pushTag := true,
    flipIxA := true,
    useConj := false,
    interleavedComplex := true,
    # standard codegen
    verify := true,
    globalUnrolling := 10000,
    includeMath := true,
));

# preset for looped DFT code
Class(loopDFT, rec(
    # scalar algorithms
    PD := true,
    PFA := true,
    RealRader := true,
    Rader := true,
    PRDFT:=false,
    URDFT:= true,
    CT := true,
    minCost := true,
    splitRadix := false,
    # basic vector configuration
    stdTTensor := false,
    pushTag := true,
    flipIxA := true,
    useConj := false,
    interleavedComplex := true,
    # standard codegen
    verify := true,
    includeMath := true,
));

Class(baseUnroll, rec(
    PFA_maxSize := 16,
    Rader_maxSize := 16,
    Mincost_maxSize := 16,
    URDFT_maxRadix := 16,
    globalUnrolling := 32,
));


# vectorization methods: real/complex
Class(cxVect, rec(realVect := false, cplxVect := true));
Class(realVect, rec(realVect := true, cplxVect := false));

# vectorization: split L vs. SVCT
Class(vecmem, rec(svct := true, splitL := false));
Class(stride, rec(svct := false, splitL := true));

# zero padding
Class(noPadd, rec(oddSizes := false));
Class(padd, rec(oddSizes := true));

# top-level algorithms
Class(F2, rec(tsplVBase := true, tsplCxVBase := true));
Class(CT, rec(tsplCT := true, tsplPFA := false, tsplRader := false, tsplBluestein := false, tsplVBase := false, tsplCxVBase := false));
Class(PFA, rec(tsplCT := false, tsplPFA := true, tsplRader := false, tsplBluestein := false, tsplVBase := false, tsplCxVBase := false));
Class(Rader_CT, rec(tsplCT := true, tsplPFA := false, tsplRader := true, raderAvoidSizes := [ ], tsplBluestein := false, tsplVBase := false, tsplCxVBase := false));
Class(Rader_PFA, rec(tsplCT := false, tsplPFA := true, tsplRader := true, raderAvoidSizes := [ ], tsplBluestein := false, tsplVBase := false, tsplCxVBase := false));
Class(Bluestein_CT, rec(tsplCT := true, tsplPFA := false, tsplRader := false, tsplBluestein := true,
    bluesteinMinPrime := 997, bluesteinExtraSizes := [ 2..100 ], tsplVBase := false, tsplCxVBase := false));

# all algorithms turned on -- unrolled code only
Class(Small_DFT_RuleSet, rec( tsplCT := true, tsplPFA := true, tsplRader := true, raderAvoidSizes := [ 47, 59, 107 ], tsplBluestein := true, bluesteinMinPrime := 997,
    bluesteinExtraSizes := [ 23, 46, 47, 49, 59, 67, 79, 83, 94, 103, 106, 107, 115 ], tsplVBase := true, tsplCxVBase := true));

SIMDGlobals.experiments := rec(
    vects := [realVect, cxVect],
    vmodes := [vecmem],
    experiments := [
        [[unrollDFT, noPadd, CT], [DFT_tSPL_CT], "SVCT"], # original SVCT
        [[unrollDFT, padd, CT], [DFT_tSPL_CT], "paddSVCT"], # zeropadded SVCT [ICASSP:07]
        [[unrollDFT, padd, PFA], [DFT_tSPL_GoodThomas], "PFA"], # PFA
        [[unrollDFT, padd, F2], [DFT_tSPL_VBase, DFT_tSPL_CxVBase], "baseF2"], # Base case
        [[unrollDFT, noPadd, Rader_CT], [DFT_tSPL_Rader],"Rader+SVCT"], # Rader + SVCT == best Rader sizes
        [[unrollDFT, padd, Rader_CT], [DFT_tSPL_Rader], "Rader+paddSVCT"], # Rader with CT
        [[unrollDFT, padd, Rader_PFA], [DFT_tSPL_Rader], "Rader+PFA"], # Rader with PFA
        [[unrollDFT, noPadd, Bluestein_CT], [DFT_tSPL_Bluestein], "Bluestein"], # Bluestein for all sizes
    ],
    benches := rec(
        unrolled := [[[unrollDFT, padd, Small_DFT_RuleSet], [DFT_tSPL_CT, DFT_tSPL_GoodThomas,
            DFT_tSPL_VBase, DFT_tSPL_CxVBase, DFT_tSPL_Rader, DFT_tSPL_Bluestein], "smallDFT"]],
        looped := [[[loopDFT, baseUnroll, noPadd, CT], [DFT_tSPL_CT], "loopSVCT"]]
    )
);
