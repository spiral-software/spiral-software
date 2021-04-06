
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(TICompose, rec(
    TICompose_DropTag := rec(
        applicable := t -> true,
        children := t -> [[ t.params[3] ]],
        apply := (t, C, Nonterms) -> ICompose(t.params[1], t.params[2], C[1])
    )
));

#F InitStreamSw(<radix>)
#F
#F Example:
#F   opts := InitStreamSw(2);
#F   c := CodeRuleTree(opts.transforms.DFT(64), opts);
#F   me   := CMatrix(c, opts);
#F   them := MatSPL(opts.transforms.DFT(64));
#F   InfinityNormMat(me-them);
#F
InitStreamSw := radix -> CopyFields(
    CplxSpiralDefaults, 
    IntelC99Mixin, # uncomment this for C99 output (default = SSE2 intrinsics)
    rec(
    breakdownRules := rec(
        DFTDR :=  [CopyFields(DFTDR_tSPL_Pease, rec(precompute:=true)) ], 
        DFT   :=  [DFT_Base, DFT_CT ],
        TTensorI := [IxA_base, IxA_L_base, L_IxA_base, AxI_base],
        TCompose := [TCompose_tag],
        TICompose := [TICompose_DropTag],
        TDiag    := [TDiag_base]
    ),

    transforms := rec(
        DFT := n -> DFTDR(n, radix, [AStream(radix)])
    ),

#    unparser := CMacroUnparserProg,

    globalUnrolling := radix
));

#F TestStreamSw(<n>, <radix>, <use_transpose>)  - software test of Pease DFT used for streaming hardware
#F
#F use_transpose = true:  DRDFT
#F use_transpose = false: DFTDR
#F
TestStreamSw := function(n, radix, use_transpose)
    local opts, t, rt, c, err, me, them;
    opts := InitStreamSw(radix);

    PrintErr("Compiling...\n");
    t := When(use_transpose, opts.transforms.DFT(n).transpose(), opts.transforms.DFT(n));
    rt := RandomRuleTree(t, opts);
    if rt = false 
        then Error("No ruletrees."); fi;
    c := CodeRuleTree(rt, opts);

    PrintErr("Running and computing the code matrix...\n");
    me := CMatrix(c, opts);

    PrintErr("Comparing to the definition...\n");
    them := MatSPL(t);
    err := InfinityNormMat(me-them);

    if err < 1e-6 then PrintErr(t, " -- ", GreenStr("OK\n")); 
    else PrintErr(t, " -- ", RedStr("FAIL\n")); fi;
end;

TestStreamSwPrint := function(n, radix, use_transpose)
    local opts, t, rt, c, err, me, them;
    opts := InitStreamSw(radix);

    t := When(use_transpose, opts.transforms.DFT(n).transpose(), opts.transforms.DFT(n));
    rt := RandomRuleTree(t, opts);
    if rt = false 
        then Error("No ruletrees."); fi;
    c := CodeRuleTree(rt, opts);
    
    return PrintCode("function_name", c, opts);
#    return rt;
end;