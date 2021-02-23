
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


EmptyCS := [
    Compile.pullDataDeclsRefs,
    Compile.declareVars
];

BaseCS := [
    c -> Compile.pullDataDeclsRefs(c),
    c -> Compile.fastScalarize(c),
    c -> UnrollCode(c),
    c -> FlattenCode(c),
    c -> UntangleChain(c), 
    CopyPropagate,
    (c, opts) -> HashConsts(c, opts),
];

NoCSE := Concatenation(BaseCS, [
    Compile.declareVars
]);

NoSchedCSE_CS := Concatenation(BaseCS, [
    (c, opts) -> BinSplit(c, opts), 
    CSE,  MarkDefUse, CopyPropagate,
    Compile.declareVars
]);

SimpleCS := Concatenation(BaseCS, [
    (c, opts) -> BinSplit(c, opts),
    CSE, MarkDefUse, DFSChain, CopyPropagate,
    Compile.declareVars
]);


# IsCoarseType: checks if <coarse_t> is more general version of <fine_t> data type as defined by UnifyPair.
#        ex: TReal is a general version of T_Real(32) data type.
IsCoarseType := (coarse_t, fine_t) -> When( 
    coarse_t = fine_t, false, 
    Try(UnifyPair(fine_t, coarse_t)) = [true, fine_t]
);

# FixValueTypes: unifies value type with the type of surrounding expression.
#        SSE unparser needs this for figuring out actual data type of constants.
#        Fixed point backends rely on this to convert constants to fixed point...
#
FixValueTypes := c -> SubstTopDownRulesNR(c, rec( 
    fixValueTypes := Rule(
        @@(1, Value, (x, cx) -> 
            ObjId(Last(cx.parents)) in 
	        [add, sub, mul, bin_and, bin_xor, bin_or, absdiff, absdiff2, idiv, ddiv] and 
		IsCoarseType(x.t, Last(cx.parents).t) and not IsPtrT(Last(cx.parents).t)),
	(e, cx) -> Last(cx.parents).t.value(e.v)
    )
)); 


DerefNthCode := c -> SubstTopDownRules(c, rec(
    deref_nth := Rule(
	[nth, @(1).cond(e -> not(ObjId(e) in [Value, param])), @(2)], 
	e -> let(
	    b := @(1).val, idx := @(2).val, 
	    Cond(
		ObjId(idx) = add, deref(ApplyFunc(add, [b] :: idx.args)),
		ObjId(idx) = sub, deref(ApplyFunc(add, [b] :: [idx.args[1], neg(idx.args)])),
                                  deref(b + idx))))
));

NthDerefCode := c -> SubstTopDownRules(c, rec(
    deref_var := Rule([deref, @(1, var)], e -> nth(@(1).val, TInt.value(0))), 
    deref_add := Rule([deref, [add, @(1, var), @(2, Value)]], e -> nth(@(1).val, @(2).val))
));

BaseIndicesCS := [
    c -> Compile.pullDataDeclsRefs(c),
    c -> Compile.fastScalarize(c),
    c -> UnrollCode(c), 
    c -> FlattenCode(c), 
    c -> UntangleChain(c), 
    (c, opts) -> CopyPropagate.initial(c, opts), 
    (c, opts) -> HashConsts(c, opts), 
    c -> MarkDefUse(c), 
    (c, opts) -> BinSplit(c, opts), 
    c -> MarkDefUse(c),
    CopyPropagate, # does CSE
];

# Uses a fast (no strength reduction) final CopyPropagate pass, which
# kicks out vars used only once. Sometimes (MMM?) its not good (prevents hoisting)
# and then IndicesCS2 should be used.
#
IndicesCS0 := Concatenation(BaseIndicesCS, [
    c -> MarkDefUse(c), 
    # kicks out vars used only once or never
    (c, opts) -> CopyPropagate.fast(c, CopyFields(opts, rec(autoinline := true))), 
    (c, opts) -> Cond(opts.finalBinSplit, BinSplit(c, opts), c),
    (c, opts) -> Cond(IsBound(opts.scheduler), opts.scheduler(c, opts), c),
    c -> Compile.declareVars(c), 
]);

# Uses a full final CopyPropagate pass, which kicks out vars used only once
# and properly simplifies out redundant double butterfly structures, i.e. F(2)*F(2)
# 
IndicesCS := Concatenation(BaseIndicesCS, [
    c -> MarkDefUse(c), 
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
    c -> MarkDefUse(c), 
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
    (c, opts) -> Cond(opts.finalBinSplit, BinSplit(c, opts), c),
    (c, opts) -> Cond(IsBound(opts.scheduler), opts.scheduler(c, opts), c),
    c -> FixValueTypes(c),
    c -> Compile.declareVars(c)
]);

# IndicesCS + extra CopyPropagate pass to do DAG pruning
IndicesCS_Prune := Concatenation(BaseIndicesCS, [
    c -> MarkDefUse(c),
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
    c -> MarkDefUse(c),
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
    (c, opts) -> Cond(opts.finalBinSplit, BinSplit(c, opts), c),
    (c, opts) -> Cond(IsBound(opts.scheduler), opts.scheduler(c, opts), c),
    c -> FixValueTypes(c),
    c -> Compile.declareVars(c)
]);

# Does not use a final CopyPropagate pass, to keep variables used once, and 
# not prevent hoisting.
#
IndicesCS2 := Concatenation(BaseIndicesCS, [
    # - kicking out variables never used is fine.
    # - kicking out variables used once should be a GLOBAL pass because
    # - right now it prevents hoisting
    (c, opts) -> Cond(opts.finalBinSplit, BinSplit(c, opts), c),
    (c, opts) -> Cond(IsBound(opts.scheduler), opts.scheduler(c, opts), c),
    c -> FixValueTypes(c),
    c -> Compile.declareVars(c)
]);

IndicesCS_FMA := Concatenation(BaseIndicesCS, [
    DoFMA,
    MarkDefUse, #
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
    (c, opts) -> Cond(opts.finalBinSplit, BinSplit(c, opts), c),
    (c, opts) -> Cond(IsBound(opts.scheduler), opts.scheduler(c, opts), c),
    c -> FixValueTypes(c),
    c -> Compile.declareVars(c)
]);

IndicesCS_Fixed := (bitwidth, fracbits) -> (
    IndicesCS ::
    [ c -> FixedPointCode(c, bitwidth, fracbits) ]
);

IndicesCS_FixedNew := 
    IndicesCS ::
    [ (c, opts) -> FixedPointCode2(c) ];


IndicesCS2_FMA := Concatenation(BaseIndicesCS, [
    DoFMA,
    CopyPropagate,
    (c, opts) -> Cond(opts.finalBinSplit, BinSplit(c, opts), c),
    (c, opts) -> Cond(IsBound(opts.scheduler), opts.scheduler(c, opts), c),
    c -> FixValueTypes(c),
    c -> Compile.declareVars(c)
]);

IndicesCS_CXFMA := Concatenation(BaseIndicesCS, [
    DoCXFMA,
    MarkDefUse, #
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
    (c, opts) -> Cond(opts.finalBinSplit, BinSplit(c, opts), c),
    (c, opts) -> Cond(IsBound(opts.scheduler), opts.scheduler(c, opts), c),
    c -> FixValueTypes(c),
    c -> Compile.declareVars(c)
]);

IndicesCS2_CXFMA := Concatenation(BaseIndicesCS, [
    DoCXFMA,
    CopyPropagate,
    (c, opts) -> Cond(opts.finalBinSplit, BinSplit(c, opts), c),
    (c, opts) -> Cond(IsBound(opts.scheduler), opts.scheduler(c, opts), c),
    c -> FixValueTypes(c),
    c -> Compile.declareVars(c)
]);

# OLD STUFF

# RCSE_CS := Concatenation(BaseCS, [
#     (c, opts) -> BinSplit(c, opts), RCSE,
#     MarkDefUse, DFSChain, CopyPropagate,
#     Compile.declareVars
# ]);

# FFTW_CS := Concatenation(BaseCS, [
#     (c, opts) -> BinSplit(c, opts), RCSE,
#     MarkDefUse, FFTWScheduleAssignments, CopyPropagate,
#     Compile.declareVars
# ]);

#
# FMA
#
FMA_CS := Concatenation(BaseCS, [
    (c, opts) -> BinSplit(c, opts), (c, opts) -> BinSplit(c, opts), CSE,
    MarkDefUse, DFSChain, CopyPropagate,
    DoFMA,
    Compile.declareVars
]);

FMA_FFTW_CS := Concatenation(BaseCS, [
    (c, opts) -> BinSplit(c, opts), (c, opts) -> BinSplit(c, opts), CSE,
    MarkDefUse, FMA, ClearDefUse,
    MarkDefUse, FFTWScheduleAssignments, CopyPropagate,
    DoFMA,
    Compile.declareVars
]);

# # seems to be slower
# NewUnrollCS := [
#     myUnrollCode, CopyPropagate,
#     HashConstantsCode,
#     MarkDefUse, DFSChain, CopyPropagate,
#     Compile.declareVars
# ];

# CompileStrategyFull := [
#     Compile.pullDataDeclsRefs, 
#     UnrollCode, FlattenCode, SSA,  CopyPropagate,
#     FoldIf, SSA, CopyPropagate,   # Remove dead IF branches
# #    Compile.scalarize,
#     SSA, CopyPropagate,
#     EliminatePhiSSA, CopyPropagate, # Eliminate Phi functions
#     HashConstantsCode,
#     (c, opts) -> BinSplit(c, opts), CSE, CopyPropagate,  # CSE
#     MarkDefUse, #DFSChain,
#     Compile.declareVars
# ];

# This compile strategy is safe for IFs inside basic blocks. but does not
# fully perform copy propagation. Thus the name 'conservative'. To fully and
# safely optimize, we need to extend the CopyPropagate pass
#
conservativeCompileSSA := [
    Compile.pullDataDeclsRefs, # -- 1
    Compile.fastScalarize,     # -- 2
    UnrollCode,    # -- 3
    FlattenCode,   # -- 4
#    SimpIndicesCode, # -- 5   (YSV: simpIndices was disabled before, not clear why)
    FoldIf,        # -- 6   FoldIf happens before SSA and SSA has to happen before Copyprop
    SSA,           # -- 7
    UntangleChain, # -- 8
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(doScalarReplacement:=false))),  # -- 9
    (c, opts) -> HashConsts(c, opts),                                                  # -- 10
    (c, opts) -> When(IsBound(opts.useDeref) and opts.useDeref, DerefNthCode(c), c), # -- 11
    MarkPreds,
    (c, opts) -> BinSplit(c, opts),                                                        # -- 12, 13
    MarkDefUse,
    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(doScalarReplacement:=false))), # 14
    EliminatePhiSSA, 
    Compile.declareVars
];


CompileSSA := [
    Compile.pullDataDeclsRefs,
    Compile.fastScalarize, UnrollCode,
    FlattenCode,
    FoldIf,
    UntangleChain,
    CopyPropagate,
    SSA,
    (c, opts) -> HashConsts(c, opts), 
    (c, opts) -> When(IsBound(opts.useDeref) and opts.useDeref, DerefNthCode(c), c),
    MarkPreds,
    (c, opts) -> BinSplit(c, opts),
    ClearDefUse,
    MarkDefUse,
    CopyPropagate,
    EliminatePhiSSA,
    CopyPropagate,
    Compile.declareVars
];


