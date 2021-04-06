
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(compiler, code, fpgen);

# temporary fix to declare variables used in _mm_loadl_xx/_mm_loadh_xx
CellCompileStrategyVector := Concatenation(
#    [ c -> vref.setNoScalar(c) ], # must be at the begin; otherwise subvector access breaks
    BaseCS,
    [
    BinSplit, CSE,
    # MarkDefUse, FFTWScheduleAssignments, CopyPropagate, <- NOTE: breaks
#    MarkDefUse, DFSChain, # currently not used
    CopyPropagate,
    Compile.declareVars,
    (c,opts) -> opts.vector.isa.fixProblems(c, opts),
    (c, opts) -> ESReduce(c, opts)
#    c -> DeadCodeElim(c),
#    c -> vref.resetNoScalar(c) # must be at the end right before declaring the missing vars!!
    #DeclareVars
]);

CellCompileStrategyVectorFP := (bits, fracbits) -> Concatenation(
    CellCompileStrategyVector,
    [ c -> FixedPointCode(c, bits, fracbits) ]
);
