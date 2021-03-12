
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#Concatenation(Conf("spiral_dir"), Conf("path_sep"), "libgen", Conf("path_sep"), 

#F InitLibgen(<opts>)
#F  opts must have a .libgen field.
#F  Example: opts := InitLibgen(LibgenDefaults);
#F
InitLibgen := function(opts) 
    local base_opts, bench;
    opts := Copy(opts);

    base_opts := CopyFields(opts, rec(
            globalUnrolling := opts.libgen.basesUnrolling,
            breakdownRules  := opts.libgen.basesBreakdowns, 
            hashFile        := opts.libgen.basesHashFile,
            benchTransforms := opts.libgen.bases));

    bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)), 
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
    bench.resumeAll();

    opts.libgen.baseBench := bench;
    opts.baseHashes := Concatenation(opts.baseHashes, [ CreateRecursBaseHash(bench.exp.bases.hashTable) ]);
    return opts;
end;

_bases := function(max)
   local t, facs, sizesTwoPower, sizesGeneral;
   sizesTwoPower := List([1..Log2Int(max)], x->2^x);
   facs := [2, 3, 5, 7, 11];
   sizesGeneral := Filtered([2 .. max], i -> ForAll(FactorsInt(i), f -> f in (facs)));
   t := []; 
   Append(t, List(sizesTwoPower, PRDFT3));
   Append(t, List(sizesTwoPower, n -> PRDFT3(n, -1).transpose()));
   Append(t, List(sizesTwoPower, n -> DFT(n, -1 mod n)));
#  Append(t, List(sizesTwoPower, URDFT));
   Append(t, List(sizesGeneral, PRDFT));
   Append(t, List(sizesGeneral, DFT));
   return t;
end;

LibgenDefaultsMixin := rec(
    codegen := RecCodegen,
    verbosity := 1,
    hashTable := HashTableDP(),

    formulaStrategies := rec(
	sigmaSpl    := [ StandardSumsRules, HfuncSumsRules ],
	preRC       := [],      
	rc          := [ StandardSumsRules, HfuncSumsRules ],
        postProcess := [
	    (s, opts) -> BlockSums(opts.globalUnrolling, s),
	    (s, opts) -> Process_fPrecompute(s, opts),
            RecursStepTerm            
        ]   
    ),

    libgen := rec(
	codeletTab := CreateCodeletHashTable(),
	terminateStrategy := [ HfuncSumsRules ], # used in SumsCodelet

	bases := _bases(34),
	basesUnrolling := 2^16, 
        basesHashFile := let(p:=Conf("path_sep"), 
            Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "realbases.hash")),
	basesBreakdowns := rec(
	    DFT    := [DFT_Base, DFT_CT, DFT_GoodThomas, DFT_PD, DFT_Rader],
	    PRDFT  := [PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_Rader, PRDFT_PD],
	    PRDFT3 := [PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1])
    )
);

LibgenDefaults := CopyFields(SpiralDefaults, LibgenDefaultsMixin, rec(
    globalUnrolling := 66,
    breakdownRules := rec(
	DFT    := [DFT_Base, DFT_CT],
	PRDFT  := [PRDFT1_Base2, PRDFT1_CT],
	PRDFT3 := [PRDFT3_Base2, PRDFT3_CT],
	PDCT4  := [PDCT4_Base2, PDCT4_CT],
	PDST4  := [PDST4_Base2, PDST4_CT],
	InfoNt := [Info_Base]),
    libgen := Copy(LibgenDefaultsMixin.libgen)
));

CplxLibgenDefaults := CopyFields(CplxSpiralDefaults, LibgenDefaultsMixin, rec(
    globalUnrolling := 34, 
    breakdownRules := rec(
	DFT    := [DFT_Base, DFT_CT],
	PRDFT  := [PRDFT1_Base2, PRDFT1_CT],
	PRDFT3 := [PRDFT3_Base2, PRDFT3_CT],
	InfoNt := [Info_Base]),
    libgen := CopyFields(Copy(LibgenDefaultsMixin.libgen), rec(
	bases := List([1..5], x->DFT(2^x)), 
        basesHashFile := let(p:=Conf("path_sep"), 
            Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "cplxbases.hash")),
	basesBreakdowns := rec(
	    DFT    := [DFT_Base, DFT_PRDFT],
	    PRDFT  := [PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_PD, PRDFT_Rader],
	    PRDFT3 := [PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1])
    ))
));
