
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(DRAMGlobals, rec(
    getOpts := meth(arg)
        local opts, brules, nrules, br, size, k, rb, lm, n, dram_datawidth, precision, dram_addrwidth,bb,throttle;
	
	rb := When (Length(arg) >= 2, arg[2], 2);
	#k := When (Length(arg) >= 2, arg[2], 2);
	lm := When (Length(arg) >= 3, arg[3], 8);
	n := When (Length(arg) >= 4, arg[4], 4);
	dram_datawidth := When (Length(arg) >= 5, arg[5], 256);
	precision := When (Length(arg) >= 6, arg[6], 64);
	dram_addrwidth := When (Length(arg) >= 7, arg[7], 27);
	bb := When (Length(arg) >= 8, arg[8], 1);
	throttle := When (Length(arg) >= 9, arg[9], 1);
	
	PrintLine("rb=",rb,", lm=",lm,", n=",n,", dram_datawidth=",dram_datawidth,", precision=",precision,", dram_addrwidth=",dram_addrwidth,", bb(BlackBoxMems)=",bb," throttle=",throttle);
	#PrintLine("k=",k,", lm=",lm,", n=",n,", dram_datawidth=",dram_datawidth,", precision=",precision,", dram_addrwidth=",dram_addrwidth,", bb(BlackBoxMems)=",bb);
	
	# Move this warning to breakdown rules
	# if(k*n > lm) then
	# 	PrintLine("Warning: Problem specification requires more than 2 stages!");
	# 	PrintLine("Warning: I can't handle it for now, sorry.");
	# 	lm := k*n;
	# 	PrintLine("Local memory size (lm) is assumed to be ",lm,"!");
	# 	PrintLine("..........");
	# fi;
	
    brules := When(IsRec(SpiralDefaults.breakdownRules),
        UserRecFields(SpiralDefaults.breakdownRules),
        Filtered(Dir(SpiralDefaults.breakdownRules), i->not i in SystemRecFields));
    nrules := rec();
    for br in brules do
        nrules.(br) := List(SpiralDefaults.breakdownRules.(br), i->CopyFields(i));
    od;
    opts := CopyFields(SpiralDefaults);
    opts.breakdownRules := nrules;

    #opts.breakdownRules.TCompose := [TCompose_tag];
    opts.breakdownRules.DFT := [DFT_Base, DFT_tSPL_CT_tiled, DFT_tSPL_push_tiled];
    #opts.breakdownRules.DFT := [];
    
	#opts.breakdownRules.MDDFT := [MDDFT_Base, MDDFT_tSPL_RowCol_break, MDDFT_tSPL_RowCol_push];
    opts.breakdownRules.MDDFT := [MDDFT_Base, MDDFT_tSPL_RowCol_break_2D, MDDFT_tSPL_RowCol_break_3D, MDDFT_tSPL_RowCol_push];
	
	opts.breakdownRules.TTwiddle := [ TTwiddle_dram ];
 
	#opts.breakdownRules.WHT := [WHT_tSPL_BinSplit, WHT_Base, WHT_BinSplit];
   
	# full applicable set of rules, should be used for all the algorithms
	opts.breakdownRules.TTensor := [AxI_IxB, IxB_AxI, L_BxI__L_AxI, AxI_L__BxI_L];
	#opts.breakdownRules.TTensor := [AxI_L__BxI_L];
	# restricted to only AxI_L__BxI_L since verification scripts assume that the 
	# generated hardware is based this breakdown
	
	
	#opts.breakdownRules.TTensorI := Concat([IxA_scratch_push, IxA_base, AxI_base, IxA_L_base, L_IxA_base], [IxA_scratch, AxI_scratch, IxAL_scratch]);
	
	opts.breakdownRules.TTensorI := [IxA_tile_base, AxI_tile_base, L_IxA_tile_base, IxA_L_tile_base, IxA_tile, AxI_tile, L_IxA_tile, IxA_L_tile];
	opts.breakdownRules.TL := [L_base, I_tileRd, I_tileWr, L_tileRd, L_tileWr, I_cubeRd, I_cubeWr, Ln3n2_cubeRd, Ln3n_cubeWr, 
								InLn2n_cubeRd, InLn2n_cubeWr];
	#opts.breakdownRules.TCompose := [AB_tile];
	
	opts.throttle := throttle;

	opts.tags := [ADram(rb,lm)];
	#opts.tags := [ATile(k,lm)];


#	opts.formulaStrategies.sigmaSpl := [ MergedRuleSet(RulesSumsScratch, RulesFuncSimpScratch, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRCScratch,RulesII,OLRules) ];
#    opts.formulaStrategies.preRC := [ MergedRuleSet(RulesSumsScratch, RulesFuncSimpScratch, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRCScratch, RulesII, OLRules), (s,o) -> ScratchModel.updateInfo(s) ];
#	opts.formulaStrategies.rc := [ MergedRuleSet(RulesSums, RulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRCScratch, RulesII, OLRules) ];
    #opts.formulaStrategies.postProcess := [(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s)];
    opts.size := [n,n];
	#opts.swp := swp;
    #opts.globalUnrolling := globalUnrolling;

	opts.dram_datawidth := dram_datawidth;
	opts.precision := precision;
	opts.dram_addrwidth := dram_addrwidth;
	opts.bb := bb;
	
#    opts.sumsgen := ScratchSumsGen;
#	opts.codegen := ScratchCodegen;
#    opts.unparser := CScratchUnparserProg;

    opts.memModifier := "__memory";
    opts.scratchModifier := "__scratch";
    opts.arrayDataModifier := "__rom";
    opts.romModifier := "__rom";
	
	opts.includes := [];
    Add(opts.includes, "\"scratch.h\"");

    opts.dmaSignal := (self, opts) >> "DMA_signal";
    opts.dmaWait := (self, opts) >> "DMA_wait";
    opts.cpuSignal := (self, opts) >> "CPU_signal";
    opts.cpuWait := (self, opts) >> "CPU_wait";
    opts.dmaFence := (self, opts) >> "DMA_fence";
    opts.dmaLoad := (self, opts) >> "DMA_load";
    opts.dmaStore := (self, opts) >> "DMA_store";
#    opts.model := ScratchModel;

    return opts;
    end
));

