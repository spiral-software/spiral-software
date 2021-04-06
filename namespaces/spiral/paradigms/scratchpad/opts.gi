
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(ScratchpadGlobals, rec(
    getOpts := meth(arg)
        local lssize, opts, swp, brules, nrules, br, nrsgmts, vlen, globalUnrolling,size, ttype;

	lssize := When (Length(arg) >= 2, arg[2], 2);
	nrsgmts := When (Length(arg) >= 3, arg[3], 1);
	vlen := When (Length(arg) >= 4, arg[4], 1);
	size := When (Length(arg) >= 5, arg[5], 2);
    ttype := When (Length(arg) >= 6, arg[6], 'R');
	swp := When (Length(arg) >= 7, arg[7], false);
	globalUnrolling := When(Length(arg) >=8, arg[8], 1);

    brules := When(IsRec(SpiralDefaults.breakdownRules),
        UserRecFields(SpiralDefaults.breakdownRules),
        Filtered(Dir(SpiralDefaults.breakdownRules), i->not i in SystemRecFields));
    nrules := rec();
    for br in brules do
        nrules.(br) := List(SpiralDefaults.breakdownRules.(br), i->CopyFields(i));
    od;
    opts := CopyFields(SpiralDefaults);
    opts.breakdownRules := nrules;

    opts.breakdownRules.TCompose := [ TCompose_tag];
    opts.breakdownRules.DFT := [DFT_Base, DFT_CT, DFT_tSPL_CT, DFT_PD, DFT_Rader];
    opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1];
 
	opts.breakdownRules.WHT := [WHT_tSPL_BinSplit, WHT_Base, WHT_BinSplit];
   
	opts.breakdownRules.TTensor := [AxI_IxB,IxB_AxI];
	opts.breakdownRules.TTensorI := Concat([IxA_scratch_push, IxA_base, AxI_base, IxA_L_base, L_IxA_base], [IxA_scratch, AxI_scratch, IxAL_scratch]);
	
	opts.tags := [Cond( ttype = 'R', ALStore(lssize,nrsgmts,vlen), ALStoreCx(lssize,nrsgmts,vlen)) ];

	opts.formulaStrategies.sigmaSpl := [ MergedRuleSet(RulesSumsScratch, RulesFuncSimpScratch, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRCScratch,RulesII,OLRules) ];
    opts.formulaStrategies.preRC := [ MergedRuleSet(RulesSumsScratch, RulesFuncSimpScratch, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRCScratch, RulesII, OLRules), (s,o) -> ScratchModel.updateInfo(s) ];
	opts.formulaStrategies.rc := [ MergedRuleSet(RulesSums, RulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRCScratch, RulesII, OLRules) ];
    #opts.formulaStrategies.postProcess := [(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s)];
    opts.size := size;
	opts.swp := swp;
    opts.globalUnrolling := globalUnrolling;

    opts.sumsgen := ScratchSumsGen;
	opts.codegen := ScratchCodegen;
    opts.unparser := CScratchUnparserProg;

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
    opts.model := ScratchModel;

    return opts;
    end
));

