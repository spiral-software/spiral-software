
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(SMP_Unparser,      SMP_UnparseMixin, CUnparserProg);
Class(SMP_MacroUnparser, SMP_UnparseMixin, CMacroUnparserProg);

Class(OpenMP_Unparser,      OpenMP_UnparseMixin, CUnparserProg);
Class(OpenMP_MacroUnparser, OpenMP_UnparseMixin, CMacroUnparserProg);

# suggested values: bufIters=64 (16 for older machines), maxRank=1 (larger value increases search space)
# Example: opts := InitGTLibgen(64, 1)
#
InitGTLibgen := function(bufIters, maxRank, useComplex)
    local opts;
    LibgenHardcodeStrides();

    opts := CopyFields(InitLibgen(When(useComplex, CplxLibgenDefaults, LibgenDefaults)),  
        rec(
            useDeref := true,
            breakdownRules := rec(
                GT  := [ CopyFields(GT_Base, rec(maxSize := 32)),
                         CopyFields(GT_BufReshape, rec(bufIters := bufIters)),
                         CopyFields(GT_DFT_CT, rec(minSize := 33, maxRank := maxRank)),
                         GT_NthLoop, GT_Par ],
                DFT := [ CopyFields(DFT_CT, rec(maxSize:=32)),
                         CopyFields(DFT_GT_CT, rec(minSize:=32)),
                         DFT_Base ],
                InfoNt := [Info_Base])
        ));
    opts.formulaStrategies.preRC := [ HfuncSumsRules ];
    return opts;
end;

InitSMPGTLibgen := function(bufIters, maxRank, useComplex, useOpenMP)
    local opts;
    opts := CopyFields(InitGTLibgen(bufIters, maxRank, useComplex), rec(
            unparser := Cond(
                useComplex and useOpenMP,         OpenMP_MacroUnparser,
                useComplex and not useOpenMP,        SMP_MacroUnparser,
                not useComplex and useOpenMP,     OpenMP_Unparser,
                not useComplex and not useOpenMP,    SMP_Unparser)));

    opts.formulaStrategies.sigmaSpl := [ MergedRuleSet(StandardSumsRules,RulesSMP), HfuncSumsRules ];
    opts.formulaStrategies.rc := opts.formulaStrategies.sigmaSpl;

    if not useOpenMP then
        opts.subParams := [var("num_threads", TInt), var("tid", TInt)];
        opts.profile := When(LocalConfig.osinfo.isWindows(),
            LocalConfig.cpuinfo.profile.threads(),
            profiler.default_profiles.linux_x86_threads
        );        
    fi;
    return opts;
end;


