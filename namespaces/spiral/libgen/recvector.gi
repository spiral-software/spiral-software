
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(paradigms.vector.rewrite);
Import(paradigms.smp);
Import(paradigms.distributed);
Import(paradigms.multibuffer);

StandardVecRules := MergedRuleSet(RulesSplitComplex, StandardSumsRules, RulesSMP, RulesVec, RulesPropagate, RulesVDiag, RulesKickout, RulesTermGrp, TerminateSymSPL);
HfuncVecRules    := MergedRuleSet(StandardVecRules, RulesHfunc);

PartialVecTermRules := MergedRuleSet(RulesVRC, RulesRC, RulesTermGrp, RulesVRCTermDiag);
FullVecTermRules    := MergedRuleSet(RulesVRC, RulesRC, RulesTermGrp, RulesVRCTerm, RulesTerm, TerminateSymSPL);


VecLibStrategiesCell := rec(
   sigmaSpl := [
        RemoveNoPull_Dist,
        StandardSumsRules
	],

    postProcess := [
		MergedRuleSet(PTensorRules, StandardSumsRules, RulesTermGrp, RemoveBuf),
		MergedRuleSet(PTensorConvertRules, StandardSumsRules),
		StandardSumsRules,
		StandardVecRules,
		MergedRuleSet(HfuncVecRules, PartialVecTermRules),
		MergedRuleSet(HfuncVecRules, PartialVecTermRules),
		RecursStepTerm,
		MergedRuleSet(StandardVecRules, FullVecTermRules),
		MergedRuleSet(RulesHfunc, RulesFuncSimp, RulesStrengthReduce),
		CellVRCTerm,
		RemoveBuf,
		DistMultiBuf,
		RulesComposeDists,
		(s, opts) -> applyCellRules(s, opts),
		MultiBufDist, #NOTE: Need to do this after RC/VRC has been dealt with, (but before marking BBs). Why not move this to where DistMBuf is?
		FixBorder,    #Border is used by MultiBufDist, and hence cannot precede it, since FixBorder removes the border
		(s, opts) -> BlockSums(opts.globalUnrolling, s),
		(s, opts) -> applyCellInplace(s, opts),
		RulesComposeStreams
    ],

    rc := [],
    preRC := [],
);

VecLibStrategies := rec(
    sigmaSpl := [ StandardSumsRules ],

    postProcess := [
    StandardSumsRules,
    StandardVecRules,
        MergedRuleSet(HfuncVecRules, PartialVecTermRules),
        MultiBufDist,
        MergedRuleSet(HfuncVecRules, PartialVecTermRules),
        (s, opts) -> BlockSums(opts.globalUnrolling, s),
        (s, opts) -> Process_fPrecompute(s, opts), # Doing this in the codegen so we can verify
        RecursStepTerm,
        MergedRuleSet(StandardVecRules, FullVecTermRules),
        MergedRuleSet(RulesHfunc, RulesFuncSimp, RulesStrengthReduce),
        CellVRCTerm,
        (s, opts) -> applyCellRules(s, opts),
        CellDFTBlockCyclicLayoutHack,
        (s, opts) -> applyCellInplace(s, opts),
        RulesComposeDists
    ],

    rc := [],
    preRC := [],
);

VecLibTerminate := [
    MergedRuleSet(StandardVecRules, FullVecTermRules),
    MergedRuleSet(RulesHfunc, RulesFuncSimp, RulesStrengthReduce),
];


Class(VecRecCodegen, RecCodegen, VectorCodegen, rec(


));

Class(OpenMP_SSEUnparser, paradigms.smp.OpenMP_UnparseMixin, platforms.sse.SSEUnparser);
Class(OpenMP_SSEUnparser_ParFor, paradigms.smp.OpenMP_UnparseMixin_ParFor, platforms.sse.SSEUnparser);
Class(SMP_SSEUnparser,    paradigms.smp.SMP_UnparseMixin,    platforms.sse.SSEUnparser);
Class(SMP_NEONUnparser,    paradigms.smp.SMP_UnparseMixin,    platforms.neon.NEONUnparser);

Class(OpenMP_AVXUnparser, paradigms.smp.OpenMP_UnparseMixin, platforms.avx.AVXUnparser);
Class(OpenMP_AVXUnparser_ParFor, paradigms.smp.OpenMP_UnparseMixin_ParFor, platforms.avx.AVXUnparser);

SMP_NEONUnparser.preprocess := x -> FixAssign0(x);
SMP_SSEUnparser.preprocess := x -> FixAssign0(x);
OpenMP_SSEUnparser.preprocess := x -> FixAssign0(x);

Declare(_InitVecLibgen);
Declare(_InitVecParLibgenNEON);
Declare(_InitVecLibgenNEON);

InitVecLibgenCell := (isa, use_functions, use_openmp, use_buffering, simdopts) ->
    _InitVecLibgen(
        InitLibgen(CopyFields(LibgenDefaults, SIMDGlobals.getOpts(isa, simdopts),
        rec(generateComplexCode:=false))),
        use_functions, use_openmp, use_buffering, false);

InitVecLibgenNEON := (opt) ->
    _InitVecLibgenNEON(opt);

InitVecParLibgenNEON := (opt, use_functions, use_openmp, use_buffering) ->
    _InitVecParLibgenNEON(
			InitLibgen(CopyFields(LibgenDefaults, _InitVecLibgenNEON(opt), rec(generateComplexCode:=false))),
        use_functions, use_openmp, use_buffering, false);

InitVecParLibgenNEONCx := (opt, use_functions, use_openmp, use_buffering) ->
    _InitVecParLibgenNEON(
			InitLibgen(CopyFields(CplxLibgenDefaults, _InitVecLibgenNEON(opt), rec(generateComplexCode:=true))),
        use_functions, use_openmp, use_buffering, false);

InitVecLibgen := (isa, use_functions, use_openmp, use_buffering) ->
    _InitVecLibgen(
        InitLibgen(CopyFields(LibgenDefaults, SIMDGlobals.getOpts(isa,
                     rec(svct:=true, splitL:=true, oddSizes:=false)), rec(generateComplexCode:=false))),
        use_functions, use_openmp, use_buffering, false);

InitParLibgen := (use_functions, use_openmp, use_buffering) ->
    _InitVecLibgen(InitLibgen(LibgenDefaults), use_functions, use_openmp, use_buffering, false);

InitParLibgenCplx := (use_functions, use_openmp, use_buffering, use_inplace) -> CopyFields(
    _InitVecLibgen(InitLibgen(CplxLibgenDefaults), use_functions, use_openmp, use_buffering, use_inplace),
    rec(unparser := CMacroUnparserProg));

_InitVecLibgenNEON := function(opt)
	opt.globalUnrolling := 64;
	opt.breakdownRules.TL := [L_cx_real, SIMD_ISA_Bases1, SIMD_ISA_Bases2, IxLxI_kmn_n, IxLxI_kmn_km, L_mn_m_vec, IxLxI_vtensor];
	return opt;
end;

_InitVecParLibgenNEON := function(opt, use_functions, use_openmp, use_buffering, use_inplace)
    local opts, m, clet_size;
	opt.globalUnrolling := 64;
	opt.breakdownRules.TL := [L_cx_real, SIMD_ISA_Bases1, SIMD_ISA_Bases2, IxLxI_kmn_n, IxLxI_kmn_km, L_mn_m_vec, IxLxI_vtensor];
    m := When(not use_functions, 64, 32);
    clet_size := m;

    opts := CopyFields(opt, rec(
        breakdownRules := CopyFields(opt.breakdownRules, rec(
            GT := [
                CopyFields(GT_Base, rec(maxSize:=false)),
                GT_NthLoop,
                CopyFields(GT_DFT_CT, rec(
                        minRank := 1,
                        maxRank := When(use_buffering, 1, 0),
                        minSize := When(use_functions, m+1, 1),
                        forTransposition := false,   # What is the right value? beta.anl needs false.
                        codeletSize := When(use_inplace, clet_size, false),
                        inplace := use_inplace)),
                CopyFields(GT_Par, rec(parEntireLoop := false, splitLoop := true)),
                GT_Vec_AxI,
                GT_Vec_IxA, GT_Vec_IxA_L, GT_Vec_L_IxA,
                GT_Vec_SplitL ],

            DFT := [
                DFT_Rader, DFT_GoodThomas, DFT_PD, DFT_Base,
                CopyFields(DFT_CT, rec(maxSize := m)),
                CopyFields(DFT_GT_CT, rec(
                        codeletSize := When(use_inplace, clet_size, false),
                        inplace := use_inplace,
                        minSize := When(use_functions, m+1, 1)))
            ],
            DFT3     := [ DFT3_Base, DFT3_CT ],
            MDDFT    := [ MDDFT_Base, MDDFT_tSPL_RowCol ],
            # Below is used for MDDFT. MDDFT with inplaceness excludes DFT with inplaceness
            # Thus setting use_inplace=false will disable DFT inplaceness, and enable it in MDDFT
            # NOTE: above is ugly! the only way to fix it is to figure out storage schemes automatically..
            TTensor  := [ CopyFields(AxI_IxB,rec(inplace:=not use_inplace)),
                          CopyFields(IxB_AxI,rec(inplace:=not use_inplace)) ],
            TTensorI := [ TTensorI_toGT ],
            TCompose := [ TCompose_tag ],
            InfoNt   := [Info_Base]
        )),

        codegen := VecRecCodegen,
        libgen := CopyFields(opt.libgen, rec(terminateStrategy := VecLibTerminate)),
        compileStrategy := IndicesCS,
        useDeref := true
    ));

    if use_buffering then Add(opts.breakdownRules.GT,
            CopyFields(GT_BufReshape, rec(bufIters := [2,4,8,16], u := [2,4]))); fi;

    if not use_functions then
        opts.baseHashes := DropLast(opts.baseHashes, 1);
        Append(opts.formulaStrategies.postProcess, opts.libgen.terminateStrategy);
    fi;

    if use_openmp then
        opts.unparser := OpenMP_SSEUnparser;
    else
        # should not be needed in latest version
        opts.subParams := [var("num_threads", TInt), var("tid", TInt)];
        opts.unparser := SMP_NEONUnparser;
        opts.profile := When(LocalConfig.osinfo.isWindows(), # or LocalConfig.osinfo.isCygwin(),
            LocalConfig.cpuinfo.profile.threads(),
            profiler.default_profiles.linux_x86_threads
        );
    fi;

    return opts;
end;

_InitVecLibgen := function(opts, use_functions, use_openmp, use_buffering, use_inplace)
    local opts, m, clet_size;
    m := When(not use_functions, 64, 32);
    clet_size := m;

    opts := CopyFields(opts, rec(
        formulaStrategies := Copy(VecLibStrategies),
        breakdownRules := CopyFields(opts.breakdownRules, rec(
            GT := [
                CopyFields(GT_Base, rec(maxSize:=false)),
                GT_NthLoop,
                CopyFields(GT_DFT_CT, rec(
                        minRank := 1,
                        maxRank := When(use_buffering, 1, 0),
                        minSize := When(use_functions, m+1, 1),
                        forTransposition := false,   # What is the right value? beta.anl needs false.
                        codeletSize := When(use_inplace, clet_size, false),
                        inplace := use_inplace)),
                CopyFields(GT_Par, rec(parEntireLoop := false, splitLoop := true)),
                GT_Vec_AxI,
                GT_Vec_IxA, GT_Vec_IxA_L, GT_Vec_L_IxA,
                GT_Vec_SplitL ],
            DFT := [
                DFT_Rader, DFT_GoodThomas, DFT_PD, DFT_Base,
                CopyFields(DFT_CT, rec(maxSize := m)),
                CopyFields(DFT_GT_CT, rec(
                        codeletSize := When(use_inplace, clet_size, false),
                        inplace := use_inplace,
            # YSV: Please don't uncomment without talking to YSV
                        #requiredFirstTag := [AVecReg, AVecRegCx, AParSMP, ParCell],
                        minSize := When(use_functions, m+1, 1)))
            ],
            DFT3     := [ DFT3_Base, DFT3_CT ],
            MDDFT    := [ MDDFT_Base, MDDFT_tSPL_RowCol ],
            # Below is used for MDDFT. MDDFT with inplaceness excludes DFT with inplaceness
            # Thus setting use_inplace=false will disable DFT inplaceness, and enable it in MDDFT
            # NOTE: above is ugly! the only way to fix it is to figure out storage schemes automatically..
            TTensor  := [ CopyFields(AxI_IxB,rec(inplace:=not use_inplace)),
                          CopyFields(IxB_AxI,rec(inplace:=not use_inplace)) ],
            TTensorI := [ TTensorI_toGT ],
            TCompose := [ TCompose_tag ],
            InfoNt   := [Info_Base]
        )),

        codegen := VecRecCodegen,
        libgen := CopyFields(opts.libgen, rec(terminateStrategy := VecLibTerminate)),
        compileStrategy := IndicesCS,
        useDeref := true
    ));

    if use_buffering then Add(opts.breakdownRules.GT,
            CopyFields(GT_BufReshape, rec(bufIters := [2,4,8,16], u := [2,4]))); fi;

    if not use_functions then
        opts.baseHashes := DropLast(opts.baseHashes, 1);
        Append(opts.formulaStrategies.postProcess, opts.libgen.terminateStrategy);
    fi;

    if use_openmp then
        opts.unparser := OpenMP_SSEUnparser;
    else
        # should not be needed in latest version
        opts.subParams := [var("num_threads", TInt), var("tid", TInt)];
        opts.unparser := SMP_SSEUnparser;
        opts.profile := When(LocalConfig.osinfo.isWindows(), # or LocalConfig.osinfo.isCygwin(),
            LocalConfig.cpuinfo.profile.threads(),
            profiler.default_profiles.linux_x86_threads
        );
    fi;

    return opts;
end;


RecursStep.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
RecursStep.needInterleavedRight := self >> self.child(1).needInterleavedRight();

RTWrap.needInterleavedRight := self >> false;
RTWrap.needInterleavedLeft := self >> false;

VGath.mkCodelet    := self >> ObjId(self)(self.func.mkCodelet(), self.v);
VGath_sv.mkCodelet := self >> ObjId(self)(self.func.mkCodelet(), self.v, self.sv);
VScat.mkCodelet    := self >> ObjId(self)(self.func.mkCodelet(), self.v);
VScat_sv.mkCodelet := self >> ObjId(self)(self.func.mkCodelet(), self.v, self.sv);
VTensor.mkCodelet  := self >> ObjId(self)(self.child(1).mkCodelet(), self.vlen);

VRC.mkCodelet   := self >> ObjId(self)(self.child(1).mkCodelet(), self.v);
VRCL.mkCodelet  := self >> ObjId(self)(self.child(1).mkCodelet(), self.v);
VRCR.mkCodelet  := self >> ObjId(self)(self.child(1).mkCodelet(), self.v);
VRCLR.mkCodelet := self >> ObjId(self)(self.child(1).mkCodelet(), self.v);

BlockVPerm.mkCodelet := self >> self;
VPerm.mkCodelet      := self >> self;
VPrm_x_I.mkCodelet   := self >> self;

VDiag.mkCodelet     := self >> ObjId(self)(self.element.mkCodelet(), self.v);
VDiag_x_I.mkCodelet := self >> ObjId(self)(self.element.mkCodelet(), self.v);
VRCDiag.mkCodelet   := self >> ObjId(self)(self.element.mkCodelet(), self.v);

VData.signature     := self >> CodeletSignature(self.func);
VData.codeletParams := self >> CodeletParams(self.func);
VData.mkCodelet     := self >> ObjId(self)(MkCodelet(self.func), self.v);
VData.codeletShape  := self >> [ObjId(self), CodeletShape(self.func), self.v];

VDup.signature     := self >> CodeletSignature(self.func);
VDup.codeletParams := self >> CodeletParams(self.func);
VDup.mkCodelet     := self >> ObjId(self)(MkCodelet(self.func), self.v);
VDup.codeletShape  := self >> [ObjId(self), CodeletShape(self.func), self.v];

