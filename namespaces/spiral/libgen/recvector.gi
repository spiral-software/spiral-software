
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

ErrorOut := function(sums, opts)
    PrintLine("------------- Begin ------------------");
    PrintLine(sums);
    PrintLine("------------- End ------------------");
    Error("BP: postprocess");
    return(sums);
end;

ErrorOut1 := function(sums, opts)
   Print(".");
   return(sums);
end;

VecLibStrategiesCell := rec(
#    sigmaSpl := [ StandardSumsRules, StandardVecRules ],
   sigmaSpl := [
               #(s, opts) -> ErrorOut(s, opts),
               RemoveNoPull_Dist,
               StandardSumsRules ],

    postProcess := [
    #(s, opts) -> ErrorOut(s, opts),
    MergedRuleSet(PTensorRules, StandardSumsRules, RulesTermGrp, RemoveBuf),
    #(s, opts) -> ErrorOut(s, opts),
    MergedRuleSet(PTensorConvertRules, StandardSumsRules),
    #(s, opts) -> ErrorOut(s, opts),
    StandardSumsRules,
    #(s, opts) -> ErrorOut(s, opts),
    StandardVecRules,
    #(s, opts) -> ErrorOut(s, opts),
        MergedRuleSet(HfuncVecRules, PartialVecTermRules),
    #(s, opts) -> ErrorOut(s, opts),
       MergedRuleSet(HfuncVecRules, PartialVecTermRules),
    #(s, opts) -> ErrorOut(s, opts),
       #(s, opts) -> BlockSums(opts.globalUnrolling, s), #Doing this later (is this a problem?)
       #(s, opts) -> Process_fPrecompute(s, opts), # Doing this in the codegen so we can verify
       RecursStepTerm,
    #(s, opts) -> ErrorOut(s, opts),
       MergedRuleSet(StandardVecRules, FullVecTermRules),
    #(s, opts) -> ErrorOut(s, opts),
       MergedRuleSet(RulesHfunc, RulesFuncSimp, RulesStrengthReduce),
    #(s, opts) -> ErrorOut(s, opts),
        CellVRCTerm,
        RemoveBuf,
      DistMultiBuf,
        #RulesSums, # So composes are flattened before RulesComposeDists runs
       RulesComposeDists,
        #RulesDistMerge, # Not required because for the DMP algorithm, we now do this at a high level
    #(s, opts) -> ErrorOut(s, opts),
      (s, opts) -> applyCellRules(s, opts),
    #(s, opts) -> ErrorOut(s, opts),
        MultiBufDist, #NOTE: Need to do this after RC/VRC has been dealt with, (but before marking BBs). Why not move this to where DistMBuf is?
    #(s, opts) -> ErrorOut(s, opts), #Uncomment line to see state after VRC rules apply.
#      DistMultiBuf, # Why is this here? Why not move it down?
    #(s, opts) -> ErrorOut(s, opts),
        FixBorder,    #Border is used by MultiBufDist, and hence cannot precede it, since FixBorder removes the border
    #(s, opts) -> ErrorOut(s, opts),
        (s, opts) -> BlockSums(opts.globalUnrolling, s),
    #(s, opts) -> ErrorOut(s, opts),
#       (s, opts) -> CellDFTBlockCyclicLayoutHackWrap(s, opts),   # This does nothing if there are no GathRecv/ScatSend that still have functions
    #(s, opts) -> ErrorOut(s, opts),
        (s, opts) -> applyCellInplace(s, opts),
        RulesComposeStreams
    ],

    rc := [],
    preRC := [],
);

VecLibStrategies := rec(
#    sigmaSpl := [ StandardSumsRules, StandardVecRules ],
    sigmaSpl := [ StandardSumsRules ],

    postProcess := [
    #MergedRuleSet(PTensorRules, StandardSumsRules),
    #MergedRuleSet(PTensorConvertRules, StandardSumsRules),
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

#
# NOTE: implement layering
#
#Class(VecRecCodegen, RecCodegen, VectorCodegen);
Class(VecRecCodegen, RecCodegen, VectorCodegen, rec(

#	vRC_Compose := Rule([vRC, @(1, Compose)], e -> Compose(List(@(1).val.children(), vRC))),
#	vRC_SUM := Rule([vRC, @(1, SUM)], e -> SUM(List(@(1).val.children(), vRC))),
#	vRC_SUMAcc := Rule([vRC, @(1, SUMAcc)], e -> SUMAcc(List(@(1).val.children(), vRC))),

#	vRC_Container := Rule([vRC, @(1, [BB,Buf,Inplace,Grp,NoPull,NoPullLeft,NoPullRight, NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight ])],
#		e -> ObjId(@(1).val)(vRC(@(1).val.child(1)))),

#	vRC_Data := Rule([vRC, @(1, Data)], e -> Data(@(1).val.var, @(1).val.value, vRC(@(1).val.child(1)))),

#	vRC_RStep := Rule([vRC, @(1, RecursStep)], e -> RecursStep(2*@(1).val.yofs, 2*@(1).val.xofs,vRC(@(1).val.child(1)))),

#	vRC_ISum := Rule([vRC, @(1, ISum)], e -> ISum(@(1).val.var, @(1).val.domain, vRC(@(1).val.child(1)))),
#	vRC_ICompose := Rule([vRC, @(1, ICompose)], e -> ICompose(@(1).val.var, @(1).val.domain, vRC(@(1).val.child(1)))),

#	vRC_Grp := Rule([vRC, @(1, Grp)], e -> Grp(vRC(@(1).val.child(1)))),


#	vRC_Scale := Rule([vRC, @(1, Scale)], e ->
#		vRC(Diag(fConst(Rows(@(1).val), @(1).val.scalar))) * vRC(@(1).val.child(1))),

#	vRC_CR := Rule([vRC, @(1, CR)], e -> @(1).val.child(1)),

#	VTensor_VScale := Rule([@(1, VTensor), [VScale, @(2), @(3), @(4)]], e -> VScale(VTensor(@(2).val, @(1).val.vlen), @(3).val, @(4).val*@(1).val.vlen)),
#	VTensor_VTensor := Rule([@(1, VTensor), @(2, VTensor)], e -> VTensor(@(2).val.child(1), @(1).val.vlen*@(2).val.vlen)),


#	vRC_SMP := Rule([@(1, vRC), @(2, [SMPSum, SMPBarrier])],
#		e -> let(s := @(2).val, CopyFields(s, rec(_children := List(s.children(), c->ObjId(@(1).val)(c)), dimensions := @(1).val.dimensions)))),

# should RC and VRC be in some other place?
#	vRC_TCvt := Rule( [@(0, [vRC, RC, VRC]), @(1, TCvt)],
#		e -> let( t := @(1).val, TCvt( 2*t.n(), t.isa_to(), t.isa_from(), t.props()).withTags(t.getTags()).takeAobj(t) )),

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
#	local opts;
#	opts := platforms.neon.benchNEON().half.1d.dft_ic.small.cmplx().getOpts();
#	opts.profile := default_profiles.linux_arm;
	opt.globalUnrolling := 64;
	opt.breakdownRules.TL := [L_cx_real, SIMD_ISA_Bases1, SIMD_ISA_Bases2, IxLxI_kmn_n, IxLxI_kmn_km, L_mn_m_vec, IxLxI_vtensor];
	return opt;
end;

_InitVecParLibgenNEON := function(opt, use_functions, use_openmp, use_buffering, use_inplace)
    local opts, m, clet_size;
#		opt := platforms.neon.benchNEON().half.1d.dft_ic.small.cmplx().getOpts();
		#	opts.profile := default_profiles.linux_arm_pthread;
		opt.globalUnrolling := 64;
		opt.breakdownRules.TL := [L_cx_real, SIMD_ISA_Bases1, SIMD_ISA_Bases2, IxLxI_kmn_n, IxLxI_kmn_km, L_mn_m_vec, IxLxI_vtensor];
    m := When(not use_functions, 64, 32);
    clet_size := m;

    opts := CopyFields(opt, rec(
#        formulaStrategies := Copy(VecLibStrategies),
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
#						DFT := opts.breakdownRules.DFT :: [  
#                CopyFields(DFT_GT_CT, rec(
#                        codeletSize := When(use_inplace, clet_size, false),
#                        inplace := use_inplace,
#                        minSize := When(use_functions, m+1, 1)))
#						],
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
#D            TTag     := [TTag_down],
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
#D            TTag     := [TTag_down],
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

# Example: doParSimdDft(1, 8, false, false, false); # no threads
# Example: doParSimdDft(4, 8, false, false, false); # 4 threads, SPMD
# Example: doParSimdDft(4, 8, false, true, false);  # 4 threads, OpenMP
# Example: doParSimdDft(4, 8, true, true, false);   # 4 threads, OpenMP, codelet reuse
#
doParSimdDft := function(arg)
    local sizes, opts, dpbench, tags, name, isa, p, logn, use_functions, use_openmp,
        use_buffering, interleavedComplex, argrec,simd_opts;

    isa := arg[1];
    p := arg[2];
    logn := arg[3];
    if Length(arg) = 4 and IsRec(arg[4]) then
        argrec := CopyFields(rec(
            use_functions := false,
            use_openmp := true,
            use_buffering := false,
            interleavedComplex := true,
            simd_opts := rec()
        ), arg[4]);
        use_functions := argrec.use_functions;
        use_openmp := argrec.use_openmp;
        use_buffering := argrec.use_buffering;
        interleavedComplex := argrec.interleavedComplex;
        simd_opts := argrec.simd_opts;
    else
        use_functions := arg[4];
        use_openmp := arg[5];
        use_buffering := arg[6];
        interleavedComplex := arg[7];
        simd_opts := rec();
    fi;

    opts := InitVecLibgen(isa, use_functions, use_openmp, use_buffering);
    if use_openmp then opts.language := "c.icl.openmp"; fi;

    if IsList(logn) then sizes := logn;
    else sizes := List([2 * isa.v * p^2 .. logn], d -> 2^d); fi;

    tags := When(p=1, [AVecReg(opts.vector.isa)], [AParSMP(p), AVecReg(opts.vector.isa)]);
    opts.benchTransforms := List(sizes, d -> When(interleavedComplex, InterleavedComplexT, SplitComplexT)(DFT(d)).withTags(tags));
    if p = 1 then
        PrintLine("ISA: ", isa, ", sizes: ", sizes);
        name := isa.name;
    else
        PrintLine("ISA: ", isa, ", threads: ",p,", sizes: ", sizes);
        name := Concat(StringInt(p), "p_", isa.name);
    fi;
    dpbench := DPBench(rec((name) := opts),
                    rec(timeBaseCases := false, verbosity:=0));
    return dpbench;
end;


doParSimdMddft := function(isa, p, logn, use_functions, use_openmp, use_buffering)
    local sizes, opts, dpbench, tags;
    opts := InitVecLibgen(isa, use_functions, use_openmp, use_buffering);

    if IsList(logn) then sizes := logn;
    else sizes := List(Cartesian([4..logn], [4..logn]), d -> [2^d[1], 2^d[2]]); fi;

    tags := When(p=1, [AVecReg(opts.vector.isa)], [AParSMP(p), AVecReg(opts.vector.isa)]);
    PrintLine("ISA: ", isa, ", threads: ",p,", sizes: ", sizes);

    opts.benchTransforms := List(sizes, d -> TRC(MDDFT(d)).withTags(tags));
    dpbench := DPBench(rec((Concat(StringInt(p), "p_", isa.name)) := opts),
                       rec(timeBaseCases := false, verbosity:=0));
    return dpbench;
end;

doParSimdWht := function(isa, p, logn, use_functions, use_openmp, use_buffering)
    local sizes, opts, dpbench, tags;
    opts := InitVecLibgen(isa, use_functions, use_openmp, use_buffering);

    if use_openmp then opts.language := "c.icl.openmp"; else opts.language := "c.icl.opt.core2"; fi;

    if IsList(logn) then sizes := logn;
    else sizes := [Log2Int(isa.v^2 * p^2) .. logn]; fi;

    tags := When(p=1, [AVecReg(opts.vector.isa)], [AParSMP(p), AVecReg(opts.vector.isa)]);
    opts.benchTransforms := List(sizes, d -> WHT(d).withTags(tags));
    PrintLine("ISA: ", isa, ", threads: ",p,", sizes: ", sizes);
    dpbench := DPBench(rec((Concat(StringInt(p), "p_", isa.name)) := opts),
                       rec(timeBaseCases := false, verbosity:=0));
    return dpbench;
end;

Class(cellopts_tmp, rec(
  sc := rec(tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := false),
  ic := rec(tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := true),
));

#F doMBufIndeParSimdDftCell_old([sizes], isa, p, mbuf_its)
doMBufIndeParSimdDftCell_old := function(arg)
#doMBufIndeParSimdWhtCell := function(arg)
    local sizes, opts, tags, dpbench, mbuf_its, name, isa, logn, use_functions, use_openmp,
        use_buffering, interleavedComplex, argrec, simd_opts, transform, p, extra_its;

    sizes := arg[1];
    isa   := arg[2];
    p     := arg[3];
    mbuf_its := arg[4];

    use_functions := false;
    use_openmp := true;
    use_buffering := false;
    interleavedComplex := true; #NOTE: change this?
    simd_opts := rec();

    opts := InitVecLibgenCell(isa, use_functions, use_openmp, use_buffering, cellopts_tmp.sc);
    opts.compileStrategy := IndicesCS2_FMA;
    opts.spus     := p;
    opts.multibuffer_its := mbuf_its;
    opts.codegen  := MBufCodegen;
    opts.unparser := isa.unparser;
    opts.profile  := isa.backendConfig.profile;

    #opts.globalUnrolling := opts.globalUnrolling*p;
    opts.globalUnrolling := 520;

    opts.measSteadyState := true;

    #opts.breakdownRules.GT := [ GT_Base, GT_NthLoop, GT_DFT_CT, GT_CellDMP_base, GT_CellDMP_gen, GT_Cell, GT_Vec_AxI, GT_Vec_IxA_Push, GT_Vec_IxA_L, GT_Vec_L_IxA, GT_MBufCell_spec ];
    opts.breakdownRules.GT := [ GT_Base, GT_NthLoop, GT_DFT_CT, GT_CellDMP_base, GT_CellDMP_gen, GT_Cell, GT_Vec_AxI, GT_Vec_IxA_Push, GT_Vec_IxA_L, GT_Vec_L_IxA ];
    opts.breakdownRules.TL := Concatenation(opts.breakdownRules.TL, [  TL_CellDMP ]);

    # Determine best packet size here. Packet size is simply size of kernel

    # Determine the correct # of extra loops here
    # For block size of 16384 bytes, we need (16384/(opts.vector.bits/8))/

    # WHT:
    #transform := k -> GT(WHT(LogInt(k,2)), GTPar, GTPar, [mbuf_its*p*((16384/(opts.vector.isa.bits/8))/k)]).withTags(
    #            Concatenation([ParCell(p, k*((16384/(opts.vector.isa.bits/8))/k)), MBufCell(mbuf_its)], opts.tags));

    # DFT_ic:
    transform := k -> GT( TRC(GT(DFT(k,1,false), GTPar, GTPar, [((16384/(opts.vector.isa.bits/8))/(2*k))])), GTPar, GTPar, [mbuf_its*p]).withTags(
                        Concatenation([ParCell(p, 2*k*((16384/(opts.vector.isa.bits/8))/(2*k))), MBufCell(mbuf_its)], opts.tags) );

    # DFT_sc: (untested)
    #transform := k -> GT( SplitComplexT(DFT(k,1,false)), GTPar, GTPar, [mbuf_its*p*((16384/(opts.vector.isa.bits/8))/(2*k))]).withTags(
    #                     Concatenation([ParCell(p, 2*k*((16384/(opts.vector.isa.bits/8))/(2*k))), MBufCell(mbuf_its)], opts.tags) );

    #Error("BP");
    opts.benchTransforms := List(sizes, transform);

    PrintLine("ISA: ", isa, ", Multibuf_its: ",mbuf_its,", sizes: ", sizes);
    name := Concat(StringInt(p), "p_", StringInt(mbuf_its), "mbuf_", isa.name, "_ic");

    dpbench := DPBench(rec((name) := opts),
                    rec(timeBaseCases := false, verbosity:=0));
    return dpbench;
end;

#F doMBufIndeParSimdDftCell([sizes], isa, p, mbuf_its)
#F Does a multibffered, parallel DFT, but not the IxDFT.
doMBufIndeParSimdDftCell_single := function(arg)
#doMBufIndeParSimdWhtCell := function(arg)
    local sizes, opts, tags, dpbench, mbuf_its, name, isa, logn, use_functions, use_openmp,
        use_buffering, interleavedComplex, argrec, simd_opts, transform, p, extra_its;

    sizes := arg[1];
    isa   := arg[2];
    p     := arg[3];
    mbuf_its := arg[4];

    use_functions := false;
    use_openmp := true;
    use_buffering := false;
    interleavedComplex := true; #NOTE: change this?
    simd_opts := rec();

    opts := InitVecLibgenCell(isa, use_functions, use_openmp, use_buffering, cellopts_tmp.sc);
    opts.compileStrategy := IndicesCS2_FMA;
    opts.spus     := p;
    opts.multibuffer_its := mbuf_its;
    opts.codegen  := MBufCodegen;
    opts.unparser := isa.unparser;
    opts.profile  := isa.backendConfig.profile;

    #opts.globalUnrolling := opts.globalUnrolling*p;
    opts.globalUnrolling := 520;

    opts.measSteadyState := true;

    #opts.breakdownRules.GT := [ GT_Base, GT_NthLoop, GT_DFT_CT, GT_CellDMP_base, GT_CellDMP_gen, GT_Cell, GT_Vec_AxI, GT_Vec_IxA_Push, GT_Vec_IxA_L, GT_Vec_L_IxA, GT_MBufCell_spec ];
    opts.breakdownRules.GT := [ GT_Base, GT_NthLoop, GT_DFT_CT, GT_CellDMP_base, GT_CellDMP_gen, GT_Cell, GT_Vec_AxI, GT_Vec_IxA_Push, GT_Vec_IxA_L, GT_Vec_L_IxA];
    opts.breakdownRules.TL := Concatenation(opts.breakdownRules.TL, [  TL_CellDMP ]);

    # Determine best packet size here. Packet size is simply size of kernel

    # Determine the correct # of extra loops here
    # For block size of 16384 bytes, we need (16384/(opts.vector.bits/8))/

    # WHT:
    #transform := k -> GT(WHT(LogInt(k,2)), GTPar, GTPar, [mbuf_its*p*((16384/(opts.vector.isa.bits/8))/k)]).withTags(
    #            Concatenation([ParCell(p, k*((16384/(opts.vector.isa.bits/8))/k)), MBufCell(mbuf_its)], opts.tags));

    # DFT_ic:
    #transform := k -> GT( TRC(DFT(k,1,false)), GTPar, GTPar, [mbuf_its*p]).withTags( Concatenation([ParCell(p, (2*k)), MBufCell(mbuf_its)], opts.tags) );

    # DFT_sc:
    transform := k -> GT( SplitComplexT(DFT(k,1,false)), GTPar, GTPar, [mbuf_its*p]).withTags( Concatenation([ParCell(p, (2*k)), MBufCell(mbuf_its)], opts.tags) );

    opts.benchTransforms := List(sizes, transform);

    PrintLine("ISA: ", isa, ", Multibuf_its: ",mbuf_its,", sizes: ", sizes);
    name := Concat(StringInt(mbuf_its), "mbuf_", StringInt(p), "p_", isa.name, "_sc");

    dpbench := DPBench(rec((name) := opts),
                    rec(timeBaseCases := false, verbosity:=0));
    return dpbench;
end;

#F doMBufIndeParSimdDftCell([sizes], isa, p, mbuf_its)
#F Does just the IxBase (no parallelization, no multibuffering)
doMBufIndeParSimdDftCell_IxDFTbase := function(arg)
#doMBufIndeParSimdWhtCell := function(arg)
    local sizes, opts, tags, dpbench, mbuf_its, name, isa, logn, use_functions, use_openmp,
        use_buffering, interleavedComplex, argrec, simd_opts, transform, p, extra_its;

    sizes := arg[1];
    isa   := arg[2];
    p     := arg[3];
    mbuf_its := arg[4];

    use_functions := false;
    use_openmp := true;
    use_buffering := false;
    interleavedComplex := true; #NOTE: change this?
    simd_opts := rec();

    opts := InitVecLibgenCell(isa, use_functions, use_openmp, use_buffering, cellopts_tmp.sc);
    opts.compileStrategy := IndicesCS2_FMA;
    opts.spus     := p;
    opts.multibuffer_its := mbuf_its;
    opts.codegen  := MBufCodegen;
    opts.unparser := isa.unparser;
    opts.profile  := isa.backendConfig.profile;

    #opts.globalUnrolling := opts.globalUnrolling*p;
    opts.globalUnrolling := 520;

    opts.measSteadyState := true;

    #opts.breakdownRules.GT := [ GT_Base, GT_NthLoop, GT_DFT_CT, GT_CellDMP_base, GT_CellDMP_gen, GT_Cell, GT_Vec_AxI, GT_Vec_IxA_Push, GT_Vec_IxA_L, GT_Vec_L_IxA, GT_MBufCell_spec ];
    opts.breakdownRules.GT := [ GT_Base, GT_NthLoop, GT_DFT_CT, GT_CellDMP_base, GT_CellDMP_gen, GT_Cell, GT_Vec_AxI, GT_Vec_IxA_Push, GT_Vec_IxA_L, GT_Vec_L_IxA ];
    opts.breakdownRules.TL := Concatenation(opts.breakdownRules.TL, [  TL_CellDMP ]);

    # Determine best packet size here. Packet size is simply size of kernel

    # Determine the correct # of extra loops here
    # For block size of 16384 bytes, we need (16384/(opts.vector.bits/8))/

    # WHT:
    #transform := k -> GT(WHT(LogInt(k,2)), GTPar, GTPar, [mbuf_its*p*((16384/(opts.vector.isa.bits/8))/k)]).withTags(
    #            Concatenation([ParCell(p, k*((16384/(opts.vector.isa.bits/8))/k)), MBufCell(mbuf_its)], opts.tags));

    # DFT_ic:
    #transform := k -> TRC(GT(DFT(k,1,false), GTPar, GTPar, [((16384/(opts.vector.isa.bits/8))/(2*k))])).withTags(opts.tags);

    # DFT_sc:
    transform := k -> GT( SplitComplexT(DFT(k,1,false)), GTPar, GTPar, [((16384/(opts.vector.isa.bits/8))/(2*k))] ).withTags(opts.tags);

    opts.benchTransforms := List(sizes, transform);

    PrintLine("ISA: ", isa, ", Multibuf_its: ",mbuf_its,", sizes: ", sizes);
    name := Concat("Ix_", isa.name, "_sc");

    dpbench := DPBench(rec((name) := opts),
                    rec(timeBaseCases := false, verbosity:=0));
    return dpbench;
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

# NOTE:

# Also children/rChildren is invalid throughout
# children/rChildren inconsistent with constructor

# cutoff_size is eg. 8, then vector length 4 results in VTensor(DFT(2), 4),
# which is size 16, and will be terminated to a Blk(...). How do we prevent this?

# niterate := function(clet_set)
#    s := vrec4(7, 8)[2]; s := prep(s);
#    UniteSet(clet_set, List(clets(s), CodeletName));
#    return clet_set;
# end;
#
# iterate := function(clet_set)
#    s := vrec4(7, 8)[2]; s := prep(s);
#    UniteSet(clet_set, clets(s));
#    return clet_set;
# end;
