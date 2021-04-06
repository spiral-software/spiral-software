
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(paradigms.vector.breakdown, paradigms.vector.rewrite, paradigms.vector.sigmaspl);

SwitchRulesName([IxLxI_kmn_n, IxLxI_kmn_km, IxLxI_IxLxI_up, IxLxI_IxLxI_down], false);
SwitchRulesName([L_base, L_nv_n_vec, L_nv_v_vec, L_mn_m_vec, IxLxI_vtensor], true);

# FF: if this is turned off, DP tries to measure TL and complex code using CR(.) dies. Why did I need it??
#TL.doNotMeasure := false;

FixDataType := function(c, opts)
    local vars, v, srec;

    # Vectors of vectors must be flattened
#    c := SubstTopDown(c, @(1, TVect, e->ObjId(e.t)=TVect), e-> TVect(@(1).val.t.t, @(1).val.size*@(1).val.t.size));

    if IsBound(opts.vector.isa.needScalarVarFix) and opts.vector.isa.needScalarVarFix then
        vars := Collect(c, @(1, var, e->e.t=TVect(TReal, 2)));
        srec := rec();

        for v in vars do
            srec.(v.id) := opts.vector.isa.scalarVar();
        od;
        SubstVars(c, srec);
#        c := DeclareVars(c);
    fi;
    return c;
end;


# Populate this with default parameter values
VectorDefaults := rec(
    expandConstants := false,
    dontRuntimeLocalize := false,
);

DEFAULT_UNROLL := 55;   # largest prime: 13, 4-way -> need 40 and 52 be unrolled
UNROLL_LO := 16;
UNROLL_HI := 64;
VERBOSITY := 0;
FAILTOL := false;
HASH := false;

_help := function()
    PrintLine("\nopts := SIMDGlobals.getOpts(<arch>, flags);" );
    PrintLine("dpopts := SIMDGlobals.getDPOpts(<arch>, opts); # for correct DP options");
    PrintLine("\nsupported flags:");
    PrintLine("scalar DFT configuartion: CT, PFA, PFA_maxSize, PD, Rader, RealRader, Rader_maxSize, Bluestein, Bluestein_maxSize, minCost, splitRadix, PRDFT, URDFT");
    PrintLine("DFT format: interleavedComplex");
    PrintLine("tspl DFT configuartion: tsplCT, tsplPFA, tsplRader, tsplBluestein, bluesteinMinPrime, bluesteinExtraSizes, raderAvoidSizes");
    PrintLine("vectorization: svct, splitL, oddSizes, realVect, cplxVect, useConj, stdTTensor, flipIxA, pushTag");
    PrintLine("options: globalUnrolling, useArea, verify, verifyDP, verifyTolerance, verbosity, faultTolerant, mode, language, highPerf, propagateNth, useDeref\n");
end;


Class(SIMDGlobals, rec(
    svct := True,
    DPOpts := rec(
        globalUnrolling := true,
        globalUnrollingMin := 128,
        globalUnrollingMax := 128,
        defaultUnrolling := 128, # NOTE: THIS AFFECTS Rader.maxSize and other things. BAD IDEA.
        verbosity := 0,
        faultTolerant := false
    ),

    getDPOpts := meth(arg)
        local self, isa, opts;
        if Length(arg) = 1 then
            _help();
            return;
        fi;
        self := arg[1];
        isa := arg[2];
        opts := When(Length(arg) >=3, arg[3], rec());

        return rec(
            globalUnrolling := When(IsBound(opts.globalUnrolling) or (IsBound(opts.globalUnrollingMin) and IsBound(opts.globalUnrollingMax)), true, self.DPOpts.globalUnrolling),
            globalUnrollingMin := Cond(IsBound(opts.globalUnrolling), opts.globalUnrolling, IsBound(opts.globalUnrollingMin), opts.globalUnrollingMin, self.DPOpts.globalUnrollingMin * isa.v),
            globalUnrollingMax := Cond(IsBound(opts.globalUnrolling), opts.globalUnrolling, IsBound(opts.globalUnrollingMax), opts.globalUnrollingMax, self.DPOpts.globalUnrollingMax * isa.v),
            verbosity := When(IsBound(opts.verbosity), opts.verbosity, self.DPOpts.verbosity),
            timeBaseCases := true
        );
    end,

#------------------------------------------
    getOpts := meth(arg)

         local self, isa, argrec, opts, profile_rec, arg3, bsizes, unr;

        if Length(arg) = 1 then
            _help();
            return;
        fi;
        self := arg[1];
        isa := arg[2];

        #   some defaults
        argrec := rec(
            #   stride permutation configuration
            stride := false,

            #   DFT configuration
            #   scalar algorithms
            splitRadix := false,
            SplitRadix_maxSize := SIMDGlobals.DPOpts.defaultUnrolling/2,
            CT  := true,
            CT_forcePrimeFactor := false,
            PFA := true,
            PFA_maxSize := SIMDGlobals.DPOpts.defaultUnrolling/2,
            PD  := true,
            Rader := false,
            RealRader := true,
            Rader_maxSize := SIMDGlobals.DPOpts.defaultUnrolling/2,
            Rader_minSize := 3,
            Bluestein := false,
            Bluestein_maxSize := false,
            minCost := false,
            Mincost_maxSize := SIMDGlobals.DPOpts.defaultUnrolling/2,
            PRDFT := false,
            PRDFT_PF_maxSize := SIMDGlobals.DPOpts.defaultUnrolling,
            URDFT := false,
            URDFT_maxRadix := SIMDGlobals.DPOpts.defaultUnrolling/2,
            #   vector algorithms -- preset for small sizes unrolled code.
            svct := true,
            splitL := false,    #   turn on for large sizes, as SVCT's payoff diminishes and complicates matters
            useConj := false,   #   not supported right now
            oddSizes := false,   #   only for unrolled code. turn off splitL as DP takes a wrong turn for small sizes
            realVect := true,   #   use real-based vectorization
            cplxVect := false,   #   use complex-based vectorization
            pushTag := true,
            flipIxA := true,
            splitComplexTPrm := false, # used in SplitComplex DFTs and in DCTs
            TRCDiag_VRCLR := false,
            #   tSPL breakdown rules
            tsplBluestein := true,
            bluesteinMinPrime := 997,   # maybe level-based threshold? basically no automatic Bluestein applicability for now
            bluesteinExtraSizes := [ 23, 46, 47, 49, 59, 67, 79, 83, 94, 103, 106, 107, 115 ],  #   for these sizes I needed Bluestein so far
            raderAvoidSizes := [ 47, 59, 107 ], # avoid Rader for these sizes as codesize explodes for unrolled code
            tsplRader := true,
            tsplPFA := true,
            stdTTensor := true,
            tsplCT := true,
            tsplVBase := true,
            tsplCxVBase := true,
            tsplCT_oddvloop := false,
            loopOddSizes := false,
            TRCbyDef := false,
            #   other defaults
            includeMath   := true,
            faultTolerant := SIMDGlobals.DPOpts.faultTolerant,
            language      := Cond(LocalConfig.cpuinfo.default_lang <> "", LocalConfig.cpuinfo.default_lang, SpiralDefaults.language),
            verify        := false,
            verifyDP := false,
            verifyTolerance := 1E-3,
            verbosity := 0,
            interleavedComplex := true,
            highPerf := true,
            SIMD := LocalConfig.cpuinfo.SIMDname,
            fracbits := When(IsBound(isa.fracbits), isa.fracbits, false),
            useArea := false,
            processIntTables := false,
            propagateNth := false,
            useDeref := true,
            globalUnrolling := SIMDGlobals.DPOpts.defaultUnrolling,

            fma := false,
            cxfma := false,
            splitVDiag := true,
            finalSReduce := false,
            fixUnalignedLoadStore := false
        );

        if Length(arg) = 3 then
            arg3 := arg[3];
            if IsBound(arg3.__name__)  then Unbind(arg3.__name__); fi;
            if IsBound(arg3.__doc__)   then Unbind(arg3.__doc__); fi;
            if IsBound(arg3.__bases__) then Unbind(arg3.__bases__); fi;
            argrec := CopyFields(argrec, arg3);
        fi;

     profile_rec :=
        When(IsBound(SpiralDefaults.profile),
           rec(
              profile := SpiralDefaults.profile,
	          measureFunction := SpiralDefaults.profile.meas
           ),
           rec()
     );

     bsizes := Filtered(argrec.bluesteinExtraSizes, i -> not IsInt(i/isa.v^2));

     opts := CopyFields(When(argrec.highPerf, SpiralDefaults.highPerf(), SpiralDefaults), isa.getOpts(), profile_rec, rec(
         baseHashes := When(argrec.svct or argrec.flipIxA, [SIMD_ISA_DB.getHash()], []),
         breakdownRules := rec(
             TTwiddle := [TTwiddle_Tw1],
             DFT := Concat([DFT_Base],
                When(argrec.realVect and argrec.tsplVBase,   [DFT_tSPL_VBase], []),
                When(argrec.cplxVect and argrec.tsplCxVBase, [DFT_tSPL_CxVBase, DFT_tSPL_CxVBase2], []),
                When(argrec.splitRadix, [CopyFields(DFT_SplitRadix, rec(maxSize := argrec.SplitRadix_maxSize)) ], []),
                When(argrec.PD,         [DFT_PD], []),

                When(argrec.Bluestein,  [CopyFields(DFT_Bluestein, rec(
		            maxSize := argrec.Bluestein_maxSize, switch := true))], []),

                When(argrec.CT,         [CopyFields(DFT_CT, rec(forcePrimeFactor := argrec.CT_forcePrimeFactor))],
		            When(argrec.URDFT and not argrec.minCost, [CopyFields(DFT_CT, rec(maxSize := 4))], [])),

                When(argrec.PFA,        [CopyFields(DFT_GoodThomas, rec(maxSize := argrec.PFA_maxSize))], []),
                When(argrec.Rader,      [CopyFields(DFT_Rader,      rec(minSize := argrec.Rader_minSize, maxSize := argrec.Rader_maxSize))], []),
                When(argrec.RealRader,  [CopyFields(DFT_RealRader,  rec(maxSize := argrec.Rader_maxSize))], []),
                When(argrec.minCost,    [CopyFields(DFT_CT_Mincost, rec(maxSize := argrec.Mincost_maxSize))], []),
                When(argrec.tsplCT,     [DFT_tSPL_CT], []),
                When(argrec.tsplCT_oddvloop,[DFT_tSPL_CT_oddvloop], []),
                When(argrec.tsplBluestein, [ CopyFields(DFT_tSPL_Bluestein, rec(
                    applicableSizes := i -> (i in bsizes) or (not IsInt(i/isa.v^2)
			                    and ForAny(Factors(i), j -> j >= argrec.bluesteinMinPrime)),
                    minRoundup := isa.v^2,
                    customFilter := DetachFunc(Subst(i -> IsInt(i/$(isa.v^2)))))) ], []),

                When(argrec.tsplPFA,    [DFT_tSPL_GoodThomas], []),

                When(argrec.tsplRader,  [CopyFields(DFT_tSPL_Rader, rec(
                    useSymmetricAlgorithm := true,
                    avoidSizes := argrec.raderAvoidSizes)) ], []),

                When(argrec.PRDFT,      [DFT_PRDFT], []),
                When(argrec.URDFT,      [CopyFields(spiral.sym.DFT_URDFT_Decomp, rec(
		    forTransposition := true,
		    maxRadix := argrec.URDFT_maxRadix))], [])
             ),
             DFT3 := [ DFT3_Base, DFT3_CT ],
             TDCT2 := [ DCT2_DCT4_tSPL ],
             TDCT3 := [ DCT3_DCT4_tSPL ],
             TDCT4 := [ DCT4_CT_tSPL ],
             TDST2 := [ DST2_DST4_tSPL ],
             TDST3 := [ DST3_DST4_tSPL ],
             TDST4 := [ DST4_CT_tSPL ],
             TMDCT := [TMDCT_DCT4_tSPL],
             TIMDCT   := [TIMDCT_TMDCT_tSPL],
             TRDFT    := [TRDFT_By_Def, TRDFT_By_Def_tr, TRDFT_DFT_NR_tSPL_New, TRDFT_CT_tSPL_New],
             TRDFT2D  := [TRDFT2D_ColRow_tSPL],
             TIRDFT2D := [TIRDFT2D_RowCol_tSPL],
             TRConv2D := [TRConv2D_TRDFT2D_tSPL],
             TDHT := [DHT_DFT_tSPL],
             TS := [ TS_vect ],
             TConjEven := [TConjEven_vec, TConjEven_vec_tr ],
             TXMatDHT := [TXMatDHT_vec],
             WHT := [ WHT_Base, WHT_BinSplit, WHT_tSPL_BinSplit, WHT_tSPL_Base ],
             MDDFT := [ MDDFT_Base, MDDFT_RowCol, MDDFT_tSPL_RowCol ],
             PrunedDFT := [ PrunedDFT_base, PrunedDFT_DFT, PrunedDFT_CT, PrunedDFT_tSPL_CT ],
             IOPrunedDFT := [
		 IOPrunedDFT_tSPL_CT, IOPrunedDFT_base, IOPrunedDFT__PrunedDFT,
		 IOPrunedDFT__PrunedDFT_T, IOPrunedDFT__Gath_PrunedDFT, IOPrunedDFT__PrunedDFT_T_Scat, IOPrunedDFT_CT ],
             InterpolateDFT := [ InterpolateDFT_tSPL_PrunedDFT ],
             Downsample := [ Downsample_base, Downsample_tag ],
             InterpolateSegmentDFT := [ InterpolateSegmentDFT_PrunedDFT, InterpolateSegmentDFT_tSPL_PrunedDFT ],

             TTensor := Concat(When(argrec.stdTTensor, [ AxI_IxB, IxB_AxI], []),
                        When(argrec.tsplPFA, [splitL_BxI__L_AxI, AxI_L__BxI_splitL], [L_BxI__L_AxI, AxI_L__BxI_L ])),

             TTensorI := Concat([ IxA_base, AxI_base, IxA_L_base, L_IxA_base, AxI_vec ],
                            When(argrec.pushTag, [ IxA_vec_push ], []),
                            When(argrec.flipIxA, [ IxA_vec ], []),
                            When(argrec.svct, [ IxA_L_vec, L_IxA_vec], []),
                            When(argrec.splitL, [ IxA_L_split_vec, L_IxA_split_vec, IxA_split_vec ], []),
                            When(argrec.oddSizes, [ AxI_svec ], []),
                            When(argrec.loopOddSizes, [ TTensorI_oddvloop ], []),
                            When(argrec.oddSizes and argrec.svct, [ IxA_L_svec ], []),
                            When(argrec.useConj, [ IxA_conj_vec ], [])
                            ),
             TL := Concat(When(argrec.cplxVect, [L_cx_real], []),
                        When(argrec.stride, [L_GV1_vtensor], []),#[L_base_vec, L_mn_m_vec, IxLxI_vtensor],
                        When(argrec.svct or argrec.flipIxA, [ L_nv_n_vec, L_nv_v_vec, L_mn_m_vec, IxLxI_vtensor ], [])
#                        When(argrec.splitL,[ L_base_vec, IxLxI_vtensor ], [])
                    ),
             TTensorInd := [ dsA_base_vec_push, L_dsA_L_base_vec, L_dsA_L_vec, L_dsA_L_base ],

             TRC := Concat(
                When(argrec.realVect, [TRC_vect], []),
                When(argrec.cplxVect, [TRC_cplx, TRC_cplx_v2], []),
		        When(argrec.TRCbyDef, [CopyFields(TRC_By_Def, rec(maxSize := 2*isa.v))], []),
                When(not(argrec.realVect) and argrec.cplxVect, [TRC_cplxvect], [])
             ),
	     TMat    := [ TMat_Base, TMat_Vec],
             TDiag   := [ TDiag_tag ],
             TRCDiag := When(argrec.TRCDiag_VRCLR, [TRCDiag_VRCLR], [ TRCDiag_tag ]),
             TGath   := [ TGath_base ],
             TScat   := [ TScat_base ],
             TRDiag  := [ TRDiag_Vec ],
             TPrm    := When(argrec.splitComplexTPrm, [TPrm_format], [ TPrm_IJ_Vec, TPrm_IP_Base1, TPrm_J, TPrm_Jv ]),
             TGrp    := [ TGrp_tag ],
             TCompose   := [ TCompose_tag ],
             TICompose  := [ TICompose_tag ],
             TRaderMid  := [ Pad_vec ],
             TDirectSum := [ A_dirsum_B_delayed ],

             TConj   := [ TConj_perm ],
             PRDFT   := [ PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT,
		 CopyFields(PRDFT1_PF, rec(maxSize := argrec.PRDFT_PF_maxSize)), PRDFT_PD, PRDFT_Rader],
             IPRDFT  := [ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD, IPRDFT_Rader],
             IPRDFT2 := [ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT],
             PRDFT3  := [ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1],
             URDFT   := [ URDFT1_Base1, URDFT1_Base2, URDFT1_Base4, CopyFields(URDFT1_CT, rec(maxRadix := argrec.URDFT_maxRadix)) ],
             GT     := [ CopyFields(GT_Base, rec(maxSize:=false)), GT_NthLoop, GT_Vec_AxI ],
             InfoNt := [ Info_Base],
             Filt   := [ spiral.transforms.filtering.Filt_Base, spiral.transforms.filtering.Filt_Blocking ],

             # old DCT/DST rules without DFT/RDFT termination
             DCT2   := [ DCT2_DCT2and4],
             DCT3   := [ DCT3_Base2, DCT3_Base3, DCT3_Base5],
             DCT4   := [ DCT4_Base2, DCT4_Base3, DCT4_DCT2andDST2, DCT4_DST4andDST2, DCT4_DCT2, DCT4_DCT2t],
             DST2   := [ DST2_Base2, DST2_toDCT2],
             DST4   := [ DST4_Base, DST4_toDCT4]
         ),
         formulaStrategies := rec(
             sigmaSpl := VectorStrategySum,
             postProcess := Concatenation(
                VectorStrategySum,
                VectorStrategyTerm,
                VectorStrategySum,
                VectorStrategySum,
                VectorStrategyTerm2,
                VectorStrategySum,
                VectorStrategySum,
                VectorStrategyRC,
                VectorStrategyRC,
                [RulesVRCTerm, s -> SubstBottomUp(s, VIxL, e->e.implement(isa))], # NOTE: WHY do I need to have these???
                VectorStrategySum,
                VectorStrategyRC,
                VectorStrategySum,
                [ BlockSumsOpts,
                  (s, opts) -> Process_fPrecompute(s, opts)
                   ],
                fix_fAdd
             ),
             preRC := [MergedRuleSet(StandardSumsRules, RulesPropagate, JoinDirectSums),
                       MergedRuleSet(StandardSumsRules, RulesPropagate, StretchRaderMid),
                       MergedRuleSet(StandardSumsRules, RulesPropagate, RulesSplitComplex),
                       TerminateDirectSums],
             rc := []
         ),

         includes := Concatenation(
            When(argrec.includeMath = false, "", ["<math.h>"]),
            When(IsBound(isa.includes), isa.includes(), [])
         ),

         generateInitFunc := true,

         XType := TPtr(isa.t.base_t()),
         YType := TPtr(isa.t.base_t()),

     # NOTE: use layering w/ CopyFields instead of When/IsBound
     language := argrec.language,
     faultTolerant := argrec.faultTolerant,
     verify := argrec.verify,
     verifyDP := argrec.verifyDP,
     verifyTolerance := argrec.verifyTolerance,
     verbosity := argrec.verbosity,
     interleavedComplex := argrec.interleavedComplex,
     useDeref := argrec.useDeref,
     propagateNth := argrec.propagateNth,
     unparser := When(IsBound(isa.unparser), isa.unparser, CMacroUnparserProg),
     codegen := VectorCodegen,
     compileStrategy := Concatenation(
                            Cond( argrec.cxfma, IndicesCS_CXFMA,
                                  argrec.fma,   IndicesCS_FMA,
                                  # else
                                  SpiralDefaults.compileStrategy),
                            When( isa.isFixedPoint, [ c -> FixedPointCode(c, isa.bits, argrec.fracbits) ] ,[]),
                            [(c,opts) -> opts.vector.isa.fixProblems(c, opts)],
                            [HashConsts, FixDataType]),

     simpIndicesInside := SpiralDefaults.simpIndicesInside :: isa.simpIndicesInside,

     vector := CopyFields(VectorDefaults, rec(
            vlen := isa.v,
            isa := isa,
            conf := argrec,
            SIMD := argrec.SIMD
        )),
        tags := isa.getTags(),
        cxtags := isa.getTagsCx()
     ));

     if IsBound(isa.countrec) then opts.countrec := isa.countrec; fi;

     if IsBound(isa.useDeref) then opts.useDeref := isa.useDeref; fi;
     if IsBound(isa.codegenStrat) then opts.codegenStrat := isa.codegenStrat; fi;
     if IsBound(isa.compileStrategy) then opts.compileStrategy := isa.compileStrategy(); fi;

     if IsBound(isa.declareConstants) then opts.declareConstants := isa.declareConstants; fi;
     opts.scalarDataModifier := "const";
     if IsBound(isa.expandVectorConstants) then opts.expandVectorConstants := isa.expandVectorConstants; fi;

     if IsBound(isa.backendConfig) then
        opts.profile := isa.backendConfig.profile;
        opts.measureFunction := isa.backendConfig.measureFunction;
     fi;

     # vector has VRC to deal with complex code so whe turn off implicit RC business in SumsRuleTreeXXX()
     opts.generateComplexCode := true;

     if IsBound(isa.arrayBufModifier) then
        opts.arrayBufModifier :=isa.arrayBufModifier;
	 elif IsBound(isa.alignmentBytes) then
	     opts.arrayBufModifier := Concat("static ", LocalConfig.compilerinfo.alignmentSpecifier(isa.alignmentBytes));
     else
         opts.arrayBufModifier := Concat("static ", LocalConfig.compilerinfo.alignmentSpecifier());
     fi;
     if IsBound(isa.arrayDataModifier) then
        opts.arrayDataModifier :=isa.arrayDataModifier;
	 elif IsBound(isa.alignmentBytes) then
	     opts.arrayDataModifier := Concat("static ", LocalConfig.compilerinfo.alignmentSpecifier(isa.alignmentBytes));
     else
         opts.arrayDataModifier := Concat("static ", LocalConfig.compilerinfo.alignmentSpecifier());
     fi;

     if IsBound(argrec.globalUnrolling) then
        # complex vectorization requires smaller global unrolling
        opts.globalUnrolling := argrec.globalUnrolling * isa.v; # /When(argrec.cplxVect, 2, 1); #NOTE: bad problem when wrapping complex guys...
     fi;
     if argrec.useArea then
        opts.globalUnrolling := 2 * opts.globalUnrolling; # * Log2Int(opts.globalUnrolling);
        opts.markBlock := MarkBlocksAreaSums;
     fi;

     if argrec.splitVDiag then
        opts.hack_vRC_VDiag := "split";
     else
        opts.hack_vRC_VDiag := "compact";
     fi;
     opts.finalSReduce := argrec.finalSReduce;
     opts.fixUnalignedLoadStore := argrec.fixUnalignedLoadStore;

#     # use Fred's magic flags to make the compiler behave -- hopefully
#     opts.propagateNth:=false;
#     opts.useDeref := true;
#     opts.doScalarReplacement:=false;
#     if argrec.safeMode then
#         opts.useDeref:=false;
#         opts.doScalarReplacement:=false;
#     fi;

      if argrec.processIntTables then
          Append(opts.formulaStrategies.preRC, [compiler.MergeIntData]);
      fi;

     opts.operations := rec(Print := (s) -> Print("<Spiral SIMD options>"));

     if IsBound(argrec.measureFinal) then opts.measureFinal := argrec.measureFinal; fi;

     return opts;

    end
));
