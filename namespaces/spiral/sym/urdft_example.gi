
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(paradigms.common, platforms.sse, libgen);
ImportAll(paradigms.vector);

__gath_scat_sv := [VGath_sv, VScat_sv];
__gath_scat_v  := [VGath,    VScat   ];
__gath_scat_uv := [VGath_u,  VScat_u ];

Class(RulesUnsafeVectorizeH, RuleSet);
RewriteRules(RulesUnsafeVectorizeH, rec(
    HH_to_H :=  Rule([HH, @(1), @(2), @(3), [ListClass, @(4)]], e -> 
        H(@(1).val, @(2).val, @(3).val, @(4).val)),
    HH0_to_H := Rule([HH, @(1), @(2), @(3), [ListClass]], e -> 
        H(@(1).val, @(2).val, @(3).val, 0)),

    VGath_sv_H := Rule( [@(0,VGath_sv), [H, @(1), @(2).cond(e->_dividesUnsafe(@(0).val.v/@(0).val.sv, e)), 
                                                  @(3).cond(e->_dividesUnsafe(@(0).val.v/@(0).val.sv, e)), _1]], 
        (e, cx) -> let( d := e.v / e.sv,
                       vg := VGath(H(@(1).val/d, @(2).val/d, @(3).val/d, 1), e.v),
                       When( not IsBound(cx.opts.autolib), vg,
                           Global.autolib.RCOND("SV_H_unaligned", Global.autolib.rcondModeConservative, 
                               <# logic_and(eq(imod(@(1).val,d), 0), eq(imod(@(3).val,d), 0)), #>
                               eq(imod(@(3).val,d), 0),
                               vg, VGath_u(H(@(1).val, @(2).val/d, @(3).val, e.v), e.v))))),

    VScat_sv_H := Rule( [@(0,VScat_sv), [H, @(1), @(2).cond(e->_dividesUnsafe(@(0).val.v/@(0).val.sv, e)), 
                                                  @(3).cond(e->_dividesUnsafe(@(0).val.v/@(0).val.sv, e)), _1]], 
        (e, cx) -> let( d := e.v / e.sv, 
                       vs := VScat(H(@(1).val/d, @(2).val/d, @(3).val/d, 1), e.v),
                       When( not IsBound(cx.opts.autolib), vs,
                           Global.autolib.RCOND("SV_H_unaligned", Global.autolib.rcondModeConservative, 
                               <# logic_and(eq(imod(@(1).val,d), 0), eq(imod(@(3).val,d), 0)), #>
                               eq(imod(@(3).val,d), 0),
                               vs, VScat_u(H(@(1).val, @(2).val/d, @(3).val, e.v), e.v))))),

    VGathScat_sv_HofTensor := Rule([@(1, [VGath_sv, VScat_sv]),
                        [fCompose, [ @(2, H), 
                                     @(3),
                                     @(4).cond(e->_dividesUnsafe(@(1).val.v/@(1).val.sv, e)), 
                                     @(5).cond(e->_dividesUnsafe(@(1).val.v/@(1).val.sv, e)),
                                     _1 ],
			           [ @(6,fTensor),
			             ..., 
			             [ fId, @@(7).cond((e, cx) -> Gcd(@(1).val.v/@(1).val.sv, EvalScalar(e)) = @(1).val.v and IsBound(cx.opts.autolib)) ]]]],
        e -> let( v   := @(1).val.v,
                  sv  := @(1).val.sv,
                  n   := EvalScalar(@@(7).val),
                  gcd := Gcd(v / sv, n),
                  hp  := @(2).val.params,
                  i   := Position(__gath_scat_sv, ObjId(@(1).val)),
                  t   := fTensor(DropLast(@(6).val.children(), 1), fId(n/gcd)),
                  Global.autolib.RCOND("SV_HofTensor_unaligned", Global.autolib.rcondModeConservative, 
                      <# logic_and(eq(imod(@(3).val, gcd), 0), eq(imod(@(5).val, gcd), 0)), #>
                      eq(imod(@(5).val, gcd), 0),
                      __gath_scat_v[i] (fCompose( H(hp[1]/gcd, hp[2]/gcd, hp[3]/gcd, 1), t), v),
                      __gath_scat_uv[i](fCompose( H(hp[1],     hp[2]/gcd, hp[3],     v), t), v)))),


));

urdftFormulaStrategies := rec(
    sigmaSpl := [ StandardSumsRules ],
    postProcess := [
	StandardSumsRules,
	MergedRuleSet(StandardVecRules, RulesUnsafeVectorizeH),
        MergedRuleSet(HfuncVecRules, RulesUnsafeVectorizeH, FullVecTermRules), 
        (s, opts) -> BlockSums(opts.globalUnrolling, s),
        (s, opts) -> Process_fPrecompute(s, opts),
    ],
    rc := [],
    preRC := [],
);

urdftFormulaStrategiesScalar := rec(
    sigmaSpl := [ StandardSumsRules, HfuncSumsRules ],
    postProcess := SpiralDefaults.formulaStrategies.postProcess,
    rc := [],
    preRC := [],
);

urdftDftDouble := function(unroll)
    local opts, d;
    opts := CopyFields(CplxSpiralDefaults, rec(
            globalUnrolling := unroll,
            compileStrategy := IndicesCS,
            useDeref := true,
            hashTable := HashTableDP(),
            formulaStrategies := urdftFormulaStrategiesScalar,
            breakdownRules := rec(
                DFT    := [ DFT_Base, 
                            CopyFields(DFT_CT, rec(maxSize := 4)), 
                            CopyFields(DFT_URDFT_Decomp, rec(forTransposition := true, maxRadix := unroll)) ],

                URDFT  := [ URDFT1_Base1, URDFT1_Base2, URDFT1_Base4, 
                            CopyFields(URDFT1_CT, rec(maxRadix := unroll)) ],

		GT     := [ GT_Base.withA(maxSize => false), GT_NthLoop, GT_Vec_AxI ],
                InfoNt := [ Info_Base ])));

    d := TestBench("urdft", List([5..15], x->DFT(2^x)), opts, rec(timeBaseCases:=false, verbosity:=0));
    return d;
end;

Class(urdftUnparser, CMacroUnparserProg, SSEUnparser, rec(
    TPtr  := (self, t, vars, i, is) >>
        Cond(vars=[],
            Print("PTR_", self.declare(t.t, [], i, is), " "),
            Print("PTR_", self.declare(t.t, [], i, is),
                DoForAllButLast(vars, v->Print(" ", v.id, ",")),
                Print(" ", Last(vars).id)))
));

fixup := c-> SubstTopDownRules(c, [
  [ [mul, @(1, Value, e->e.v = [1,-1]), @(2)], e -> v_neg23(@(2).val), "toNeg23"], 
  [ [mul, @(1, Value, e->e.v = [-1,1]), @(2)], e -> v_neg01(@(2).val), "toNeg01"], 
  [ [mul, @(1, Value, e->e.v = [1,E(4)]), @(2)], e -> v_mul1j(@(2).val), "toMul1j"],
# [ [v_mul1j, [v_hi2, @(1)]], e -> v_neg2(v_revhi2(@(2).val))]
]);

urdftDftSingle := function()
    local opts, d, unroll, r, s;
    unroll := 32;
    opts := SIMDGlobals.getOpts(SSE_2x64f);
    opts := CopyFields(opts, rec(
            unparser := urdftUnparser, 
            dataType := "complex",
            customDataType := "float_cplx",
            generateComplexCode := true,
            includes := ["<include/complex_gcc_sse.h>"],
            XType := TComplex,
            YType := TComplex, 
            globalUnrolling := unroll,
            compileStrategy := Concatenation(IndicesCS, [fixup]),
            useDeref := true,
            hashTable := HashTableDP(),
            codegen := VectorCodegen,
            formulaStrategies := urdftFormulaStrategies,

            breakdownRules := CopyFields(opts.breakdownRules, rec(
                DFT    := [ DFT_Base, DFT_Base4_VecCx,
                            CopyFields(DFT_CT, rec(maxSize := 4)), 
                            CopyFields(DFT_URDFT_Decomp_VecCx, rec(maxRadix := unroll)),
                            CopyFields(DFT_URDFT_Decomp, rec(requiredFirstTag := ANoTag)) ],
                URDFT  := [ URDFT1_Base1, URDFT1_Base2, URDFT1_Base4, 
                            CopyFields(URDFT1_CT, rec(maxRadix := unroll)) ],
                rDFT   := [ rDFT_Base4, rDFT_Base4_Vec, rDFT_Decomp ], 
		GT     := [ GT_Base.withA(maxSize => false), GT_NthLoop, GT_Vec_AxI ],
                InfoNt := [ Info_Base ]
            ))));

    r := RandomRuleTree(DFT(32).withTags([AVecRegCx(SSE_4x32f)]), opts);
    s := SumsRuleTree(r, opts);
    return [opts, r, s];
end;

cplxASP := function(unroll)
    local opts, d, unroll, r, s;
    opts := CplxSpiralDefaults;
    opts := CopyFields(opts, rec(
            globalUnrolling := Maximum(opts.globalUnrolling, unroll),
            useDeref := true,
            hashTable := HashTableDP(),
            unparser := CMacroUnparserProg,
            compileStrategy := IndicesCS :: [fixup],
            includes := opts.includes ::  ["<include/complex_gcc_sse2.h>"],
            formulaStrategies := urdftFormulaStrategies,
            breakdownRules := rec(
                DFT    := [ DFT_Base, DFT_CT, DFT_GoodThomas, DFT_RealRader, DFT_PD ],
                PRDFT  := [ PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_Rader, PRDFT_PD],
                PRDFT3 := [ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1],
                ASPF   := [ 
                    ASPF_CT1_URFT.withA(maxRadix => unroll),
                    ASPF_CT3_RFT .withA(maxRadix => unroll),
                    ASPF_SmallCpx.withA(maxSize => unroll), 
                    #?> ASPF_CTSkew_RFT.withA(maxRadix => unroll),
                    ASPF_Base2, 
                    ASPF_URDFT_Base4, 
                    ASPF_RDFT1_Base4, 
                    #ASPF_rDFT_Base4,
                    ASPF_rDFT_BaseN  .withA(maxSize => unroll), 
                    ASPF_RDFT_toPRDFT.withA(maxSize => unroll),
                ],
                GT := [ GT_Base.withA(maxSize => false), GT_NthLoop ],
                InfoNt := [Info_Base]
            )));
    d := TestBench("cplx", [],  opts, rec(timeBaseCases:=false, verbosity:=0));
    return d;
end;


vecASP := function(unroll)
    local opts, d, unroll, r, s;
    opts := SIMDGlobals.getOpts(SSE_2x64f);
    opts := CopyFields(opts, rec(
            globalUnrolling := Maximum(opts.globalUnrolling, unroll),
            useDeref := true,
            hashTable := HashTableDP(),
            codegen := VectorCodegen,
	    generateComplexCode := false,

            compileStrategy := IndicesCS :: [fixup],
            includes := opts.includes ::  ["<include/vbase.h>"],
            formulaStrategies := urdftFormulaStrategies,

            breakdownRules := CopyFields(opts.breakdownRules, rec(
                ASPF := [ ASPF_CT1_URFT.withA(maxRadix => unroll),
                          ASPF_CT3_RFT .withA(maxRadix => unroll),
                          ASPF_CTSkew_RFT.withA(maxRadix => unroll, requiredFirstTag => AVecReg),
                          ASPF_Base2,
                          ASPF_RDFT1_Base4,   ASPF_RDFT1_Base4_Vec2, 
                          ASPF_URDFT_Base4,   ASPF_URDFT_Base4_Vec2, 
                          ASPF_rDFT_Base4,    ASPF_rDFT_Base4_Vec2,
                          ASPF_rDFT_BaseN.setA(maxSize => unroll),                           
                ],
                DFT := [ DFT_Base, DFT_CT ],
		GT  := [ GT_Base.withA(maxSize => false), GT_NthLoop, GT_Vec_AxI ],
                InfoNt := [ Info_Base ]
            ))));
    
    d := TestBench("rdft_2x64f", [], opts, rec(timeBaseCases:=false, verbosity:=0));
    d.transforms := List([2..15], x -> ASP.RDFT(2^x).withTags([AVecReg(SSE_2x64f)]));
    s := param(TReal, "scale");
    d.scaled := List([2..15], x -> ASPF(XN_min_1(2^x, 1), Time_TX, Freq_E(s, 2*s)));
    return d;
end;


urdftTest := function(n, unroll, do_vector)
    local opts, r, s, c;
    opts := vecASP(unroll).opts;
    r := When(do_vector, RandomRuleTree(ASP.URDFT(n).withTags([AVecReg(SSE_2x64f)]), opts), 
                         RandomRuleTree(ASP.URDFT(n), opts));
    s := SumsRuleTree(r, opts);
    c := CodeSums(s, opts);
    Try(PrintCode("sub1", c, opts));
    return [opts,r,s,c];
end;


# Example: 
# vecRdft(64, LocalConfig.bench.SSE().4x32f.1d.dft_sc.medium().getOpts());
# vecRdft(64, LocalConfig.bench.SSE().4x32f.1d.trdft().getOpts());
vecRdft := function(unroll, split_complex_vector_opts)
    local opts, isa, d;
    isa  := split_complex_vector_opts.vector.isa;
    opts := CopyFields(split_complex_vector_opts, rec(
        globalUnrolling := Maximum(split_complex_vector_opts.globalUnrolling, unroll),
        hashTable := HashTableDP(),
	generateComplexCode := false,
	finalBinSplit := true,
        includes := split_complex_vector_opts.includes ::  ["<include/vbase.h>"],

        breakdownRules := CopyFields(split_complex_vector_opts.breakdownRules, rec(
            ASPF := 
	       [ ASPF_CT1_URFT  .withA(maxRadix => unroll),
                 ASPF_CT3_RFT   .withA(maxRadix => unroll),
                 ASPF_CTSkew_RFT.withA(maxRadix => unroll, requiredFirstTag => AVecReg),
                 ASPF_Base2, 
                 ASPF_RDFT1_Base4,
		 ASPF_URDFT_Base4, 
		 ASPF_rDFT_BaseN.setA(maxSize => unroll) ]
	       ::
	       Cond(isa.v<>2, [], [ ASPF_RDFT1_Base4_Vec2,  ASPF_URDFT_Base4_Vec2,  ASPF_rDFT_Base4_Vec2 ])
	       ::
	       [ ASPF_NonSkew_Base_VecN, 
		 ASPF_NonSkew_Base_VecN_tr,
		 ASPF_RDFT1_toTRDFT .withA(maxSize => unroll),
		 ASPF_rDFT_Base_VecN.withA(maxSize => unroll) ],

            GT := [ GT_Base.withA(maxSize => false), GT_NthLoop, GT_Vec_AxI ],
            InfoNt := [Info_Base]
    ))));
    
    d := TestBench("rdft_" :: isa.name, List([2..10], x->ASP(-1).RDFT(2^x).withTags([AVecReg(isa)])), 
	           opts, rec(timeBaseCases:=false, verbosity:=0));
    d.fileTransform := (self, t, opts) >> self.outputDir :: Conf("path_sep") :: self.funcTransform(t, opts) :: ".c";
    d.funcTransform := (self, t, opts) >> "rdft_" :: String(Rows(t));
    return d;
end;

exRdftSingle := function(unroll) 
    local t, opts, r, s;
    t := vecRdft(unroll, LocalConfig.bench.SSE().4x32f.1d.dft_sc.medium().getOpts());
    opts := t.opts;
    r := SemiRandomRuleTree(t.transforms[5], ASPF_CT1_URFT, opts);
    s := SumsRuleTree(r, opts);
    return [t, opts, r, s];
end;
exRdftDouble := function(unroll) 
    local t, opts, r, s;
    t := vecRdft(unroll, LocalConfig.bench.SSE().2x64f.1d.dft_sc.medium().getOpts());
    opts := t.opts;
    r := SemiRandomRuleTree(t.transforms[5], ASPF_CT1_URFT, opts);
    s := SumsRuleTree(r, opts);
    return [t, opts, r, s];
end;
