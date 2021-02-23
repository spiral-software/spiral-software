
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

X.t.t := TComplex;
Y.t.t := TComplex;

# NOTE: Double RC's happen if PRDFT_CT rules are enabled,
#        eg. RC(RCdiag) breaks

SwitchRulesByName(DFT, [DFT_Base, DFT_PRDFT, DFT_CT_Mincost, DFT_Rader, DFT_PD]);
SwitchRulesByName(DFT3, [DFT3_Base, DFT3_CT, DFT3_PRDFT3]);
SwitchRulesByName(PRDFT, [PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT_Radix2, PRDFT_PD]);
SwitchRulesByName(PRDFT3, [PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT_Radix2, PRDFT3_OddToPRDFT1]);
SwitchRulesByName(PRDFT4, [PRDFT4_Base1, PRDFT4_Base2, PRDFT4_CT]);

DFT2_PRDFT2.forTransposition := false;
DFT3_PRDFT3.forTransposition := false;
DFT4_PRDFT4.forTransposition := false;

opts := CopyFields(SpiralDefaults, rec(
	dataType := "complex",
	generateComplexCode := true,
	includes := ["<include/complex_gcc_sse2.h>"],
	compflags := "-O1  -fomit-frame-pointer -malign-double -fstrict-aliasing -march=pentium4 -msse2",

	unparser := CMacroUnparser,
	formulaStrategies := rec(
	    sigmaSpl := LibStrategy,
	    rc := RCStrategy,
	    postProcess := [ ProcessMemos, Localize ] 
	),
        # NOTE: memo variables were breaking with other compile strategies
#	compileStrategy := Concatenation(SimpleCS, [VHashConstantsCode,DeclareHidden])
	compileStrategy := SimpleCS
));

c1:=CodeRuleTreeOpts(SemiRandomRuleTree(DFT(5), DFT_PRDFT), opts);
c2:=CodeRuleTreeOpts(SemiRandomRuleTree(DFT(5), DFT_Rader), opts);
c3:=CodeRuleTreeOpts(SemiRandomRuleTree(DFT(5), DFT_PD), opts);

PrintCode("sub", c2, opts);

bf:=HashTableDP();
DP(DFT(16), rec(hashTable:=bf, verbosity:=4), opts);

# 43 82 246 800
