comment("");
comment("3D Pruned Convolution from Hockney");
comment("");
#=================================================================================================
# Symmetric Symbol, size 36 ^ 3
#=================================================================================================
#====================================================================================
Import(realdft);
Import(filtering);
# set up options
# there seems to be a bug in the rules below
PRDFT1_Base1.forTransposition := false;
PRDFT1_Base2.forTransposition := false;
PRDFT3_Base1.forTransposition := false;
PRDFT3_Base2.forTransposition := false;
PRDFT1_CT_Radix2.forTransposition := false;
PRDFT3_CT_Radix2.forTransposition := false;
PRDFT1_CT.forTransposition := false;
PRDFT_PF.forTransposition := false;
PRDFT_PD.forTransposition := false;
PRDFT_Rader.forTransposition := false;
PRDFT_PD.forTransposition := false;
PRDFT3_OddToPRDFT1.forTransposition := false;
IPRDFT1_Base1.forTransposition := false;
IPRDFT1_Base2.forTransposition := false;
IPRDFT2_Base1.forTransposition := false;
IPRDFT2_Base2.forTransposition := false;
IPRDFT3_Base1.forTransposition := false;
IPRDFT3_Base2.forTransposition := false;
IPRDFT1_CT_Radix2.forTransposition := false;
IPRDFT1_CT.forTransposition := false;
IPRDFT3_CT.forTransposition := false;
IPRDFT_PD.forTransposition := false;
IPRDFT_Rader.forTransposition := false;

opts := SpiralDefaults;
#opts.useDeref := false;
opts.breakdownRules.PRDFT := [ PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_PD, PRDFT_Rader];
opts.breakdownRules.IPRDFT := [ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD, IPRDFT_Rader];
opts.breakdownRules.IPRDFT2 := [ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT];
opts.breakdownRules.PRDFT3 := [ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1];
opts.breakdownRules.URDFT := [ URDFT1_Base1, URDFT1_Base2, URDFT1_Base4, URDFT1_CT ];
opts.breakdownRules.DFT := [DFT_Base, DFT_CT, DFT_Rader, DFT_CT, DFT_GoodThomas, DFT_PD];
#opts.breakdownRules.PrunedDFT := [ PrunedDFT_base ];

opts.topTransforms := [DFT, MDDFT, PrunedDFT, PrunedIDFT, IOPrunedConv, PrunedPRDFT, PrunedIPRDFT];

n := 36;
ns := 13;
nd := 16;

nfreq := n/2+1;
name := "prconv3d_"::StringInt(n)::"_"::StringInt(ns)::"_"::StringInt(nd)::"sym";

#================
sym := var.fresh_t("S", TArray(TReal, nfreq*n^2));
i := Ind(2*nfreq*n^2);
j := Ind(2*nfreq*n^2);

# symbol symmetry
f := Lambda(j, cond(geq(j, nfreq*n^2), 2*nfreq*n^2-j-1, j));
symf := Lambda(i, nth(sym, f.at(i)));

t := IOPrunedMDRConv([n, n, n], symf, 1, [[n-nd..n-1], [n-nd..n-1], [n-nd..n-1]], 1, [[0..ns-1], [0..ns-1], [0..ns-1]], true);
rt := RandomRuleTree(t, opts);

ss := SumsRuleTree(rt, opts);

c := CodeSums(ss, opts);
Add(c.cmds[1].cmds[2].params, sym);

##  PrintCode(name, c, opts);
##  PrintTo(name::".c", PrintCode(name, c, opts));
