
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

h := HashTableDP();
opts := CopyFields(PRDFTDefaults, rec(
	globalUnrolling := 64,
	compileStrategy := SimpleCS,
	compflags := "-O3 -fomit-frame-pointer"
));

opts.breakdownRules := rec(
    PRDFT := [PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT],
    PRDFT3 := [PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT],
    DFT := [DFT_Base, DFT_CT_Mincost],
    DFT3 := [DFT3_Base, DFT3_CT],

    BRDFT1 := [ BRDFT1_Base2, BRDFT1_Decomp ],
    BRDFT3 := [ BRDFT3_Base2, BRDFT3_Base4, BRDFT3_Decomp ],
    BSkewPRDFT := [ BSkewPRDFT_Base2, BSkewPRDFT_Base4, BSkewPRDFT_Decomp ],

    InfoNt := [Info_Base]
);

bopts := CopyFields(opts, rec(breakdownRules := CopyFields(opts.breakdownRules, rec(
    DFT := [DFT_Base, DFT_Bruun_Decomp],
    BSkewDFT3 := [BSkewDFT3_Base2, BSkewDFT3_Base4, BSkewDFT3_Decomp], 
    PkRDFT1 := [PkRDFT1_Base2, PkRDFT1_Bruun_Decomp],
    PRDFT3 := [PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT]
))));

	
DP(BRDFT1(64), rec(hashTable:=h), opts);
DP(PRDFT1(64), rec(hashTable:=h), opts);

brdft := List([2..6], i->HashLookup(h, BRDFT1(2^i))[1].ruletree);
prdft := List([2..6], i->HashLookup(h, PRDFT1(2^i))[1].ruletree);

codes_brdft := List(brdft, r -> CodeRuleTree(r, opts));
codes_prdft := List(prdft, r -> CodeRuleTree(r, opts));

acost_brdft := List(codes_brdft, ArithCostCode);
acost_prdft := List(codes_prdft, ArithCostCode);

# ---------------------
a := AllRuleTrees(BRDFT1(32), opts);;
p := x -> Chain(PrintLine(x), x);

codes := [];
DoForAll([1..Length(a)], i -> Add(codes, CodeRuleTree(a[p(i)], opts)));
times := List(codes, x->CMeasure(x, opts));
best := Minimum(times);
besti := Position(times, best);
bestcode := codes[besti];

a := AllRuleTrees(BRDFT1(64), opts);;
