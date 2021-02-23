
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(VPRDFT, PRDFT);

PRF12_VCT_Children := (vlen, N,k,PRFt,DFTt,PRFtp,PRF1) -> let(m:=vlen, n:=N/vlen, 
	[[ PRFt(m,k), DFTt(m,k), PRF1(n,k), PRFtp(m,k) ]]
);

Class(CompileBarrier, Buf);
CompileBarrier.code := (self, y, x) >> chain(skip(), compiler._CodeSums(self.child(1), y, x), skip());

PRF12_VCT_Rule := (N,k,C,Conj,Tw) -> let(m:=Cols(C[1]), n:=Cols(C[3]), Nf:=Int(N/2),
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),

    (SUM(
	RC(Scat(H(Nf+1,mf+1,0,n))) * C[1] * Gath(H(2*(nf+1)*m, m, 0, 1)), #2*(nf+1))),

	When(nc=1, [], 
	ISum(j, 
	     RC(Scat(BH(Nf+1,N,m,j+1,n))) *
	     Conj * RC(C[2]) * Tw(j) * L(2*m, m) * 
	     RC(Gath(H((nf+1)*m, m, (j+1)*m, 1))))),  #j+1, nf+1

	When(IsOddInt(n), [],
	RC(Scat(H(Nf+1,mc,nf,n))) * C[4] * Gath(H(2*(nf+1)*m, m, 2*m*nf, 1)))#2*nf, 2*(nf+1))))
    )) * 
    #L(Rows(C[3])*m,m)) * 
    CompileBarrier((transforms.VTensor(C[3], m)))
);

RulesFor(VPRDFT, rec(
    #F PRDFT1_CT: projection of DFT_CT 
    VPRDFT_VCT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	vlen := 4,
	allChildren  := (self, P) >> PRF12_VCT_Children(self.vlen, P[1], P[2], PRDFT1, DFT1, PRDFT3, PRDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Cols(C[1]),
	    PRF12_VCT_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1)))))))
));

# opts := CopyFields(PRDFTDefaults, rec(globalUnrolling:=0, unparser:=CMacroUnparser, includes:=["<include/real_sse.h>"], compflags:="-O1 -msse -fomit-frame-pointer"));
# c:=CodeRuleTreeOpts(RandomRuleTree(VPRDFT(16)), opts);
# PrintCode("sub1", c, opts);
# PropagateTypes(c);
# CMacroUnparser(c, 4, 4);

# r := RandomRuleTree(VPRDFT(64));
# s := RecurseOpts(r, PRDFTDefaults);
# s1 := Collect(s, RecursStep)[2];
Import(search);

VPRDFT_example_gcc := function(n)
    SwitchRulesByName(PRDFT, [PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT_Rader, PRDFT_PD, PRDFT1_PF]);
    SwitchRulesByName(PRDFT3, [PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1]);

    Local.opts := CopyFields(spiral.PRDFTDefaults, 
	rec(globalUnrolling:=34, 
	    unparser:=compiler.CMacroUnparser, 
	    includes:=["<include/real_sse.h>"], compflags:="-O3 -msse2 -fomit-frame-pointer -ffast-math"));
    if not IsBound(Local.hash) then Local.hash := HashTableDP(); fi;
    
    return DP(VPRDFT(n), rec(hashTable := Local.hash), Local.opts);
end;


VPRDFT_example := function(n)
    SwitchRulesByName(PRDFT, [PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT_Rader, PRDFT_PD, PRDFT1_PF]);
    SwitchRulesByName(PRDFT3, [PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1]);

    Local.opts := CopyFields(spiral.PRDFTDefaults, 
	rec(globalUnrolling:=34, 
	    unparser:=compiler.CMacroUnparser, 
	    includes:=["<include/real_sse.h>"], compflags:="-O3"));
    if not IsBound(Local.hash) then Local.hash := HashTableDP(); fi;
    
    return DP(VPRDFT(n), rec(hashTable := Local.hash), Local.opts);
end;


