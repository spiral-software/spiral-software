
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#############################
# General Case Cooley-Tukey
#############################

# R1 -> (R1, C1, R3) (R1)
# R2 -> (R2, C2, R4) (R1)
PRF12_CT_Children := (N,k,PRFt,DFTt,PRFtp,PRF1) -> Map2(DivisorPairs(N),
    (m,n) -> When(IsEvenInt(n),
	[ PRFt(m,k), DFTt(m,k), PRF1(n,k), PRFtp(m,k) ],
	[ PRFt(m,k), DFTt(m,k), PRF1(n,k) ] )
);

PRF12_CT_Rule := (N,k,C,Conj,Tw) -> let(m:=Cols(C[1]), n:=Cols(C[3]), Nf:=Int(N/2),
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),

    SUM(
	RC(Scat(H(Nf+1,mf+1,0,n))) * C[1] * Gath(H(2*(nf+1)*m, m, 0, 2*(nf+1))),

	When(nc=1, [], 
	ISum(j, 
	     RC(Scat(BH(Nf+1,N,m,j+1,n))) *
	     Conj * RC(C[2]) * Tw(j) *
	     RC(Gath(H((nf+1)*m, m, j+1, nf+1))))),

	When(IsOddInt(n), [],
	RC(Scat(H(Nf+1,mc,nf,n))) * C[4] * Gath(H(2*(nf+1)*m, m, 2*nf, 2*(nf+1))))
    ) * 
    Tensor(I(m), C[3]) * L(N,m)
);


IPRF12_CT_Rule := (N,k,C,Conj,Tw) -> let(m:=Rows(C[1]), n:=Rows(C[3]), Nf:=Int(N/2),
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),

    L(N,n) * Tensor(I(m), C[3]) * 
    SUM(
	Scat(H(2*(nf+1)*m, m, 0, 2*(nf+1))) * C[1] * RC(Gath(H(Nf+1,mf+1,0,n))),

	When(nc=1, [], 
	ISum(j, 
	     RC(Scat(H((nf+1)*m, m, j+1, nf+1))) * 
	     Tw(j) *
	     RC(C[2]) * 
	     Conj * 
	     RC(Gath(BH(Nf+1,N,m,j+1,n))))),

	When(IsOddInt(n), [],
	Scat(H(2*(nf+1)*m, m, 2*nf, 2*(nf+1))) * C[4] * RC(Gath(H(Nf+1,mc,nf,n))))
    )
  );

####################
# Prime Factor    
####################

PRF12_PF_Children := (N,k,PRFt,DFTt,PRF1) -> Map2(DivisorPairsRP(N),
    (m,n) -> When(IsEvenInt(n),
	[ PRFt(m,k*n), DFTt(m,k*n), PRF1(n,k*m), PRF1(m,k*n) ],
	[ PRFt(m,k*n), DFTt(m,k*n), PRF1(n,k*m) ] )
);

 RC.toAMat := self >> AMatMat(RCMatCyc( MatSPL(self.child(1)) ));

PRF12_PF_Rule := (N,k,C) -> let(m:=Cols(C[1]), n:=Cols(C[3]), Nf:=Int(N/2), Nc:=Int((N+1)/2), 
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),
    alpha := (1/n) mod m,
    beta := (1/m) mod n,
    jj:=Ind(m), q := Ind(2*m), 

    SUM(
	RC(Scat(H(Nf+1,mf+1,0,n))) * C[1] * Gath(H(2*(nf+1)*m, m, 0, 2*(nf+1))),

	When(nc=1, [], 
	ISum(j, 
	     RC(Scat(Refl(Nf+1, N, m, HZ(N, m, (j+1)*m, n)))) *
	     # conjugate those values which indices will be reflected by the scatter Refl
	     # NOTE: Unless 1.0 is used below, vector code will break due to TVect(TInt, ...)
	     Diag(Lambda(q, cond(neq(imod(q, 2),0), cond(leq(imod(n*idiv(q, 2) + (j+1)*m, N), Nc-1), 1.0, -1.0), 1.0))) *
	     RC(C[2]) * 
	     RC(Gath(H((nf+1)*m, m, j+1, nf+1))))),

	When(IsOddInt(n), [],
	     let(inds := HZ(N, mc, nc*m, n).tolist(),
		 conj := ConcatList(inds{[1..mc]}, i -> When(i.v > Nf, [1.0,-1.0], [1.0,1.0])),

		 RC(Scat(Refl(Nf+1, N, mc, HZ(N, mc, nc*m, n)))) *
		 #RC(Scat(H(Nf+1, mc, nf, n))) *
		 Diag(conj) * C[4] * 
		 Gath(H(2*(nf+1)*m, m, 2*nf, 2*(nf+1)))
	     ))
    ) * 
    Tensor(I(m), C[3]) * CRT(m,n,1,1)
);

# s := PRF12_PF_Rule(6,1,PRF12_PF_Children(6,1,PRDFT1, DFT1, PRDFT1)[1]);
# Print(s);
# me := MatSPL(s);
# them := MatSPL(PRDFT(6));

###########
# RDFT    #
###########

Declare(PRDFT1_PF);

RulesFor(PRDFT1, rec(
    PRDFT1_Base1 := BaseRule(PRDFT1, [1, @]),
    PRDFT1_Base2 := BaseRule(PRDFT1, [2, @]),

    #F PRDFT1_CT: projection of DFT_CT 
    PRDFT1_CT := rec(
	forcePrimeFactor := false,
	isApplicable := (self, P) >> not IsPrime(P[1]) and 
	    When(self.forcePrimeFactor, not PRDFT1_PF.isApplicable(P), true),

	allChildren  := P -> PRF12_CT_Children(P[1], P[2], PRDFT1, DFT1, PRDFT3, PRDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Cols(C[1]),
	    PRF12_CT_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1))))))),

    #F PRDFT1_Complex: computes PRDFT using half of the outputs of complex DFT
    #F                this rule can be successfully pruned by the compiler, and thus
    #F                works for all sizes, including primes
    PRDFT1_Complex := rec(
	switch           := false,
	isApplicable     := P -> not Is2Power(P[1]),
	allChildren      := P -> [[ DFT(P[1], P[2]) ]],
	forTransposition := false,
	rule             := (P,C) -> let(n := P[1], nn:=Int(n/2)+1, 
	    Buf(Gath(H(2*n,2*nn, 0, 1))) * 
	    Diag(diagDirsum(fConst(2*nn,1), fConst(2*n-2*nn,0))) * 
	    RC(C[1]) *
	    Tensor(I(n), Diag(1,0)) *
	    Buf(Scat(H(2*n,n,0,2)))
	)
    ),

    # transpose of RC(DFT(n)) is RC(DFT(n,-1)), since correct transposition is not implemented, 
    # we need a separate rule
    PRDFT1_Complex_T := rec(
	switch           := false,
	isApplicable     := P -> not Is2Power(P[1]),
	allChildren      := P -> [[ DFT(P[1], -P[2]) ]],
	forTransposition := false,
	transposed := true,
	rule             := (P,C) -> let(n := P[1], nn:=Int(n/2)+1, 
	    Buf(Gath(H(2*n,n,0,2))) *
	    Tensor(I(n), Diag(1,0)) *
	    RC(C[1]) *
	    Diag(diagDirsum(fConst(2*nn,1), fConst(2*n-2*nn,0))) * 
	    Buf(Scat(H(2*n,2*nn, 0, 1)))
	)
    ),

    #F PRDFT1_Trig: PRDFT1_n -> P (DCT1_(n/2+1) dirsum DST1_(n/2-1)) A
    #F
    PRDFT1_Trig := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]) and (P[2] mod P[1]) in [1,-1 mod P[1]],
	allChildren := P -> [[ DCT1(P[1]/2+1), DST1(P[1]/2-1) ]],
	rule := (P, C) -> let(n:=P[1]/2, 
	    Z(2*n+2,2) *
	    DirectSum(Mat([[1,0],[0,0],[0,1],[0,0]]), L(2*n-2,n-1)) * 
	    DirectSum(Z(n+1,-1)*C[1], Scale(P[2],C[2])) * SymSplit1(n))),

    PRDFT1_PF := rec(
	maxSize := false, 
    	isApplicable := (self, P) >> (self.maxSize=false or P[1]<=self.maxSize) and not IsPrime(P[1]) and DivisorPairsRP(P[1])<>[],
    	allChildren  := P -> PRF12_PF_Children(P[1], P[2], PRDFT1, DFT1, PRDFT1), 
    	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Cols(C[1]),
    	    PRF12_PF_Rule(N, k, C)))
));

RulesFor(IPRDFT1, rec(
    IPRDFT1_Base1 := BaseRule(IPRDFT1, [1, @]),
    IPRDFT1_Base2 := BaseRule(IPRDFT1, [2, @]),

    IPRDFT1_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF12_CT_Children(P[1], P[2], IPRDFT1, DFT1, IPRDFT2, IPRDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Rows(C[1]),
	    IPRF12_CT_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1))))))),

    IPRDFT1_Complex := rec(
 	switch           := false,
 	isApplicable     := P -> IsPrime(P[1]),
 	allChildren      := P -> [[ DFT(P[1], P[2]) ]],
	forTransposition := false,
	rule             := (P,C) -> let(n := P[1], nn:=Int(n/2)+1, 
	    Mat(MatSPL(Gath(H(2*n,n,0,2)))) *
	    #Tensor(I(n), Diag(1,0)) *
	    RC(C[1]) *
	    Mat(MatSPL(Scat(H(2*n,2*nn, 0, 1)))) 
	)
    )
));

RulesFor(PRDFT2, rec(
    PRDFT2_Base1 := BaseRule(PRDFT2, [1, @]),
    PRDFT2_Base2 := BaseRule(PRDFT2, [2, @]),

    #F PRDFT2_CT: projection of DFT2_CT 
    PRDFT2_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF12_CT_Children(P[1], P[2], PRDFT2, DFT2, PRDFT4, PRDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Cols(C[1]),
	    PRF12_CT_Rule(N, k, C, Diag(BHD(m,-1,1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,1/2,j+1)))))))
));

RulesFor(IPRDFT3, rec(
    IPRDFT3_Base1 := BaseRule(IPRDFT3, [1, @]),
    IPRDFT3_Base2 := BaseRule(IPRDFT3, [2, @]),

    IPRDFT3_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF12_CT_Children(P[1], P[2], IPRDFT3, DFT3, IPRDFT4, IPRDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Rows(C[1]),
	    IPRF12_CT_Rule(N, k, C, Diag(BHD(m,-1,1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,1/2,j+1)))))))
));

###########
# Hartley #
###########

RulesFor(PDHT1, rec(
    PDHT1_Base2 := rec(
	isApplicable := P -> P[1]=2,
	rule := (P, C) -> Tensor(I(2), Mat([[1],[1]])) * F(2)
    ),
    
    PDHT1_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF12_CT_Children(P[1], P[2], PDHT1, DFT1, PDHT3, PDHT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Cols(C[1]),
	    PRF12_CT_Rule(N, k, C, TopHalf(m, J(2)), j->HTwid(N,m,k,0,0,j+1,J(2)).obj)))
));

RulesFor(PDHT2, rec(
   PDHT2_Base2 := rec(
	isApplicable := P -> P[1]=2,
	rule := (P, C) -> DirectSum(Mat([[1],[1]]), Mat([[1],[-1]])) * Mat(MatSPL(PRDFT2(P[1], P[2])){[1,4]})
    ),
    PDHT2_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF12_CT_Children(P[1], P[2], PDHT2, DFT2, PDHT4, PDHT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Cols(C[1]),
	    PRF12_CT_Rule(N, k, C, TopHalf(m, -J(2)), j->HTwid(N,m,k,0,1/2,j+1,-J(2)).obj)))
));
