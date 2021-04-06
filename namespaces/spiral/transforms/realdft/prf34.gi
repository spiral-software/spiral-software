
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#####################
# General Rule
#####################

PRF34_CT_Children := (N,k,DFTp,PRF3,PRFt) -> Map2(DivisorPairs(N),
    (m,n) -> When(IsEvenInt(n),
	[ DFTp(m, k), PRF3(n, k) ],
	[ DFTp(m, k), PRF3(n, k), PRFt(m, k) ])
);

PRF34_CT_Rule := (N,k,C,Conj,Tw) -> let(
    m:=Rows(C[1]), n:=Cols(C[2]), Nc:=Int((N+1)/2), 
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nf),
    
    SUM(
	ISum(j, RC(Scat(BH(Nc, N-1, m, j, n))) *
	        Conj * RC(C[1]) * Tw(j) * 
		RC(Gath(H(nc*m, m, j, nc)))
	), 
	When(IsEvenInt(n), [], 
	     RC(Scat(H(Nc, mc, nf, n))) * C[3] * Gath(H(2*nc*m, m, 2*nf, 2*nc)))
    ) *
    Tensor(I(m), C[2]) *
    L(N,m)
);

IPRF34_CT_Rule := (N,k,C,Conj,Tw) -> let(
    m:=Rows(C[1]), n:=Rows(C[2]), Nc:=Int((N+1)/2), 
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nf),
    
    L(N,n) *
    Tensor(I(m), C[2]) *
    SUM(
	ISum(j, 
	    RC(Scat(H(nc*m, m, j, nc))) *
	    Tw(j) * 
	    RC(C[1]) * 
	    Conj * 
	    RC(Gath(BH(Nc, N-1, m, j, n)))
	), 
	When(IsEvenInt(n), [], 
	     Scat(H(2*nc*m, m, 2*nf, 2*nc)) * C[3] * RC(Gath(H(Nc, mc, nf, n))))
    )
);

#####################
# Special Cases
#####################

RulesFor(PRDFT4, rec(
    PRDFT4_Base1 := BaseRule(PRDFT4, [1, @]),
    PRDFT4_Base2 := rec( 
	isApplicable := P -> P[1]=2,
	rule := (P, C) -> F(2) * Mat(1/2*MatSPL(PDHT4(2,P[2])))  # DHT4(2) is a diagonal
    ),
    PRDFT4_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF34_CT_Children(P[1], P[2], DFT2, PRDFT3, PRDFT4), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Rows(C[1]),
	    PRF34_CT_Rule(N, k, C, Diag(BHD(m,-1,1)), j->RC(Diag(fPrecompute(Twid(N,m,k,1/2,1/2,j))))))),
    PRDFT4_Trig := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]) and P[2]=1,
	allChildren := P -> [[ DCT4(P[1]/2), DST4(P[1]/2) ]],
	rule := (P, C) -> let(n:=P[1]/2, 
	    L(2*n, n) * DirectSum(C[1], C[2]) * SymSplit4(n)))
));

RulesFor(IPRDFT4, rec(
    IPRDFT4_Base1 := BaseRule(IPRDFT4, [1, @]),
    IPRDFT4_Base2 := rec( 
	isApplicable := P -> P[1]=2,
	rule := (P, C) -> Mat(1/2*MatSPL(PDHT4(2,P[2]))) * F(2)  # DHT4(2) is a diagonal
    ),
    IPRDFT4_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF34_CT_Children(P[1], P[2], DFT3, IPRDFT2, IPRDFT4), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Rows(C[1]),
	    IPRF34_CT_Rule(N, k, C, Diag(BHD(m,-1,1)), j->RC(Diag(fPrecompute(Twid(N,m,k,1/2,1/2,j)))))))
));

RulesFor(PRDFT3, rec(
    PRDFT3_Base1 := BaseRule(PRDFT3, [1, @]),
    PRDFT3_Base2 := rec(
	isApplicable := P -> P[1]=2,
	rule := (P, C) -> Cond(
	    P[2] mod 4 = 1, I(2), 
	    P[2] mod 4 = 3, Diag(1,-1), 
	    Error("Bad second parameter for PRDFT3(n,k), gcd(n,k)=1 does not hold"))
    ),
    PRDFT3_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF34_CT_Children(P[1], P[2], DFT1, PRDFT3, PRDFT3), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Rows(C[1]),
	    PRF34_CT_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,1/2,0,j))))))),
    PRDFT3_Trig := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]) and P[2]=1,
	allChildren := P -> [[ DCT3(P[1]/2), DST3(P[1]/2) ]],
	rule := (P, C) -> let(n:=P[1]/2, 
	    L(2*n, n) * DirectSum(C[1], C[2]) * SymSplit3(n))),

    PRDFT3_OddToPRDFT1 := rec(
	isApplicable := P -> IsOddInt(P[1]),
	allChildren := P -> [[ PRDFT(P[1], P[2]) ]],
	rule := (P, C) -> rperm_ev(P[1]) * C[1] * pdiag(P[1], P[2])
    )
));

RulesFor(IPRDFT2, rec(
    IPRDFT2_Base1 := BaseRule(IPRDFT2, [1, @]),
    IPRDFT2_Base2 := rec(
	isApplicable := P -> P[1]=2,
	rule := (P, C) -> Cond(
	    P[2] mod 4 = 1, Diag(2,-2),
	    P[2] mod 4 = 3, Diag(2,2),
	    Error("Bad second parameter for IPRDFT2(n,k), gcd(n,k)=1 does not hold"))
    ),
    IPRDFT2_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF34_CT_Children(P[1], P[2], DFT1, IPRDFT2, IPRDFT2), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Rows(C[1]),
	    IPRF34_CT_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,1/2,0,j)))))))
));

RulesFor(PDHT4, rec(
    PDHT4_Base2 := BaseRule(PDHT4, [2, @]),
    PDHT4_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF34_CT_Children(P[1], P[2], DFT2, PDHT3, PDHT4), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Rows(C[1]),
	    PRF34_CT_Rule(N, k, C, TopHalf(m, -J(2)), 
		          j -> RCDiag(RCData(fPrecompute(Twid(N,m,-k,1/2,1/2,j))), -J(2))))),
    PDHT4_Trig := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]) and P[2]=1,
	allChildren := P -> [[ DCT4(P[1]/2), DST4(P[1]/2) ]],
	rule := (P, C) -> let(n:=P[1]/2, 
	    Tensor(I(n), F(2)) * L(2*n,n) * DirectSum(C[1], C[2]) * SymSplit4(n)))
));

RulesFor(PDHT3, rec(
    PDHT3_Base2 := rec(
	isApplicable := P -> P[1]=2,
	rule := (P, C) -> Cond(
	    P[2] mod 4 = 1, F(2), 
	    P[2] mod 4 = 3, J(2)*F(2),
	    Error("Bad second parameter for PRDFT3(n,k), gcd(n,k)=1 does not hold"))
    ),

    PDHT3_CT := rec(
	isApplicable := P -> not IsPrime(P[1]),
	allChildren  := P -> PRF34_CT_Children(P[1], P[2], DFT1, PDHT3, PDHT3), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=Rows(C[1]),
	    PRF34_CT_Rule(N, k, C, TopHalf(m, J(2)), 
		          j -> RCDiag(RCData(fPrecompute(Twid(N,m,-k,1/2,0,j))), J(2))))),
    PDHT3_Trig := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]) and P[2]=1,
	allChildren := P -> [[ DCT3(P[1]/2), DST3(P[1]/2) ]],
	rule := (P, C) -> let(n:=P[1]/2, 
	    Tensor(I(n), F(2)) * L(2*n,n) * DirectSum(C[1], C[2]) * SymSplit3(n)))

));
