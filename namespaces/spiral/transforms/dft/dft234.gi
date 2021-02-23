
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RulesFor(DFT, rec(
    #F DFT_DFT1and3 : DFT1_2n -> L (DFT1_n dirsum DFT3_n) L (I2 tensor F2) L
    #F Derived using polynomial factorization:
    #F    x^2n -> (x^n-1) (x^n+1)
    #F
    DFT_DFT1and3 := rec(
        isApplicable := P -> P[1] > 2 and P[1] mod 2 = 0,
        allChildren := P -> [[ DFT1(P[1]/2, P[2]), DFT3(P[1]/2, P[2]) ]],
        rule := (P,C) -> let(n := P[1]/2, 
            L(2*n, n) * DirectSum(C[1], C[2]) * L(2*n, 2) * 
            Tensor(I(n), F(2)) *
            L(2*n, n))
    )
));

lperm := p -> Z(p, (p-1)/2); 
rperm := p -> Z(p, (p+1)/2);
pdiag := (p,k) -> When(IsEvenInt(k), 
    I(p),
    Diag(List([0..p-1], i -> (-1)^i)));

RulesFor(DFT2, rec(
    DFT2_CT := Inherit(DFT_CT, rec(
	switch := false,
	allChildren  := P -> Map2(DivisorPairs(P[1]), 
	    (m,n) -> [ DFT2(m, P[2]), DFT(n, P[2]) ]),
       
	rule := (P,C) -> let(mn := P[1], m := Rows(C[1]), n := Rows(C[2]), 
	    Tensor(C[1], I(n)) * Diag(fPrecompute(Tw2(mn, n, P[2]))) *
	    Tensor(I(m), C[2]) * L(mn, m))
    )),
));

NewRulesFor(DFT2, rec(
    DFT2_PF := rec(
    	info         := "DFT2_n*k -> diag * perm * (DFT2_n_a tensor DFT2_k_b) * perm",
    	forTransposition := true,
    	applicable     := nt -> nt.params[1] > 2 and DivisorPairsRP(nt.params[1]) <> [] and not nt.hasTags(),
    	allChildren := nt -> let(N := nt.params[1], k:= nt.params[2], Map2(DivisorPairsRP(N), 
    		(r,s) -> [ DFT2(r, 1/s mod r), DFT2(s, 1/r mod s) ])),
    	
        apply := (nt, C, cnt) -> let(
    	    N := nt.params[1],
    	    r := Rows(C[1]), 
    	    s := Rows(C[2]),
    	    alpha := 1 / s mod r,
    	    beta  := 1 / r mod s,
    	    i := Ind(r*s),
    
    	    CRT(r,s).transpose() * 
    	    Diag(Lambda(i, -sign(imod(s*alpha*idiv(i,s) + r*beta*imod(i,s), 2*N) - N))) *
    	    Tensor(C[1], C[2]) * CRT(r,s)
    	)
#D    	isApplicable     := P -> P[1] > 2 and DivisorPairsRP(P[1]) <> [] and Length(P[3]) = 0,
#D    	allChildren := P -> let(N := P[1], k:= P[2], Map2(DivisorPairsRP(N), 
#D    		(r,s) -> [ DFT2(r, 1/s mod r), DFT2(s, 1/r mod s) ])),
#D    	
#D    	rule := (P,C) -> let(
#D    	    N := P[1],
#D    	    r := Rows(C[1]), 
#D    	    s := Rows(C[2]),
#D    	    alpha := 1 / s mod r,
#D    	    beta  := 1 / r mod s,
#D    	    i := Ind(r*s),
#D    
#D    	    CRT(r,s).transpose() * 
#D    	    Diag(Lambda(i, -sign(imod(s*alpha*idiv(i,s) + r*beta*imod(i,s), 2*N) - N))) *
#D    	    Tensor(C[1], C[2]) * CRT(r,s)
#D    	)
    )
));

RulesFor(DFT3, rec(
    DFT3_Base := BaseRule(DFT3, [2, ...]),

    DFT3_CT := Inherit(DFT_CT, rec(
	switch := false,
	allChildren  := P -> Map2(DivisorPairs(P[1]), 
	    (m,n) -> [ DFT(m, P[2]), DFT3(n, P[2]) ]),
       
	rule := (P,C) -> let(mn := P[1], m := Rows(C[1]), n := Rows(C[2]), 
	    Tensor(C[1], I(n)) * Diag(fPrecompute(Tw3(mn, n, P[2]))) *
	    Tensor(I(m), C[2]) * L(mn, m))
    )),

    DFT3_CT_Radix2 := rec(
	isApplicable := P -> P[1] > 2 and (P[1] mod 2) = 0,
	allChildren  := P -> [[ DFT(P[1]/2, P[2]) ]],       
	rule := (P,C) -> let(n := P[1]/2,
	    Tensor(C[1], I(2)) * Diag(fPrecompute(Tw3(2*n, 2, P[2]))) *
	    Tensor(I(n), F(2)*Diag(1,E(4)^P[2])) * L(2*n, n))
    ),

    DFT3_OddToDFT1 := rec(
	isApplicable := P -> IsOddInt(P[1]),
	allChildren := P -> [[ DFT(P[1], P[2]) ]],
	rule := (P, C) -> rperm(P[1]) * C[1] * pdiag(P[1], P[2])
    ),

    DFT3_2xOddToDFT1 := rec(
	isApplicable := P -> P[1] mod 4 = 2 and P[1] > 2,
	allChildren := P -> [[ DFT(P[1], P[2]) ]],
	rule := (P, C) -> let(n:=P[1], k:=P[2],
	    Z(n,-(n-2)/4) * C[1] * Diag(List([0..n-1], x->E(4)^(k*x))))
    )

));

RulesFor(DFT4, rec(	
    DFT4_Base := rec(
	isApplicable := P -> P[1]=2,
	forTransposition := false,
	rule := (P,C) -> E(8)*Mat((1/E(8))*MatSPL(ApplyFunc(DFT4, P)))
    ),

    DFT4_CT := Inherit(DFT_CT, rec(
	allChildren  := P -> Map2(DivisorPairs(P[1]), 
	    (m,n) -> [ DFT2(m, P[2]), DFT3(n, P[2]) ]),
       
	rule := (P,C) -> let(mn := P[1], m := Rows(C[1]), n := Rows(C[2]), 
	    Tensor(C[1], I(n)) * Diag(fPrecompute(Tw4(mn, n, P[2]))) *
	    Tensor(I(m), C[2]) * L(mn, m))
    )),
));

NewRulesFor(DFT4, rec(
    DFT4_PF := rec(
    	info         := "DFT4_n*k -> diag * perm * (DFT4_n_a tensor DFT4_k_b) * perm",
    	forTransposition := true,
    	applicable     := nt -> nt.params[1] > 2 and DivisorPairsRP(nt.params[1]) <> [] and not nt.hasTags(),
        children := nt -> let(N := nt.params[1], k:= nt.params[2], Map2(DivisorPairsRP(N), 
    		(r,s) -> [ DFT4(r, 1/s mod r), DFT4(s, 1/r mod s) ])),
    	
        apply := (nt, C, cnt) -> let(
    	    N := nt.params[1],
    	    r := Rows(C[1]), 
    	    s := Rows(C[2]),
    	    alpha := 1 / s mod r,
    	    beta  := 1 / r mod s,
    	    i := Ind(r*s),
    	    d := Diag(Lambda(i, -sign(imod(s*alpha*idiv(i,s) + r*beta*imod(i,s), 2*N) - N))),
    	    CRT(r,s).transpose() * 
    	    d * (-E(4))*
    	    Tensor(C[1], C[2]) * 
    	    d *
    	    CRT(r,s)
    	)
#D    	isApplicable     := P -> P[1] > 2 and DivisorPairsRP(P[1]) <> [] and Length(P[3]) = 0,
#D    	allChildren := P -> let(N := P[1], k:= P[2], Map2(DivisorPairsRP(N), 
#D    		(r,s) -> [ DFT4(r, 1/s mod r), DFT4(s, 1/r mod s) ])),
#D    	
#D    	rule := (P,C) -> let(
#D    	    N := P[1],
#D    	    r := Rows(C[1]), 
#D    	    s := Rows(C[2]),
#D    	    alpha := 1 / s mod r,
#D    	    beta  := 1 / r mod s,
#D    	    i := Ind(r*s),
#D    	    d := Diag(Lambda(i, -sign(imod(s*alpha*idiv(i,s) + r*beta*imod(i,s), 2*N) - N))),
#D    	    CRT(r,s).transpose() * 
#D    	    d * (-E(4))*
#D    	    Tensor(C[1], C[2]) * 
#D    	    d *
#D    	    CRT(r,s)
#D    	)
    )
));

