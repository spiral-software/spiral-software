
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(PkRDFT12_Base, PRDFT_Base, rec(
    dims := self >> [self.params[1], self.params[1]],
    terminate := self >> let(N := self.params[1], k := self.params[2], 
	rr := When(self.transposed, Cols(self), Rows(self)), 
	mat := Cond(IsEvenInt(N), 
            Mat(Concatenation(
                    [List([0..N-1], c-> When(IsEvenInt(c), 1,1))],
                    [List([0..N-1], c-> When(IsEvenInt(c), 1,-1))],
                    List([2..rr-1], r -> When(r mod 2 = 0, 
                        List([0..N-1], c -> self.projRe(self.omega(N,k,Int(r/2),c))),
                        List([0..N-1], c -> self.projIm(self.omega(N,k,Int(r/2),c))))))),
            Mat(Concatenation(
                    [List([0..N-1], c -> 1)],
                    List([2..rr], r -> When(r mod 2 = 0, 
                        List([0..N-1], c -> self.projRe(self.omega(N,k,Int(r/2),c))),
                        List([0..N-1], c -> self.projIm(self.omega(N,k,Int(r/2),c)))))))),
        When(self.transposed, mat.transpose(), mat)),
));

Class(PkRDFT1, PkRDFT12_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosIntSym(n), IsIntSym(k), 
                                  IsSymbolic(n) or IsSymbolic(k) or Gcd(n,k)=1, [n, k mod n]) ],
    omega := (N,k,r,c) -> E(N)^(k*r*c)
));
Class(PkRDFT2, PkRDFT12_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosIntSym(n),  [n, 1]),
                 (n,k) -> Checked(IsPosIntSym(n), IsIntSym(k), 
                                  IsSymbolic(n) or IsSymbolic(k) or Gcd(n,k)=1, [n, k mod (2*n)]) ],
    omega := (N,k,r,c) -> E(2*N)^(k*r*(2*c+1))
));

Class(PkDHT1, PkRDFT1, rec(
    terminate := self >> let(n := self.params[1], k := self.params[2], i := When(IsEvenInt(n), 2, 1),
        DirectSum(I(i), Tensor(I(Int((n-1)/2)), F(2))) * 
        PkRDFT1(n, k).terminate()
    )
));

Class(PkDHT2, PkRDFT2, rec(
    terminate := self >> let(n := self.params[1], k := self.params[2], i := When(IsEvenInt(n), 2, 1),
        DirectSum(I(i), Tensor(I(Int((n-1)/2)), Diag(1,-1)*F(2))) * 
        PkRDFT2(n, k).terminate()
    )
));

Class(URealDFT_Base, NonTerminal, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod n]) ],
    terminate := self >> let(N := self.params[1], k := self.params[2], 
	rr := When(self.transposed, Cols(self), Rows(self)),
	Mat(Concatenation(
	    [List([0..N-1], c-> When(IsEvenInt(c), 1,0))],
	    [List([0..N-1], c-> When(IsOddInt(c), 1,0))],
	    List([2..rr-1], r -> When(r mod 2 = 0, 
		 List([0..N-1], c -> self.projRe(self.omega(N,k,Int(r/2),c))),
		 List([0..N-1], c -> self.projIm(self.omega(N,k,Int(r/2),c)))))))),
    isReal := True,
    SmallRandom := () -> Random([2..16]), 
    LargeRandom := () -> 2 ^ Random([6..15])
));

Class(URDFT_Base, URealDFT_Base, rec(
    projRe := Re,
    projIm := Im,
    hashAs := self >> self
));

Class(URDFT, URDFT_Base, rec(
    dims := self >> let(n:=self.params[1], [ 2*(Int((n+1)/2)), n ]),
    omega := (N,k,r,c) -> E(N)^(k*r*c),
));

URDFT1:=URDFT;

# Regularized Rules
# R1 -> (R1, C1) (R1')
# R2 -> (R2, C2) (R1')
PRF12_CTReg_Children := (N,k,PRFt,DFTt,PRF1prime, maxRadix) -> Map2(
    Filtered(DivisorPairs(N),d->d[1] <= maxRadix/2 and d[2]<>2),
    (m,n) -> When(IsEvenInt(n),
	[ PRFt(2*m,k), DFTt(m,k), PRF1prime(n,k) ],
	[ PRFt(m,k),   DFTt(m,k), PRF1prime(n,k) ] )
);

PRF12_CTReg_Rule := (N,k,C,Conj,Tw) -> let(mm:=Cols(C[1]), m:=When(IsEvenInt(N), mm/2, mm),
    n:=Cols(C[3]), Nf:=Int(N/2), Nc:=Int((N+1)/2),
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),

    SUM(
	BB(RC(Scat(H(Nc,m,0,n/2))) * C[1] * L(mm,2) * RC(Gath(H(nc*m, m, 0, nc)))),

	When(nc=1, [], 
	ISum(j, BB(
	     RC(Scat(BH(Nc,N,m,j+1,n))) *
	     Conj * RC(C[2]) * Tw(j) *
	     RC(Gath(H(nc*m, m, j+1, nc))))))
    ) * 
    Tensor(I(m), C[3]) * L(N,m)
);

PRF12_CTReg2_Rule := (N,k,C,Conj,Tw) -> let(mm:=Cols(C[1]), m:=When(IsEvenInt(N), mm/2, mm),
    n:=Cols(C[3]), Nf:=Int(N/2), Nc:=Int((N+1)/2),
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),

    SUM(
	RC(Scat(H(Nc,m,0,n/2))) * C[1] * RC(Gath(H(nc*m, m, 0, 1))),

	When(nc=1, [], 
	ISum(j, 
	     RC(Scat(BH(Nc,N,m,j+1,n))) *
	     Conj * RC(C[2]) * Tw(j) * L(2*m, m) * 
	     RC(Gath(H(nc*m, m, m*(j+1), 1)))))
    ) * 
    Tensor(C[3], I(m)) 
);

PRF12_CTReg3_Rule := (N,k,C,Conj,Tw) -> let(mm:=Cols(C[1]), m:=When(IsEvenInt(N), mm/2, mm),
    n:=Cols(C[3]), Nf:=Int(N/2), Nc:=Int((N+1)/2),
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),

    SUM(
	RC(Scat(H(Nc,m,0,n/2))) * C[1] * L(2*m, 2) * Gath(H(N, 2*m, 0, nc)),

	When(nc=1, [], 
	ISum(j, 
	     RC(Scat(BH(Nc,N,m,j+1,n))) *
	     Conj * RC(C[2]) * Tw(j) *
	     Gath(H(N, 2*m, j+1, nc))))
    ) * 
    Tensor(I(m), L(n,2)*C[3]) * L(N,m)
);

PRF12_CTReg4_Rule := (N,k,C,Conj,Tw) -> let(mm:=Cols(C[1]), m:=When(IsEvenInt(N), mm/2, mm),
    n:=Cols(C[3]), Nf:=Int(N/2), Nc:=Int((N+1)/2),
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),

    RC(Tensor(I(m/2), DirectSum(I(n/2+1), J(n/2-1)))) *
    RC(L(N/2, m)) * 
    DirectSum(
        C[1] * L(2*m, 2),
	IterDirectSum(j, RC(M(m,m/2)) * Conj * RC(C[2]) * Tw(j))
    ) * 
    L(N, nc) * 
    Tensor(I(m), L(n,2)*C[3]) * L(N,m)
);

PRF12_CTReg5_Rule := (N,k,C,Conj,Tw) -> let(mm:=Cols(C[1]), m:=When(IsEvenInt(N), mm/2, mm),
    n:=Cols(C[3]), Nf:=Int(N/2), Nc:=Int((N+1)/2),
    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nc-1),

    RC(Tensor(I(m/2), DirectSum(I(n/2+1), J(n/2-1)))) *
    RC(L(N/2, m)) * 
    DirectSum(
        C[1],
	IterDirectSum(j, RC(M(m,m/2)) * Conj * RC(C[2]) * Tw(j) * L(2*m, m))
    ) * 
    Tensor(C[3], I(m))
);

unrc := m -> let(r := Dimensions(m)[1], c := Dimensions(m)[2],
    MatSPL(Gath(H(r,r/2,0,2))) * m * MatSPL(Scat(H(c,c/2,0,2))));

RulesFor(URDFT1, rec(
    URDFT1_Base1 := BaseRule(URDFT1, [1, @]),
    URDFT1_Base2 := BaseRule(URDFT1, [2, @]),
    URDFT1_Base4 := BaseRule(URDFT1, [4, @]),

    URDFT1_CT := rec(
        maxRadix := 32,
	isApplicable := P -> not IsPrime(P[1]) and not (IsEvenInt(P[1]) and IsPrime(P[1]/2)),
	allChildren  := (self, P) >> PRF12_CTReg_Children(P[1], P[2], URDFT1, DFT1, URDFT1, self.maxRadix), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=When(IsEvenInt(N), Cols(C[1])/2, Cols(C[1])),
	    PRF12_CTReg_Rule(N, k, C, Diag(BHD(m,1,-1)), 
                j -> RC(DirectSum(I(1), Diag(fPrecompute(fCompose(Twid(N,m,k,0,0,j+1), fAdd(m, m-1, 1))))))))),

    # slightly different variant of same rule, 
    URDFT1_CT2 := rec(
	isApplicable := P -> not IsPrime(P[1]) and not (IsEvenInt(P[1]) and IsPrime(P[1]/2)),
	allChildren  := P -> PRF12_CTReg_Children(P[1], P[2], URDFT1, DFT1, URDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=When(IsEvenInt(N), Cols(C[1])/2, Cols(C[1])),
	    PRF12_CTReg2_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1))))))),

    URDFT1_CT3 := rec(
	isApplicable := P -> not IsPrime(P[1]) and not (IsEvenInt(P[1]) and IsPrime(P[1]/2)),
	allChildren  := P -> PRF12_CTReg_Children(P[1], P[2], URDFT1, DFT1, URDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=When(IsEvenInt(N), Cols(C[1])/2, Cols(C[1])),
	    PRF12_CTReg3_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1))))))),

    URDFT1_CT4 := rec(
	isApplicable := P -> not IsPrime(P[1]) and not (IsEvenInt(P[1]) and IsPrime(P[1]/2)),
	allChildren  := P -> PRF12_CTReg_Children(P[1], P[2], URDFT1, DFT1, URDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=When(IsEvenInt(N), Cols(C[1])/2, Cols(C[1])),
	    PRF12_CTReg4_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1))))))),

    URDFT1_CT5 := rec(
	isApplicable := P -> not IsPrime(P[1]) and not (IsEvenInt(P[1]) and IsPrime(P[1]/2)),
	allChildren  := P -> PRF12_CTReg_Children(P[1], P[2], URDFT1, DFT1, URDFT1), 
	rule := (P,C) -> let(N:=P[1], k:=P[2], m:=When(IsEvenInt(N), Cols(C[1])/2, Cols(C[1])),
	    PRF12_CTReg5_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1))))))),

));


