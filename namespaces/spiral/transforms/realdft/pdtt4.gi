
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(PDTT_Base, TaggedNonTerminal, rec(
    abbrevs := [ n -> Checked(IsPosIntSym(n), [n]) ],
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    print := (self,i,is) >> Print(self.__name__, "(", PrintCS(self.params), ")", 
	When(self.transposed, Print(".transpose()")),
        When(self.tags<>[], Print(".withTags(", self.tags, ")"))),

    normalizedArithCost := self >> let(n := self.params[1],
	floor(2.5 * n * log(n) / log(2.0)) )
));

#
# Discrete Trigonometric Transforms (DTTs) via PRF
# 
Declare(PDST3, PDCT3);

Class(PDST4, PDTT_Base, rec(
    transpose := self >> Copy(self),
    terminate := self >> Mat(DST_IVunscaled(EvalScalar(self.params[1]))),
));

Class(PDCT4, PDTT_Base, rec(
    transpose := self >> Copy(self),
    terminate := self >> Mat(DCT_IVunscaled(EvalScalar(self.params[1]))),
));

Class(PDCT2, PDTT_Base, rec(
    terminate := self >> Mat(DCT_IIunscaled(EvalScalar(self.params[1]))), 
    transpose := self >> PDCT3(self.params[1])
));

Class(PDST2, PDTT_Base, rec(
    terminate := self >> Mat(DST_IIunscaled(EvalScalar(self.params[1]))), 
    transpose := self >> PDST3(self.params[1])
));

Class(PDCT3, PDTT_Base, rec(
    terminate := self >> Mat(DCT_IIIunscaled(EvalScalar(self.params[1]))), 
    transpose := self >> PDCT2(self.params[1])
));

Class(PDST3, PDTT_Base, rec(
    terminate := self >> Mat(DST_IIIunscaled(EvalScalar(self.params[1]))), 
    transpose := self >> PDST2(self.params[1])
));

RulesFor(PDST4, rec(
    PDST4_Base2 := DST4_Base, #BaseRule(PDST4, 2),
    PDST4_CT := rec(
	isApplicable     := P -> P[1] > 2, #not IsPrime(P),
	forTransposition := false,
	allChildren  := P -> let(N := P[1], Map2(DivisorPairs(2*N), (m,n) -> Cond(
		IsEvenInt(m) and IsEvenInt(n), [ PRDFT3(m,-1).transpose(), PRDFT3(n) ],
		#IsEvenInt(m) and IsEvenInt(n), [ PDHT3(m).transpose(), PDHT3(n) ],
		IsEvenInt(m) and IsOddInt(n),  [ PRDFT3(m,-1).transpose(), PRDFT3(n), PDST4(m/2) ],
		IsOddInt(m)  and IsEvenInt(n), [ PRDFT3(m,-1).transpose(), PRDFT3(n), PDST4(n/2) ],
		IsOddInt(m)  and IsOddInt(n),  Error("This can't happen")))),

	rule := (P,C) -> let(N := P[1], m := Rows(C[1]), n := Cols(C[2]),
	    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nf), i:=Ind(mf),    
	    # mult twid by -E(4) for RDFT, by 1/2 for DHT
	    t := ScaledTwid(2*N, mf, 1, 1/2, 1/2, j, -E(4)),
	    T := When(IsOddInt(m), Diag(diagDirsum(t,fConst(1,1))), Diag(t)), 
	    Et := Tensor(I(mf),Mat([[0,-1],[-1,0]])),
	    Ett := Tensor(I(mf),Mat([[0,-1],[1,0]])),
	    SUM(
		ISum(j, 
		    Scat(BH(N, 2*N-1, m, j, n)) * C[1] * #Ett *
		    RC(T) * 
		    #Et * # DHT only
		    RC(Gath(H(mc*nc, mc, j, nc)))
		),
		When(IsEvenInt(n), [], Scat(H(N,m/2,nf,n))*C[3]*Gath(H(2*mc*nc, m/2, 2*nf, 2*nc)))
	    ) *
	    SUM(
		ISum(i, RC(Scat(H(mc*nc, nc, i*nc, 1))) * C[2] * Gath(BH(N, 2*N-1, n, i, m))),
                # output is imaginary, but we output into real slot, because subseq transform
		# is jIR2, which we implement as IR2 * (-j)
		When(IsEvenInt(m), [], Scat(H(2*mc*nc, nc, 2*mf*nc, 2))*C[3]*Gath(H(N,n/2,mf,m)))
	    ))),

    PDST4_CT_SPL := rec(
	isApplicable     := P -> P[1] > 2 and ForAny(DivisorPairs(2*P[1]), d->IsEvenInt(d[1]) and IsEvenInt(d[2])), 
	forTransposition := false,

	allChildren  := P -> let(N := P[1], 
	    Map2(Filtered(DivisorPairs(2*N), d -> IsEvenInt(d[1]) and IsEvenInt(d[2])),
		(m,n) -> [ PRDFT3(m,-1).transpose(), PRDFT3(n) ])),

	rule := (P,C) -> let(N := P[1], m := Rows(C[1]), n := Cols(C[2]),
            TT := Diag(fPrecompute(diagMul(fConst(N/2, -E(4)), 
                        fCompose(dOmega(8 * N, 1), 
                            diagTensor(dLin(N/m, 2, 1, TInt), dLin(m/2, 2, 1, TInt)))))),

	    Prm(Refl(N, 2*N-1, N, L(m*n, m))) *
	    Tensor(I(n/2), C[1]) * 
            RC(TT) * 
            RC(L(m*n/4, n/2)) *
	    Tensor(I(m/2), C[2]) *
	    Prm(Refl(N, 2*N-1, N, L(m*n, m)))
	)
    )
));

RulesFor(PDCT4, rec(
    PDCT4_Base2 := rec(isApplicable:=P->P[1]=2, rule:=(P,C)->DCT4(2).terminate()),
    PDCT4_Base4 := rec(isApplicable:=P->P[1]=4, 
	               allChildren := P -> [[ DCT4(4) ]], 
		       rule:=(P,C)->C[1]),

    PDCT4_CT := rec(
	isApplicable     := P -> P[1] > 2, #not IsPrime(P),
	forTransposition := false,
	allChildren  := P -> let(N := P[1], Map2(DivisorPairs(2*N), (m,n) -> Cond(
		IsEvenInt(m) and IsEvenInt(n), [ PRDFT3(m,-1).transpose(), PRDFT3(n) ],
		#IsEvenInt(m) and IsEvenInt(n), [ PDHT3(m).transpose(), PDHT3(n) ],
		IsEvenInt(m) and IsOddInt(n),  [ PRDFT3(m,-1).transpose(), PRDFT3(n), PDCT4(m/2) ],
		IsOddInt(m)  and IsEvenInt(n), [ PRDFT3(m,-1).transpose(), PRDFT3(n), PDCT4(n/2) ],
		IsOddInt(m)  and IsOddInt(n),  Error("This can't happen")))),

	rule := (P,C) -> let(N := P[1], m := Rows(C[1]), n := Cols(C[2]),
	    nf:=Int(n/2), nc:=Int((n+1)/2), mf:=Int(m/2), mc:=Int((m+1)/2), j:=Ind(nf), i:=Ind(mf),  
	    t := fPrecompute(Twid(2*N, mf, 1, 1/2, 1/2, j)),
	    T := When(IsOddInt(m), Diag(diagDirsum(t,fConst(1,1))), Diag(t)), 
	    Et := Tensor(I(mf),Mat([[0,1],[1,0]])),
	    SUM(
		ISum(j, 
		    Scat(BH(N, 2*N-1, m, j, n)) * Diag(BHN(m)) * C[1] *
		    # mult twid by 1/2 for DHT
		    RC(T) * 
		    #Et * # DHT only
		    RC(Gath(H(mc*nc, mc, j, nc)))
		),
		When(IsEvenInt(n), [], Scat(H(N,m/2,nf,n))*C[3]*Gath(H(2*mc*nc, m/2, 2*nf, 2*nc)))
	    ) *
	    SUM(
		ISum(i, RC(Scat(H(mc*nc, nc, i*nc, 1))) * C[2] * Diag(BHN(n)) * Gath(BH(N, 2*N-1, n, i, m))),
		When(IsEvenInt(m), [], Scat(H(2*mc*nc, nc, 2*mf*nc, 2))*C[3]*Gath(H(N,n/2,mf,m)))
	    ))),
));

NewRulesFor(PDCT4, rec(
    PDCT4_CT_SPL := rec(
        libApplicable := t -> eq(imod(t.params[1], 2), 0),
	applicable := t -> IsSymbolic(t.params[1]) or (t.params[1] > 2 and (t.params[1] mod 4) = 0), 
        extraLeftTags := [],
	forTransposition := false,

        freedoms := t -> [ divisorsIntNonTriv(t.params[1]/2) ], 

	child  := (self, t, fr) >> let(N := t.params[1], f := fr[1], 
	    [ spiral.sym.ASP(-1).RDFT3(2*f).withTags(self.extraLeftTags).transpose(), 
	      spiral.sym.ASP.RDFT3(div(N,f)) ]),

	apply := (t,C,Nonterms) -> let(N := t.params[1], m := Rows(C[1]), n := Cols(C[2]),
            mh := Rows(C[1])/2, nh := Cols(C[2])/2,
            D := RC(Diag(fPrecompute(fCompose(dOmega(8 * N, 1), 
                            diagTensor(dLin(N/m, 2, 1, TInt), dLin(m/2, 2, 1, TInt)))))),

            Grp(Scat(Refl1(nh, m)) * 
                Tensor(I(nh), Diag(BHN(m)) * C[1]) * 
                D *
                Tensor(I(nh), Tr(2, mh))) * 
            #RC(Tr(mh, nh)) *
            Grp(Tr(mh, n) *
                Tensor(I(mh), C[2] * Diag(BHN(n))) *
                Gath(Refl1(mh, n)))
	)
    )
));

RulesFor(PDCT4, rec(
    # Variant better suited for vectorization
    PDCT4_CT_SPL_Vec := rec(
	isApplicable     := P -> P[1] > 2 and (P[1] mod 2) = 0, 
	forTransposition := false,

	allChildren  := P -> let(N := P[1], 
	    Map2(Filtered(DivisorPairs(2*N), d -> IsEvenInt(d[1]) and IsEvenInt(d[2])),
		(m,n) -> [ PRDFT3(m,-1).transpose(), PRDFT3(n) ])),

	rule := (P,C) -> let(N := P[1], m := Rows(C[1]), n := Cols(C[2]),
            TT := Diag(fPrecompute(fCompose(dOmega(8 * N, 1), 
                        diagTensor(dLin(N/m, 2, 1, TInt), dLin(m/2, 2, 1, TInt))))),

            Prm(condIJ(N, n/2)) * 
            Prm(L(N, m)) * 
	    Tensor(I(n/2), 
                   condM(m,m/2) * Diag(BHN(m)) * C[1]) * 
            RC(TT) * 
            Prm(L(N, n/2)) *
	    Tensor(I(m/2), 
                   L(n, 2) * C[2] * Diag(BHN(n)) * condK(n, 2)) *
            Prm(L(N, m/2)) * 
            Prm(condIJ(N, m/2))
	)
    )

));

testPDCT4 := function(n)
    local opts, r, s;
    opts := CopyFields(SpiralDefaults, rec(
            breakdownRules := rec(
                PDCT4 := [PDCT4_Base2, PDCT4_CT_SPL_Vec],
                PRDFT1 := [PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT],
                PRDFT3 := [PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT],
                DFT := [DFT_Base, DFT_CT]
            )));
    r := RandomRuleTree(PDCT4(n), opts);
    s := SumsRuleTree(r, opts);
    return [opts,r,s];
end;

    
NewRulesFor(PDCT2, rec(
    PDCT2_Base2 := rec(
	forTransposition := true,
        applicable := t -> t.params[1] = 2, 
        apply := (t, C, Nonterms) -> DCT2(2).terminate()
    ),

    PDCT2_Base4 := rec(
	forTransposition := true,
        applicable := t -> t.params[1] = 4, 
        apply := (t, C, Nonterms) -> 
            LIJ(4) * 
            DirectSum(
                Diag(FList(TReal, [ 1, 0.70710678118654757 ])) * F(2),
                Rot(cospi(13/8), sinpi(13/8)) * J(2)) * 
            (Tensor(I(2), F(2))) ^ LIJ(4)
    ),

    PDCT2_CT_SPL := rec(
	forTransposition := true,
        libApplicable := t -> eq(imod(t.params[1], 2), 0),
        extraLeftTags := [],

	applicable := t -> let(N:=t.params[1], 
            IsSymbolic(N) or 
            (N > 2 and IsEvenInt(N) and (N mod 4) = 0)),

	freedoms := t -> [ divisorsIntNonTriv(t.params[1]/2) ], 

        # N/2 = k*m
        child := (self, t, fr) >> let(N:=t.params[1], k:=fr[1], m:=N/2/k, 
            [ PDCT2(2*k).withTags(self.extraLeftTags), 
              spiral.sym.ASP(-1).RDFT3(2*k).transpose().withTags(self.extraLeftTags), 
              spiral.sym.ASP(+1).URDFT(2*m) ]), 

	apply := (t,C,Nonterms) -> let(N:=t.params[1], k:=Rows(C[1])/2, m:=Cols(C[3])/2,
            D := RC(Diag(fPrecompute(fCompose(dOmega(4 * N, 1), 
                        diagTensor(dLin(m-1, 1, 1, TInt), dLin(k, 2, 1, TInt)))))),
            Grp(
                Scat(Refl0_u(m, 2*k)) *  
                DirectSum(
                    C[1] * KK(k, 2), 
                    Tensor(I(m-1), Diag(BHN(2*k)) * C[2]) * D
                )) *
            Grp(
                RC(Tr(k, m)) *
                Tensor(I(k), C[3]) *
                Gath(Refl1(k, 2*m))
            )
	)
    )
));

NewRulesFor(PDST2, rec(
    PDST2_Base2 := rec(
        applicable := t -> t.params[1] = 2, 
        apply := (t, C, Nonterms) -> DST2(2).terminate()
    ),

    PDST2_CT_SPL := rec(
	forTransposition := true,
	applicable := t -> let(N:=t.params[1], N > 2 and IsEvenInt(N) and (N mod 4) = 0),
	freedoms := t -> [ DivisorsIntNonTriv(t.params[1]/2) ], 

        # N/2 = k*m
        child := (t, fr) -> let(N:=t.params, k:=fr[1], m:=N/2/k, 
                [ DST2(2*k), PRDFT3(2*k,-1).transpose(), URDFT(2*m)  ]), 

	apply := (t,C,Nonterms) -> let(N:=t.params[1], k:=Rows(C[1])/2, m:=Cols(C[3])/2,
            TT := Diag(fPrecompute(fCompose(dOmega(4 * N, 1), 
                        # ie multiply all twiddles by -E(4)
                        diagAdd(diagDirsum(fConst(k, 0), fConst(k*(m-1), 3*N)), 
                                diagTensor(dLin(m, 1, 0, TInt), dLin(k, 2, 1, TInt)))))),

            J(N) * 
            Kp(N, 2*k) * 
	    DirectSum(
                J(2*k) * C[1] * Diag(BHN(2*k)) * K(2*k, 2), 
                Tensor(I(m-1), J(2*k) * M(2*k, k) * C[2])
            ) * 
            RC(TT) * 
            RC(L(N/2, m)) *
	    Tensor(I(k), C[3] * Diag(BHN(2*m))) *
	    Prm(Refl(N, 2*N-1, N, L(2*N, 2*k)))
	)
    )

));

RulesFor(DCT5, rec(
  DCT5_Rader := rec(
	forTransposition := false,
	isApplicable     := P -> P > 2 and IsPrime(2*P-1),
	allChildren      := P -> [[ IPRDFT(P-1,-1), PRDFT(P-1,-1) ]],

	diag := N -> Sublist(
	    DFT_Rader.raderDiag(2*N-1,1,PrimitiveRootMod(2*N-1)), 
	    [1..Int((N-1)/2)]*2),

	# 3rd col with 0's is for PRDFT, not necessary for (non-packed) RDFT
	raderMid := (self, N) >> let(Fsize := 2*N-1, 
	             DirectSum(Mat([[1, 1, 0], [1, -1/(Fsize-1), 0], [0,0,0]]),
			       RC(Diag(FData(self.diag(N)))))),

	# NOTE, special case for even P-1
	rule := (self,P,C) >> let(N := P, 
	    #RealRR(N).transpose() * 
		
	    Gath(H(2*P-1, P, 0, 1)) *
	    RR(2*P-1).transpose() *
	    DirectSum(I(1), VStack(I(P-1), I(P-1))) *

	    DirectSum(I(1), C[1]) *
#		C[1]*Diag(1,1, Replicate(2*Int((P-2)/2), 2), 
#		               When(IsEvenInt(P-1), [1,1],[]))) * 
	    self.raderMid(N) *
	    DirectSum(I(1), C[2]) *

            # replace by RealRR
	    Gath(H(2*P-1, P, 0, 1)) *
	    RR(2*P-1) * DirectSum(I(1), VStack(I(P-1), J(P-1)))
    ))
));

#q:=ExpandSPL(DCT5(6))[1];
#s:=SPLRuleTree(q);
#me := MatSPL(s);
#them := remat(MatSPL(q.node));

#RDFT-11,  DCT5(6) 15(6/9 rots) + 19(12/7 irdft) + 17(12/5 rdft) = 30/21 = 51
#           DST5(5)
#           10(blk) 
# fftw 60/50

# RDFT-13
#  DCT5(7)  18(8/10=4/8+0/1+2/1 rots) + 18(14/4 irdft) + 18(14/4 rdft) = 36/18=54
#  DST5(6)  
#  12(blk)
#  fftw 76/34=110

# DFT-13
# fftw 176+68=244
# us = 54*2 (dct5) + 24(blk)*2 + dst5 
