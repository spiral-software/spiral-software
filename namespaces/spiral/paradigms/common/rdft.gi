
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_RDFT_CONST := 2.5;

#F TRDFT(<n>, <k>) - RDFT Nonterminal
Class(TRDFT, TaggedNonTerminal, rec(
    abbrevs := [ (n) -> Checked(IsPosIntSym(n),  [n, 1]),
                 (n,k) -> Checked(IsPosIntSym(n), IsIntSym(k), 
                             Cond( IsSymbolic(n) or IsSymbolic(k), [n, k],
                                    Checked(Gcd(n,k)=1, [n, k mod n]))) ],
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >> let(
	mat := PkRDFT1(self.params[1], self.params[2]).terminate(),
	When(self.transposed, mat.transpose(), mat)),

    SmallRandom := () -> Random([2,4,6,8,10,12,16,18,24,30,32]),
    normalizedArithCost := (self) >> let(n := self.params[1], floor(_RDFT_CONST * n * log(n) / log(2.0)))
));


_conjEven := (N, rot) -> let(
    m := Mat([[1,0],[0,-1]]),
    d1 := Diag(diagDirsum(fConst(TReal, 2, 1.0), fConst(TReal, N-2, 1/2))),
    m1 := DirectSum(   m, _SUM(I(N-2), Tensor(J((N-2)/2), m))),
    m2 := DirectSum(J(2), _SUM(I(N-2), Tensor(J((N-2)/2), -m))),
    i := I(N),
    d2 := RC(Diag(fPrecompute(diagDirsum(
		    fConst(TReal, 1, 1.0), 
		    diagMul(fConst(TComplex, N/2-1, omegapi(-1/2)), 
			    fCompose(dOmega(N, rot), fAdd(N/2, N/2-1, 1))))))),
    d1 * _HStack(m1, m2) * VStack(i, d2)
);


#F TConjEven(<n>) - extract conjugate even from RC(DFT) of half the size
Class(TConjEven, TaggedNonTerminal, rec(
    abbrevs := [ (n)      -> Checked(IsPosIntSym(n), [n, 1]),
                 (n, rot) -> Checked(IsPosIntSym(n), IsIntSym(rot), [n, rot]) ],

    dims := self >> [self.params[1], self.params[1]],
    isReal := True,

    terminate := self >> let(mat := _conjEven(self.params[1], self.params[2]),
	Cond(self.transposed, mat.transpose(), mat)),

    conjTranspose := self >> self.transpose(),
    SmallRandom := () -> [Random([2,4,6,8,10,12,16,18,24,30,32]), 1],
    doNotMeasure := true
));


#F  "FFT of Single Real Function"
#F  Numerical Recipes in C, Chapter 12, pp 512--514
NewRulesFor(TRDFT, rec(
    TRDFT_DFT_NR_tSPL_New := rec(
        switch := true,
	forTransposition := true,
        applicable := (self, t) >> IsEvenInt(t.params[1]), # and t.hasTags(),
        children := (self, t) >> let(N := t.params[1], rot := t.params[2], 
            [[
                TConjEven(N, rot).withTags(t.getTags()),
                TRC(DFT(N/2, t.params[2])).withTags(t.getTags())
            ]]),
        apply := (self, t, C, Nonterms) >> C[1] * C[2]
    ),

    TRDFT_DFT_NR_tSPL := rec(
        switch := true,
	forTransposition := true,
        useComplexCh := false,
        applicable := (self, t) >> IsEvenInt(t.params[1]), # and t.hasTags(),
        children := (self, t) >> let(N := t.params[1], rot := t.params[2], 
            cmpxtags := t.getTags(),
            realtags := List(cmpxtags, e -> Cond(e.kind()=spiral.paradigms.vector.AVecRegCx, spiral.paradigms.vector.AVecReg(e.params[1]), e)),
            [[
                TConjEven(N, rot).withTags(realtags),
                TRC(DFT(N/2, t.params[2])).withTags(Cond(self.useComplexCh, cmpxtags, realtags))
            ]]),
        apply := (self, t, C, Nonterms) >> C[1] * C[2]
    ),

    TRDFT_DFT_NR_tSPL_Cplx := CopyFields(~.TRDFT_DFT_NR_tSPL, rec(
        useComplexCh := true,
        applicable   := (self, t) >> IsEvenInt(t.params[1]) and t.hasTag(spiral.paradigms.vector.AVecRegCx),
    )),

   TRDFT_CT_tSPL_New := rec(
        applicable := t -> let(n:=t.params[1],  t.hasTags() and 
            not IsPrime(n) and not (IsEvenInt(n) and IsPrime(n/2))),

        freedoms := t -> let(N := t.params[1], v2 := t.firstTag().params[1].v, 
            # TRC(TDiag(f)) needs 2*v | f.domain(), not to mention TConj
            [ Filtered(DivisorsIntNonTriv(N), x->x>2 and N/x mod v2=0) ]),

        child := (self, t, fr) >> let(
            N := t.params[1],  k := t.params[2],
            n := fr[1],        m := N/n,         j := Ind(Int((n+1)/2)-1),
	    tags := t.getTags(),

            [ When(IsEvenInt(n), TRDFT(2*m, k).withTags(tags), 
                                 TRDFT(m,   k).withTags(tags)),
              When(IsEvenInt(n), TTensorI(URDFT(n, k), m, AVec, AVec).withTags(tags),
                                 TTensorI(PkRDFT1(n, k), m, AVec, AVec).withTags(tags)),
              InfoNt(j),
              TConj(TRC(DFT(m, k) * TDiag(fPrecompute(Twid(N,m,k,0,0,j+1)))), 
		    fId(2*m), L(2*m, m)).withTags(tags)
	    ]
        ),

	apply := (self, t, C, Nonterms) >> let(
            N  := t.params[1],           m  := Nonterms[2].params[2],  n  := N/m,
            Nf := Int(N/2),              mf := Int(m/2),               nf := Int(n/2), 
            Nc := Int((N+1)/2),          mc := Int((m+1)/2),           nc := Int((n+1)/2), 
            j  := Nonterms[3].params[1], k  := t.params[2],            tags := t.getTags(),

            RC(Tensor(I(m/2), DirectSum(I(n/2+1), J(n/2-1)))) *
            RC(L(N/2, m)) * 
            DirectSum(
                C[1],
                IDirSum(j, RC(LIJ(m)) * Diag(BHD(m,1,-1)) * C[4])
            ) * 
            C[2]
        )
    ),

    TRDFT_CT_tSPL := rec(
#        requiredFirstTag := [AVecReg, AVecRegCx],

        # rule TRDFT_CT_tSPL_Cplx sets useComplexCh to <true> and overrides some methods
        useComplexCh := false,
        
        applicable := t -> let(n:=t.params[1],  t.hasTags() and 
            not IsPrime(n) and not (IsEvenInt(n) and IsPrime(n/2))),

        freedoms := t -> let( N := t.params[1], v2 := 2*t.firstTag().params[1].v, 
            # TRC(TDiag(f)) needs 2*v | f.domain(), not to mention TConj
            [ Filtered(DivisorsIntNonTriv(N), x->x>2 and N/x mod v2=0) ]),

        child := (self, t, fr) >> let(
            N := t.params[1],  k := t.params[2],
            n := fr[1],        m := N/n,         j := Ind(Int((n+1)/2)-1),

            cmpxtags := t.getTags(),
            realtags := List(cmpxtags, e -> Cond(e.kind()=spiral.paradigms.vector.AVecRegCx, spiral.paradigms.vector.AVecReg(e.params[1]), e)),

            [ When(IsEvenInt(n), TRDFT(2*m, k).withTags(cmpxtags), 
                                 TRDFT(m,   k).withTags(cmpxtags)),
              When(IsEvenInt(n), TTensorI(URDFT(n, k), m, AVec, AVec).withTags(realtags),
                                 TTensorI(PkRDFT1(n, k), m, AVec, AVec).withTags(realtags)),
              InfoNt(j)
            ] :: 
            Cond( self.useComplexCh,
              [ TRC(DFT(m, k)).withTags(cmpxtags),
                TRC(TDiag(fPrecompute(Twid(N,m,k,0,0,j+1)))).withTags(cmpxtags),
                TL(2*m, m, 1, 1).withTags(realtags) ],
              [ TConj(TRC(DFT(m, k)), fId(2*m), L(2*m, m)).withTags(realtags),
                TConj(TRC(TDiag(fPrecompute(Twid(N,m,k,0,0,j+1)))), L(2*m, 2), L(2*m, m)).withTags(realtags)] )
        ),

	apply := (self, t, C, Nonterms) >> let(
            N  := t.params[1],           m  := Nonterms[2].params[2],  n  := N/m,
            Nf := Int(N/2),              mf := Int(m/2),               nf := Int(n/2), 
            Nc := Int((N+1)/2),          mc := Int((m+1)/2),           nc := Int((n+1)/2), 
            j  := Nonterms[3].params[1], k  := t.params[2],            tags := t.getTags(),

            dft:= Cond(self.useComplexCh, C[4]*C[5]*C[6], C[4]*C[5]),

            RC(Tensor(I(m/2), DirectSum(I(n/2+1), J(n/2-1)))) *
            RC(L(N/2, m)) * 
            DirectSum(
                C[1],
                IDirSum(j, RC(LIJ(m)) * Diag(BHD(m,1,-1)) * dft )
            ) * 
            C[2]
        )
    ),

    TRDFT_CT_tSPL_Cplx := CopyFields(~.TRDFT_CT_tSPL, rec(
        useComplexCh := true,
        applicable := t -> let(n:=t.params[1],  t.hasTag(spiral.paradigms.vector.AVecRegCx) and 
            not IsPrime(n) and not (IsEvenInt(n) and IsPrime(n/2))),
        
        freedoms := t -> let( N := t.params[1], v := t.firstTag().v, 
            # TRC(TDiag(f)) needs v/2 | f.domain()
            [ Filtered(DivisorsIntNonTriv(N), x->x>2 and N/x mod v=0) ]),
    )),

    TRDFT_By_Def := rec(
         forTransposition := false,
         applicable := (self, t) >> (t.firstTagIs(spiral.paradigms.vector.AVecReg) or 
                                     t.firstTagIs(spiral.paradigms.vector.AVecRegCx)) and 
                                     t.params[1]<=t.firstTag().isa.v,
         apply := (self, t, C, Nonterms) >> spiral.paradigms.vector.breakdown.VectorizedMatSPL(t.firstTag().isa, t),
    ),

    TRDFT_By_Def_tr := rec(
         forTransposition := false,
	 transposed := true,
         applicable := (self, t) >> (t.firstTagIs(spiral.paradigms.vector.AVecReg) or 
                                     t.firstTagIs(spiral.paradigms.vector.AVecRegCx)) and 
                                     t.params[1]<=t.firstTag().isa.v,
         apply := (self, t, C, Nonterms) >> spiral.paradigms.vector.breakdown.VectorizedMatSPL(t.firstTag().isa, t),
    ),

));

NewRulesFor(TConjEven, rec(
    TConjEven_base := rec(
        switch := true,
        applicable := (self, t) >> IsEvenInt(t.params[1]),
        apply := (self, t, C, Nonterms) >> _conjEven(t.params[1], t.params[2])
    )
));
