
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(TDCT2, TDCT3, TDCT4);

_DCT_CONST := 2.5;

#######################################################################################
#   tSPL DCT rules


#    PDCT4_CT_SPL := rec(
#    isApplicable     := P -> P > 2 and ForAny(DivisorPairs(2*P), d->IsEvenInt(d[1]) and IsEvenInt(d[2])),
#    forTransposition := false,
#
#    allChildren  := P -> let(N := P,
#        Map2(Filtered(DivisorPairs(2*N), d -> IsEvenInt(d[1]) and IsEvenInt(d[2])),
#        (m,n) -> [ PRDFT3(m,-1).transpose(), PRDFT3(n) ])),
#
#    rule := (P,C) -> let(N := P, m := Rows(C[1]), n := Cols(C[2]),
#        j := Ind(n/2),
#        T := Diag(Twid(2*N, m/2, 1, 1/2, 1/2, j)),
#
#        Prm(Refl(N, 2*N-1, N, L(m*n, m))) *
#        IterDirectSum(j, j.range, Diag(BHN(m)) * C[1] * RC(T)) *
#        RC(L(m*n/4, n/2)) *
#        Tensor(I(m/2), C[2] * Diag(BHN(n))) *
#        Prm(Refl(N, 2*N-1, N, L(m*n, m)))
#        )
#    )
#


#F TDCT4(<size>, tags) - Discrete Cosine Transform, Type IV, non-terminal
#F Definition: (n x n)-matrix  [ cos((k+1/2)*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F Note:       DCT4 is symmetric
#F Example:    DCT4(8)
Class(TDCT4, TaggedNonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ] ,
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >> Mat(DCT_IVunscaled(self.params[1])),
    transpose := self >> Copy(self),
    SmallRandom := () -> Random([2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32]),
    normalizedArithCost := (self) >> let(n := self.params[1], IntDouble(_DCT_CONST * n * d_log(n) / d_log(2)))
));


NewRulesFor(TDCT4, rec(
#    DCT4_CT_tSPL := rec(
#        switch := false,
#
#        applicable := (self, t) >> let(P:=t.params, P[1] > 2 and ForAny(DivisorPairs(2*P[1]), d->IsEvenInt(d[1]) and IsEvenInt(d[2])) and HasTags(t)),
#
#        children := (self, t) >> let(tags := GetTags(t), N := t.params[1],
#                                    Map2(Filtered(DivisorPairs(2*N), d -> IsEvenInt(d[1]) and IsEvenInt(d[2])),
#                                        (m,n) -> [
#                                            SetTag(TCompose([
#                                                TScat(Refl(N, 2*N-1, N, L(m*n, m))),
#                                                TTensorI(Diag(BHN(m)) * PRDFT3(m,-1).transpose(), n/2, APar, APar),
#                                                TRC(TDiag(fPrecompute(
#                                                    fCompose(dOmega(8 * N, 1), diagTensor(dLin(N/m, 2, 1, TInt), dLin(m/2, 2, 1, TInt)))
#                                                ))),
#                                                TRC(TL(m*n/4, n/2, 1, 1)),
#                                                TTensorI(PRDFT3(n) * Diag(BHN(n)), m/2, APar, APar),
#                                                TGath(Refl(N, 2*N-1, N, L(m*n, m))) ]),
#                                                tags)
#                                        ])),
#
#        apply := (self, t, C, Nonterms) >> C[1]
#    ),
#
#    # Variant better suited for vectorization
#    PDCT4_CT_SPL_Vec := rec(
#    isApplicable     := P -> P > 2 and (P mod 2) = 0,
#    forTransposition := false,
#
#    allChildren  := P -> let(N := P,
#        Map2(Filtered(DivisorPairs(2*N), d -> IsEvenInt(d[1]) and IsEvenInt(d[2])),
#        (m,n) -> [ PRDFT3(m,-1).transpose(), PRDFT3(n) ])),
#
#    rule := (P,C) -> let(N := P, m := Rows(C[1]), n := Cols(C[2]),
#            TT := Diag(fPrecompute(fCompose(dOmega(8 * N, 1),
#                        diagTensor(dLin(N/m, 2, 1, TInt), dLin(m/2, 2, 1, TInt))))),
#
#            IJ(N, n/2) *
#            L(N, m) *
#        Tensor(I(n/2),
#                   M(m,m/2) * Diag(BHN(m)) * C[1]) *
#            RC(TT) *
#            L(N, n/2) *
#        Tensor(I(m/2),
#                   L(n, 2) * C[2] * Diag(BHN(n)) * K(n, 2)) *
#            L(N, m/2) *
#            IJ(N, m/2)
#    )
#    )


    DCT4_CT_tSPL := rec(
        switch := false,

        applicable := (self, t) >> let(
            P:=t.params,
            P[1] > 2
            and ForAny(DivisorPairs(2*P[1]), d ->
                IsEvenInt(d[1]) and IsEvenInt(d[2])
            )
            and t.hasTags()
        ),

        children := (self, t) >> let(
            tags := t.getTags(),
            N := t.params[1],
            Map2(
                Filtered(DivisorPairs(2*N), d -> IsEvenInt(d[1]) and IsEvenInt(d[2])),
                (m,n) -> List([
                    TPrm(IJ(N, n/2)),
                    TTensorI(condM(m,m/2) * Diag(BHN(m)) * PRDFT3(m,-1).transpose(), n/2, AVec, AVec),
                    TTensorI(L(n, 2) * PRDFT3(n) * Diag(BHN(n)) * condK(n, 2), m/2, APar, AVec),
                    TPrm(IJ(N, m/2))
                ], i -> i.setTags(tags))
            )
        ),

        apply := (self, t, C, Nonterms) >> let(
            N:=t.params[1],
            n:=2*Rows(Nonterms[1].params[1].params[2]),
            m:=2*Rows(Nonterms[4].params[1].params[2]),
            Grp(
                C[1] * C[2]
                * ConjDiag(RC(Diag(fPrecompute(
                    fCompose(dOmega(8 * N, 1), diagTensor(dLin(N/m, 2, 1, TInt), dLin(m/2, 2, 1, TInt)))
                ))), L(N, m), L(N, n/2))
            )
            * C[3] * C[4]
        )
    )
));


#F TDCT2(<n>) - Discrete Cosine Transform, Type II, non-terminal
#F Definition: (n x n)-matrix [ cos(k*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F Note:       DCT2 is the transpose of DCT3
#F Example:    DCT2(8)
Class(TDCT2, TaggedNonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ] ,
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >> Mat(DCT_IIunscaled(self.params[1])),
    transpose := self >> TDCT3(self.params[1]),
    SmallRandom := () -> Random([2,3,4,5,6,8,9,10,12,15,16,18,24,27,30,32]),
    normalizedArithCost := (self) >> let(n := self.params[1], IntDouble(_DCT_CONST * n * d_log(n) / d_log(2)))
));


NewRulesFor(TDCT2, rec(
    DCT2_DCT4_tSPL := rec(
        switch := false,
        applicable := (self, t) >> true,
        children := (self, t) >> [[ TS(t.params[1]).withTags(t.getTags()), TDCT4(t.params[1]).withTags(t.getTags()) ]],
        apply := (self, t, C, Nonterms) >> let(P := t.params[1],
            Diag(Concat([V(2.0)], List([1..P-1], i->V(1.0)))) *
            C[1].transpose() * C[2] *
            Diag(List([0..P - 1], i -> 1/(2 * CosPi((2*i + 1)/(4*P))))))
    ))
);



#F TDCT3(<n>) - Discrete Cosine Transform, Type III, non-terminal (unscaled)
#F Definition: (n x n)-matrix [ cos((k+1/2)*l*pi/n) | k,l = 0...n-1 ]
#F  [scaled]   (n x n)-matrix [ a_l*cos((k+1/2)*l*pi/n) | k,l = 0...n-1 ]
#F                              a_j = 1/sqrt(2) for j = 0 and = 1 else
#F Note:       DCT3 is the transpose of DCT2, scaled NOT supported yet
#F Example:    DCT3(8)
Class(TDCT3, TaggedNonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ] ,
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >> Mat(DCT_IIIunscaled(self.params[1])),
    transpose := self >> TDCT2(self.params[1]),
    SmallRandom := () -> Random([2,3,4,5,6,8,9,10,12,15,16,18,24,27,30,32]),
    normalizedArithCost := (self) >> let(n := self.params[1], IntDouble(_DCT_CONST * n * d_log(n) / d_log(2)))
));


NewRulesFor(TDCT3, rec(
    DCT3_DCT2_tSPL := rec(
        switch := false,
        applicable := (self, t) >> true,
        children := (self, t) >> [[ TDCT2(t.params[1]).withTags(t.getTags()) ]],
        apply := (self, t, C, Nonterms) >> C[1].transpose()
    ),
    DCT3_DCT4_tSPL := rec(
        switch := false,
        applicable := (self, t) >> true,
        children := (self, t) >> [[ TDCT4(t.params[1]).withTags(t.getTags()), TS(t.params[1]).withTags(t.getTags()) ]],
        apply := (self, t, C, Nonterms) >> let(P := t.params[1],
            Diag(List([0..P - 1], i -> 1/(2 * CosPi((2*i + 1)/(4*P))))) *
            C[1] * C[2] *
            Diag(Concat([V(2.0)], List([1..P-1], i->V(1.0)))))
    )
));
