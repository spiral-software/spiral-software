
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(TDST2, TDST3, TDST4);

_DST_CONST := 2.5;

#######################################################################################
#   tSPL DST rules

#F DST4(<n>) - Discrete Sine Transform, Type IV, non-terminal
#F Definition: (n x n)-matrix [ sin((k-1/2)*(l-1/2)*pi/n) | k,l = 1...n ]
#F Note:       DST4 is symmetric
#F Example:    DST4(8)
Class(TDST4, TaggedNonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ] ,
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >> Mat(DST_IVunscaled(self.params[1])),
    transpose := self >> Copy(self),
    SmallRandom := () -> Random([2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32]),
    normalizedArithCost := (self) >> let(n := self.params[1], IntDouble(_DST_CONST * n * d_log(n) / d_log(2)))
));


NewRulesFor(TDST4, rec(
    DST4_CT_tSPL := rec(
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
                    TTensorI(condM(m,m/2) * PRDFT3(m,-1).transpose(), n/2, AVec, AVec),
                    TTensorI(L(n, 2) * PRDFT3(n) * condK(n, 2), m/2, APar, AVec),
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
                * ConjDiag(RC(Diag(fPrecompute(diagMul(fConst(TComplex, N/2, -E(4)),
                        fCompose(dOmega(8 * N, 1),
                            diagTensor(dLin(N/m, 2, 1, TInt), dLin(m/2, 2, 1, TInt))))
                ))), L(N, m), L(N, n/2))
            )
            * C[3] * C[4]
        )
    )
));


#F TDST2(<n>) - Discrete Sine Transform, Type II, non-terminal
#F Definition: (n x n)-matrix [ sin(k*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F Note:       DST2 is the transpose of DST3
#F Example:    DST2(8)
Class(TDST2, TaggedNonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ] ,
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >> Mat(DST_IIunscaled(self.params[1])),
    transpose := self >> TDST3(self.params[1]),
    SmallRandom := () -> Random([2,3,4,5,6,8,9,10,12,15,16,18,24,27,30,32]),
    normalizedArithCost := (self) >> let(n := self.params[1], IntDouble(_DST_CONST * n * d_log(n) / d_log(2)))
));


NewRulesFor(TDST2, rec(
    DST2_DST4_tSPL := rec(
        switch := false,
        applicable := (self, t) >> true,
        children := (self, t) >> [[ TS(t.params[1]).withTags(t.getTags()), TDST4(t.params[1]).withTags(t.getTags()) ]],
        apply := (self, t, C, Nonterms) >> let(P := t.params[1],
            Diag(Concat(List([1..P-1], i->V(1.0)), [V(2.0)])) *
            C[1] *
            C[2] *
            Diag(List([0..P - 1], i -> 1/(2 * CosPi((2*i + 1)/(4*P)))))
        )
    ))
);



#F TDST3(<n>) - Discrete Sine Transform, Type III, non-terminal
#F Definition: (n x n)-matrix [ sin((k-1/2)*l*pi/n) | k,l = 1...n ]
#F Note:       DST3 is the transpose of DST2
#F Example:    DST3(8)
#F Scaled variant (not supported) is:
#F                [ a_l*sin(k*(l-1/2)*pi/n) | k,l = 1...n ]
#F            with  a_j = 1/sqrt(2) for j = n and = 1 else.
Class(TDST3, TaggedNonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ] ,
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >> Mat(DST_IIIunscaled(self.params[1])),
    transpose := self >> TDST2(self.params[1]),
    SmallRandom := () -> Random([2,3,4,5,6,8,9,10,12,15,16,18,24,27,30,32]),
    normalizedArithCost := (self) >> let(n := self.params[1], IntDouble(_DST_CONST * n * d_log(n) / d_log(2)))
));


NewRulesFor(TDST3, rec(
    DST3_DST4_tSPL := rec(
        switch := false,
        applicable := (self, t) >> true,
        children := (self, t) >> [[ TDST4(t.params[1]).withTags(t.getTags()), TS(t.params[1]).withTags(t.getTags()) ]],
        apply := (self, t, C, Nonterms) >> let(P := t.params[1],
            Diag(List([0..P - 1], i -> 1/(2 * CosPi((2*i + 1)/(4*P))))) *
            C[1] * C[2].transpose() *
            Diag(Concat(List([1..P-1], i->V(1.0)), [V(2.0)])))
    )
));
