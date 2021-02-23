
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_DHT_CONST := 2.5;

#F TDHT(<n>, <k>) - RDFT Nonterminal
Class(TDHT, TaggedNonTerminal, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n]) ],
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >> PkDHT1(self.params[1]).terminate(),
    SmallRandom := () -> Random([2,4,6,8,10,12,16,18,24,30,32]),
    normalizedArithCost := (self) >> let(n := self.params[1], IntDouble(_DHT_CONST * n * d_log(n) / d_log(2)))
));


_DHTtoRDFT := N -> DirectSum(I(2), Tensor(I(N/2-1), F(2)));

#F TXMatDHT(<n>) - X-matrix to translate a RC(DFT(N)) -> DHT(2*N)
Class(TXMatDHT, TaggedNonTerminal, rec(
    abbrevs := [ (n) -> Checked(IsPosIntSym(n), [n]) ],
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,
    terminate := self >>  _DHTtoRDFT(self.params[1]) * TConjEven(self.params[1]).terminate(),
    SmallRandom := () -> Random([2,4,6,8,10,12,16,18,24,30,32]),
    doNotMeasure := true
));


#F  Translate DHT into RDFT
NewRulesFor(TDHT, rec(
    DHT_DFT_tSPL := rec(
        switch := true,
        applicable := (self, t) >> IsEvenInt(t.params[1]), # and t.hasTags(),
        children := (self, t) >> let(N := t.params[1], tags := t.getTags(),
            [[
                TXMatDHT(N).withTags(tags),
                TRC(DFT(N/2)).withTags(tags)
            ]]),
        apply := (self, t, C, Nonterms) >> C[1] * C[2]
    )
));


NewRulesFor(TXMatDHT, rec(
    TXMatDHT_TConjEven := rec(
        switch := true,
        applicable := (self, t) >> IsEvenInt(t.params[1]),
        children := (self, t) >> let(N := t.params[1], tags := t.getTags(),
            [[
                 TConjEven(N).withTags(tags),
           ]]),
        apply := (self, t, C, Nonterms) >> _DHTtoRDFT(t.params[1]) * C[1]
    )
));
