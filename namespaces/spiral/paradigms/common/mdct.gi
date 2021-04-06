
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(TMDCT, TIMDCT);

_DCT_CONST := 2.5;


Class(TMDCT, TaggedNonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ] ,
    dims := self >> MDCT(self.params[1]).dims(),
    isReal := True,
    terminate := self >> MDCT(self.params[1]).terminate(),
    transpose := self >> TIMDCT(self.params[1]),
    SmallRandom := () -> Random([2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32]),
    normalizedArithCost := (self) >> MDCT(self.params[1]).normalizedArithCost()
));

Class(TIMDCT, TaggedNonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ] ,
    dims := self >> IMDCT(self.params[1]).dims(),
    isReal := True,
    terminate := self >> IMDCT(self.params[1]).terminate(),
    transpose := self >> TMDCT(self.params[1]),
    SmallRandom := () -> Random([2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32]),
    normalizedArithCost := (self) >> IMDCT(self.params[1]).normalizedArithCost()
));

NewRulesFor(TMDCT, rec(
    TMDCT_DCT4_tSPL := rec(
        switch := false,
        applicable := (self, t) >> true,
        children := (self, t) >> let(tags := t.getTags(), n := t.params[1], [[
                                    TDCT4(n).withTags(tags),
                                    TTensorI(RowVec([1,1,1]), n, AVec, AVec).withTags(tags),
                                    TTensorI(ColVec([0,1]), n/2, AVec, AVec).withTags(tags),
                                    TCompose([TTensorI(RowVec(-1), n, AVec, AVec), TPrm(J(n)), TTensorI(RowVec(1), n, AVec, AVec)]).withTags(tags),
                                    TTensorI(ColVec([-1,0]), n/2, AVec, AVec).withTags(tags)
                                 ]]),
        apply := (self, t, C, Nonterms) >> C[1] * C[2] * DirectSum(C[3], C[4], C[5])
    )
));

NewRulesFor(TIMDCT, rec(
    TIMDCT_TMDCT_tSPL := rec(
        switch := false,
        applicable := (self, t) >> true,
        children := (self, t) >> [[ TMDCT(t.params[1]).withTags(t.getTags()) ]],
        apply := (self, t, C, Nonterms) >> C[1].transpose()
    )
));
