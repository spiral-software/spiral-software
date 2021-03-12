
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Declare(TTensor3, TTensorI3);

Class(TTensor3, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (A, B, C) -> Checked(IsNonTerminal(A) and IsNonTerminal(B) and IsNonTerminal(C), [A,B,C]) ],
    dims := self >> let(a:=self.params[1].dims(), b:=self.params[2].dims(), c:=self.params[3].dims(), [a[1]*b[1]*c[1],a[2]*b[2]*c[2]]),
    terminate := self >> Tensor(self.params[1], self.params[2], self.params[3]),
    transpose := self >> TTensor3(self.params[1].transpose(), self.params[2].transpose(), self.params[3].transpose()).withTags(self.getTags()),
    isReal := self >> self.params[1].isReal() and self.params[2].isReal() and self.params[3].isReal(),
    normalizedArithCost := self >> self.params[1].normalizedArithCost() * Rows(self.params[2]) + self.params[2].normalizedArithCost() *  Cols(self.params[1]), # this is incorrect, but its okay i think
    HashId := self >> let(h := List(self.params, i->When(IsBound(i.HashId), i.HashId(), i)),
        When(IsBound(self.tags), Concatenation(h, self.tags), h))

#D    setpv := (self, pv) >> TTensor(self.params[1], self.params[2], pv),
#D    tagpos := 3,
));


Class(TTensorI3, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (t1, t2, t3) -> Checked(true, [t1, t2, t3])],
    dims := self >> self.params[1].dims()[2]*self.params[2].dims()[2]*self.params[3].dims(),
    SPLtSPL := (self) >> let(ll:=Length(self.params),
									Tensor(self.params[1],self.params[2],self.params[3])
									),
    terminate := self >> let(P := self.params,
						Tensor(P[1].terminate(),P[2].terminate(),P[3].terminate())),
    transpose := self >> TTensorI3(self.params[1].transpose(), self.params[2].transpose(), self.params[3].transpose()).withTags(self.getTags()),
    isReal := self >> self.params[1].isReal(),
    #normalizedArithCost := self >> self.params[1].normalizedArithCost() * self.params[2],

    doNotMeasure := true,
	
    #HashId := self >> let(h := [ When(IsBound(self.params[1].HashId), self.params[1].HashId(), self.params[1]), self.params[2], self.params[3], self.params[4] ],
    #    When(IsBound(self.tags), Concatenation(h, self.tags), h)),

#D    setpv := (self, pv) >> TTensorI(self.params[1], self.params[2], self.params[3], self.params[4], pv),
#D    tagpos := 5
));

IsIIA := (p) -> ((not IsNonTerminal(p[1])) and (not IsNonTerminal(p[2])) and (IsNonTerminal(p[3])));
IsIAI := (p) -> ((not IsNonTerminal(p[1])) and (IsNonTerminal(p[2])) and (not IsNonTerminal(p[3])));
IsAII := (p) -> ((IsNonTerminal(p[1])) and (not IsNonTerminal(p[2])) and (not IsNonTerminal(p[3])));

IsIAA := (p) -> ((not IsNonTerminal(p[1])) and (IsNonTerminal(p[2])) and (IsNonTerminal(p[3])));
IsAIA := (p) -> ((IsNonTerminal(p[1])) and (not IsNonTerminal(p[2])) and (IsNonTerminal(p[3])));
IsAAI := (p) -> ((IsNonTerminal(p[1])) and (IsNonTerminal(p[2])) and (not IsNonTerminal(p[3])));
