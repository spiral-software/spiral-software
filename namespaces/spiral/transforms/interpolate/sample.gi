
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_vlist := (l,t)->List(Flat(l), x->t.value(x));

Class(fNearestNeighbor, FuncClass, rec(
#    def := (N,n,base,stride) -> Checked(IntFloat(base + (n-1)*stride) < N, rec(N:=N, n:=n)),
    def := (N,n,base,stride) -> rec(N:=N, n:=n),
    domain := self >> self.params[2],
    range := self >> self.params[1],
    lambda := self >> let(h := var.fresh_t("h", TInt), Lambda(h, floor(self.params[3]+h*self.params[4]+0.5))),
    # NOTE. FF: that is a temporary solution to make this function work inside Lambda. need to talk to YSV how to do it right.
    t := TFunc
));

Class(fZeroPadMiddle, FuncClass, rec(
    def := (N,n) -> Checked(IsEvenInt(n), rec(N:=N, n:=n)),
    domain := self >> self.params[2],
    range := self >> self.params[1],
    lambda := self >> let(N:=self.params[1], n:=self.params[2], fStack(fAdd(N, n/2, 0), fAdd(N, n/2, N-n/2)))
));

Class(Downsample, TaggedNonTerminal, rec(
    abbrevs := [ (dsfunc) -> [dsfunc] ],
    dims      := self >> [self.params[1].domain(), self.params[1].range()],
    terminate := self >> Gath(self.params[1]),
    isReal    := True,
    normalizedArithCost := self >> 0,
    TType := T_Complex(T_Real(64)),
    doNotMeasure := true,
    HashId := self >> let(h := [ ObjId(self.params[1]), self.params[1].range() ],
        When(IsBound(self.tags), Concatenation(h, self.tags), h)),

));

Class(Upsample, TaggedNonTerminal, rec(
    abbrevs := [ (usfunc) -> [usfunc] ],
    dims      := self >> [self.params[1].range(), self.params[1].domain()],
    terminate := self >> Scat(self.params[1]),
    isReal    := True,
    normalizedArithCost := self >> 0,
    TType := T_Complex(T_Real(64))
));

NewRulesFor(Upsample, rec(
    Upsample_base := rec(
        applicable := (self, nt) >> not nt.hasTags(),
        apply := (nt, C, cnt) -> Mat(MatSPL(Scat(nt.params[1])))
    )
));


NewRulesFor(Downsample, rec(
    Downsample_base := rec(
        applicable := (self, nt) >> not nt.hasTags(),
        apply := (nt, C, cnt) -> NeedInterleavedComplex(ScatGath(fId(nt.params[1].domain()), nt.params[1])) 
    )
));


