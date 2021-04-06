
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_RDFT_CONST := 2.5;

#F TRDFT2D([<m>, <n>], <k>)
Class(TRDFT2D, TaggedNonTerminal, rec(
    abbrevs := [
        P     -> Checked(IsList(P), Length(P) = 2, ForAll(P,i->IsPosInt(i) and IsEvenInt(i)), Product(P) > 1,
                [ RemoveOnes(P), 1]),
        (P,k) -> Checked(IsList(P), Length(P) = 2, ForAll(P,IsPosInt), IsInt(k), Product(P) > 1,
                        Gcd(Product(P), k)=1,
                [ RemoveOnes(P), k mod Product(P)])
    ],
    dims := self >> let(m := self.params[1][1], n := self.params[1][2], d := [n*(m+2), m*n], When(self.transposed, Reversed(d), d)),
    isReal := True,
    terminate := self >>
            Gath(fTensor(fAdd(self.params[1][1], (self.params[1][1]+2)/2, 0), fId(2*self.params[1][2]))) *
            RC(MDDFT(self.params[1], self.params[2]).terminate()) *
            Scat(fTensor(fId(Product(self.params[1])), fBase(2, 0))),
    normalizedArithCost := (self) >> let(n := Product(self.params[1]), IntDouble(_RDFT_CONST * n * d_log(n) / d_log(2))),
    compute := meth(self, data)
        local rows, cols, cxrows, m1, m2;

        rows := self.params[1][1];
        cols := self.params[1][2];
        cxrows := Rows(self)/cols;

        m1 := List(TransposedMat(List(TransposedMat(data), i->ComplexFFT(i){[1..cxrows/2]})), ComplexFFT);
        m2 := Flat(List(Flat(m1), i->[ReComplex(i), ImComplex(i)]));

        return List([0..cxrows-1], i->m2{[cols*i+1..cols*(i+1)]});
    end
));


#F TIRDFT2D([<m>, <n>], <k>)
Class(TIRDFT2D, TaggedNonTerminal, rec(
    abbrevs := [
        P     -> Checked(IsList(P), Length(P) = 2, ForAll(P,i->IsPosInt(i) and IsEvenInt(i)), Product(P) > 1,
                [ RemoveOnes(P), 1]),
        (P,k) -> Checked(IsList(P), Length(P) = 2, ForAll(P,IsPosInt), IsInt(k), Product(P) > 1,
                        Gcd(Product(P), k)=1,
                [ RemoveOnes(P), k mod Product(P)])
    ],
    dims := self >> let(m := self.params[1][1], n := self.params[1][2], d := [m*n, n*(m+2)], When(self.transposed, Reversed(d), d)),
    isReal := True,
    terminate := self >> let(cxrows := (self.params[1][1]+2)/2, rows := self.params[1][1], cols := self.params[1][2], k := self.params[2],
        Tensor(PRDFT(rows,k).inverse().terminate(), I(cols)) * Tensor(I(cxrows), L(2*cols, 2)*RC(DFT(cols, -k)))),
    normalizedArithCost := (self) >> let(n := Product(self.params[1]), IntDouble(_RDFT_CONST * n * d_log(n) / d_log(2)))
));


#F 2D RDFT Rule
NewRulesFor(TRDFT2D, rec(
    TRDFT2D_ColRow_tSPL := rec(
        switch := true,
        applicable := (self, t) >> true,
        children := (self, t) >> let(m := t.params[1][1], n:=t.params[1][2], k:= t.params[2], tags := t.getTags(),
            [[
                TCompose([
                    TTensorI(TCompose([TRC(DFT(n, k)), TPrm(L(2*n, n))]), (m+2)/2, APar, APar),
                    TTensorI(PRDFT1(m, k), n, AVec, AVec)
                ]).withTags(t.getTags())
            ]]),
        apply := (self, t, C, Nonterms) >> C[1]
    )
));


#F 2D IRDFT Rule
NewRulesFor(TIRDFT2D, rec(
    TIRDFT2D_RowCol_tSPL := rec(
        switch := true,
        applicable := (self, t) >> true,
        children := (self, t) >> let(m := t.params[1][1], n:=t.params[1][2], k:= t.params[2], tags := t.getTags(),
            [[
                TCompose([
                    TTensorI(PRDFT1(m, k).inverse(), n, AVec, AVec),
                    TTensorI(TCompose([TPrm(L(2*n, 2)), TRC(DFT(n, -k))]), (m+2)/2, APar, APar)
                ]).withTags(t.getTags())
            ]]),
        apply := (self, t, C, Nonterms) >> C[1]
    )
));
