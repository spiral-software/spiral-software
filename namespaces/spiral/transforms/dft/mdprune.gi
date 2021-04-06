
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(PrunedMDDFT, TaggedNonTerminal, rec(
    abbrevs := [
        (n,k,blk,pat) -> Checked(ForAll(n, IsPosIntSym), IsIntSym(k), IsPosIntSym(blk), ForAll(pat, IsList), 
            [_unwrap(n), _unwrap(k), _unwrap(blk), pat, ]),
        ],
    dims      := self >> [Product(self.params[1]), self.params[3]*Product(self.params[4], i->Length(i))],
    isReal    := self >> false,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := TComplex,
    terminate := self >> let(nlist := self.params[1],
                                blk := self.params[3], pat_md := self.params[4],
                                scat3d := Tensor(List(Zip2(pat_md, nlist), j->let(pat := j[1], size := j[2],
                                    Tensor(Mat(List(pat, i->BasisVec(size/blk, i))).transpose(), I(blk)).terminate()))),
                                dft3d := MDDFT(nlist, self.params[2]),
                                t := dft3d * scat3d,
                                t.terminate()
                            )
                        
));

Class(PrunedIMDDFT, TaggedNonTerminal, rec(
    abbrevs := [
        (n,k,blk,pat) -> Checked(ForAll(n, IsPosIntSym), IsIntSym(k), IsPosIntSym(blk), ForAll(pat, IsList), 
            [_unwrap(n), _unwrap(k), _unwrap(blk), pat, ]),
        ],
    dims      := self >> [ self.params[3]*Product(self.params[4], i->Length(i)), Product(self.params[1]) ],
    isReal    := self >> false,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := TComplex,
    terminate := self >> let(nlist := self.params[1],
                                blk := self.params[3], pat_md := self.params[4],
                                gath3d := Tensor(List(Zip2(pat_md, nlist), j->let(pat := j[1], size := j[2],
                                    Tensor(Mat(List(pat, i->BasisVec(size/blk, i))), I(blk)).terminate()))),
                                dft3d := MDDFT(nlist, self.params[2]),
                                t := gath3d * dft3d,
                                t.terminate()
                            )
                        
));





NewRulesFor(PrunedMDDFT, rec(
    PrunedMDDFT_Base := rec(
        info := "PrunedMDDFT -> PrunedDFT",
        applicable     := nt -> Length(nt.params[1])=1,
        children       := nt -> let(P := nt.params, tags := nt.getTags(), [[ PrunedDFT(P[1][1], P[2], P[3], P[4][1]).withTags(tags) ]]),
        apply          := (nt, C, Nonterms) -> C[1]
    ),
    PrunedMDDFT_RowCol := rec (
        info := "PrunedMDDFT_n -> PrunedMDDFT_n/d, PrunedMDDFT_d",
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            pats := nt.params[4],
            List([1..len-1],
            i -> [ PrunedMDDFT(dims{[1..i]}, nt.params[2], nt.params[3], pats{[1..i]}), 
                   PrunedMDDFT(dims{[i+1..len]}, nt.params[2], nt.params[3], pats{[i+1..len]}) ])),

        apply := (nt, C, Nonterms) -> let(
            n1 := Cols(Nonterms[1]),
            n2 := Rows(Nonterms[2]),
            Tensor(C[1], I(n2)) *
            Tensor(I(n1), C[2])
        )
    )
));

NewRulesFor(PrunedIMDDFT, rec(
    PrunedIMDDFT_Base := rec(
        info := "PrunedIMDDFT -> PrunedIDFT",
        applicable     := nt -> Length(nt.params[1])=1,
        children       := nt -> let(P := nt.params, tags := nt.getTags(), [[ PrunedIDFT(P[1][1], P[2], P[3], P[4][1]).withTags(tags) ]]),
        apply          := (nt, C, Nonterms) -> C[1]
    ),
    PrunedIMDDFT_RowCol := rec (
        info := "PrunedIMDDFT_n -> PrunedIMDDFT_n/d, PrunedIMDDFT_d",
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            pats := nt.params[4],
            List([1..len-1],
            i -> [ PrunedIMDDFT(dims{[1..i]}, nt.params[2], nt.params[3], pats{[1..i]}), 
                   PrunedIMDDFT(dims{[i+1..len]}, nt.params[2], nt.params[3], pats{[i+1..len]}) ])),

        apply := (nt, C, Nonterms) -> let(
            n1 := Cols(Nonterms[1]),
            n2 := Rows(Nonterms[2]),
            Tensor(C[1], I(n2)) *
            Tensor(I(n1), C[2])
        )
    )
));

