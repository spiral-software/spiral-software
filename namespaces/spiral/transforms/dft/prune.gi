
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(PrunedDFT, TaggedNonTerminal, rec(
    abbrevs := [
        (n,blk,pat) -> Checked(IsPosIntSym(n), IsPosIntSym(blk), IsList(pat),
            AnySyms(n,blk) or (IsInt(_unwrap(n)/_unwrap(blk)) and ForAll(pat, i->IsInt(i) and i < n/blk)),
            [_unwrap(n), 1, _unwrap(blk), pat]),
        (n,k,blk,pat) -> Checked(IsPosIntSym(n), IsIntSym(k), IsPosIntSym(blk), IsList(pat),
            AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
            AnySyms(n,blk) or (IsInt(_unwrap(n)/_unwrap(blk)) and ForAll(pat, i->IsInt(i) and i < n/blk)),
            [_unwrap(n), When(AnySyms(n,k), k, k mod _unwrap(n)), _unwrap(blk), pat])
        ],

    dims := self >> let(
        size := self.params[1],
        d := [size, self.params[3]*Length(self.params[4])],
        When(self.transposed, [d[2], d[1]], d)
    ),

    terminate := self >> let(
        size := self.params[1], blk := self.params[3], pat := self.params[4],
        res := DFT(size, self.params[2]).terminate() *
               Tensor(Mat(List(pat, i->BasisVec(size/blk, i))).transpose(), I(blk)).terminate(),
        When(self.transposed, res.transpose(), res)
    ),

    isReal    := self >> false,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := T_Complex(T_Real(64))
));



Class(PrunedIDFT, TaggedNonTerminal, rec(
    abbrevs := [
        (n,blk,pat) -> Checked(IsPosIntSym(n), IsPosIntSym(blk), IsList(pat),
            AnySyms(n,blk) or (IsInt(_unwrap(n)/_unwrap(blk)) and ForAll(pat, i->IsInt(i) and i < n/blk)),
            [_unwrap(n), 1, _unwrap(blk), pat]),
        (n,k,blk,pat) -> Checked(IsPosIntSym(n), IsIntSym(k), IsPosIntSym(blk), IsList(pat),
            AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
            AnySyms(n,blk) or (IsInt(_unwrap(n)/_unwrap(blk)) and ForAll(pat, i->IsInt(i) and i < n/blk)),
            [_unwrap(n), When(AnySyms(n,k), k, k mod _unwrap(n)), _unwrap(blk), pat])
        ],

    dims := self >> let(
        size := self.params[1],
        d := [size, self.params[3]*Length(self.params[4])],
        When(not self.transposed, [d[2], d[1]], d)
    ),

    terminate := self >> let(
        size := self.params[1], blk := self.params[3], pat := self.params[4],
        res := DFT(size, self.params[2]).terminate() *
               Tensor(Mat(List(pat, i->BasisVec(size/blk, i))).transpose(), I(blk)).terminate(),
        When(not self.transposed, res.transpose(), res)
    ),

    isReal    := self >> false,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := T_Complex(T_Real(64))
));


_pruned_children := function(m, n, scatpat, pdft)
    local pdfts, scatpats, iterparts, uspats, uuspats, uspats, spats;
    spats := Map([0..m-1], i->List(Intersection(scatpat, i+m*[0..n-1]), j->j));
    uspats := Map([0..Length(spats)-1], i->(spats[i+1]-i)/m);
    uuspats := Set(uspats);

    # partition iterations into unique patterns
    iterparts := List([1..Length(uuspats)], j->Filtered([1..m], i->uspats[i] = uuspats[j])-1);
    Sort(iterparts);

    scatpats := List(iterparts, lst->uspats[lst[1]+1]);
    pdfts := Map(scatpats, sp->pdft(n, sp));
    return pdfts;
end;

_ctpr_applicable := function(m, n, scatpat)
    local spats, gpats, strides;
    spats := Map([0..m-1], i->List(Intersection(scatpat, i+m*[0..n-1]), j->j));
    gpats := Map(spats, lst->Map(lst, i->Position(scatpat, i)-1));
    strides := Set(Flat(Map(gpats, lst->Map([1..Length(lst)-1], j->lst[j+1]-lst[j]))));
    return ForAll(spats, i->i<>[]) and  (Length(strides) in [0, 1]);
end;

_build_basef := lst -> When(lst = lst[1] + [0..Length(lst)-1], let(i := Ind(Length(lst)), Lambda(i, lst[1] + i)), FData(Map(lst, V)));

_build_stack := function(m, n, scatpat, pdfts, NI, gsop, istck, stck, cmpse)
    local stack, isu, is, itervars, gatscats, lbds, lbdvars, itercounts, iterbasesf, iterbases, iterparts, uuspats, uspats, bases, strides, stride, spats, gpats;
    spats := Map([0..m-1], i->List(Intersection(scatpat, i+m*[0..n-1]), j->j));
    gpats := Map(spats, lst->Map(lst, i->Position(scatpat, i)-1));
    #When(ForAny(spats, i->i=[]), Error("need at least one element per child DFT"));
    strides := Set(Flat(Map(gpats, lst->Map([1..Length(lst)-1], j->lst[j+1]-lst[j]))));
    stride := When(Length(strides) = 1, strides[1], 1);
    bases := Map(gpats, i->i[1]);

    uspats := Map([0..Length(spats)-1], i->(spats[i+1]-i)/m);
    uuspats := Set(uspats);

    iterparts := List([1..Length(uuspats)], j->Filtered([1..m], i->uspats[i] = uuspats[j])-1);
    Sort(iterparts);
    iterbases := Map(iterparts, lst->Map(lst, i->bases[i+1]));
    iterbasesf := List(iterbases, _build_basef);
    itercounts := List(iterparts, Length);
    itervars := List(iterparts, lst->Ind(Length(lst)));
    lbdvars := List(iterparts, lst->Ind(Length(gpats[lst[1]+1])));
    lbds := List([1..Length(itervars)], i->Lambda(lbdvars[i], stride*lbdvars[i]+iterbasesf[i].at(itervars[i])).setRange(NI));
    gatscats := List(lbds, i->ApplyFunc(gsop, [i]));

    is := List([1..Length(itervars)], i->ApplyFunc(istck, [itervars[i], cmpse(pdfts[i], gatscats[i])]));
    isu := Map([1..Length(is)], i->When(itervars[i].range=1, is[i].unroll().child(1), is[i]));
    stack := When(Length(isu) = 1, isu[1], ApplyFunc(stck, isu));
    return stack;
end;


NewRulesFor(PrunedDFT, rec(
    PrunedDFT_base := rec(
       forTransposition := false,
       maxSize := 256,
       applicable := (self, nt) >> not nt.hasTags() and nt.params[1] <= self.maxSize and nt.params[3] = 1,
       children := nt -> [[DFT(nt.params[1], nt.params[2])]],
       apply := (nt, C, cnt) -> let(size := nt.params[1], blk := nt.params[3], pat := nt.params[4],
            C[1] * Tensor(Mat(List(pat, i->BasisVec(size/blk, i))).transpose(), I(blk)).terminate())
    ),
    PrunedDFT_DFT := rec(
       forTransposition := true,
       applicable := (self, nt) >> nt.params[1] = nt.params[3] and nt.params[4] = [0],
       children := nt -> [[ DFT(nt.params[1], nt.params[2]).withTags(nt.getTags()) ]],
       apply := (nt, C, cnt) -> C[1]
    ),
    PrunedDFT_CT := rec(
       forTransposition := true,
       applicable := (self, nt) >> nt.params[1] > 2
            and not nt.hasTags()
            and not IsPrime(nt.params[1])
            and nt.params[3] > 1,
        children  := nt -> Map2(Filtered(DivisorPairs(nt.params[1]), (l) -> IsInt(nt.params[3]/l[1])),
            (m,n) -> [ DFT(m, nt.params[2] mod m), PrunedDFT(n, nt.params[2] mod n, nt.params[3]/m, nt.params[4]) ]
        ),
        apply := (nt, C, cnt) -> let(mn := nt.params[1], m := Rows(C[1]), n := Rows(C[2]),
            Tensor(C[1], I(n)) *
            Diag(fPrecompute(Tw1(mn, n, nt.params[2]))) *
            L(mn, m) * Tensor(C[2], I(m))
        )),
    PrunedDFT_CT_rec_block := rec(
       forTransposition := true,
       applicable := (self, nt) >> nt.params[1] > 2
            and not nt.hasTags()
            and not IsPrime(nt.params[1])
            and nt.params[3] = 1
            and Last(nt.params[4])+1-nt.params[4][1] = Length(Set(nt.params[4]))
            and ForAny(Map2(DivisorPairs(nt.params[1]), (m,n)->_ctpr_applicable(m, n, nt.params[4])), i->i),
       children := (nt) -> Filtered(Map2(DivisorPairs(nt.params[1]), (m, n) -> [ DFT(m, nt.params[2] mod m) ]::
                              _pruned_children(m, n, nt.params[4], (r, sp) -> PrunedDFT(r, nt.params[2] mod n, nt.params[3], sp))),
                              lst -> ForAll(Filtered(lst, i->ObjId(i) = PrunedDFT), j->j.params[4] <> [])),
        apply := (nt, C, cnt) -> let(mn := nt.params[1], m := Rows(C[1]), n := Rows(C[2]),
            Tensor(C[1], I(n)) *
            Diag(fPrecompute(Tw1(mn, n, nt.params[2]))) *
            _build_stack(m, n, nt.params[4], Drop(C, 1), nt.dims()[2], Gath, IterVStack, VStack, (a,b)->a*b)
        )
    )
));

NewRulesFor(PrunedIDFT, rec(
    PrunedIDFT_base := rec(
       forTransposition := false,
       maxSize :=256,
       applicable := (self, nt) >> not nt.hasTags() and nt.params[1] <= self.maxSize and nt.params[3] = 1,
       children := nt -> [[DFT(nt.params[1], nt.params[2])]],
       apply := (nt, C, cnt) -> let(size := nt.params[1], blk := nt.params[3], pat := nt.params[4],
            Tensor(Mat(List(pat, i->BasisVec(size/blk, i))) * C[1], I(blk)).terminate())
    ),
    PrunedIDFT_CT_rec_block := rec(
       forTransposition := true,
       applicable := (self, nt) >> nt.params[1] > 2
            and not nt.hasTags()
            and not IsPrime(nt.params[1])
            and nt.params[3] = 1
            and Last(nt.params[4])+1-nt.params[4][1] = Length(Set(nt.params[4]))
            and ForAny(Map2(DivisorPairs(nt.params[1]), (m,n)->_ctpr_applicable(m, n, nt.params[4])), i->i),
       children := (nt) -> Filtered(Map2(DivisorPairs(nt.params[1]), (n, m) ->
                              _pruned_children(m, n, nt.params[4], (r, sp) -> PrunedIDFT(r, nt.params[2] mod n, nt.params[3], sp)) ::
                              [ DFT(m, nt.params[2] mod m) ]),
                              lst -> ForAll(Filtered(lst, i->ObjId(i) = PrunedIDFT), j->j.params[4] <> [])),
        apply := (nt, C, cnt) -> let(mn := nt.params[1], m := Rows(Last(C)), n := Cols(C[1]),
            _build_stack(m, n, nt.params[4], DropLast(C, 1), nt.dims()[1], Scat, IterHStack1, HStack1, (a,b)->b*a) *
            Diag(fPrecompute(Tw1(mn, n, nt.params[2]))) *
            Tensor(Last(C), I(n))
        )
    )
));



Class(IOPrunedDFT, TaggedNonTerminal, rec(
    abbrevs := [
        (n,oblk,opat,iblk,ipat) -> Checked(IsPosIntSym(n), IsPosIntSym(oblk), IsList(opat), IsPosIntSym(iblk), IsList(ipat),
            AnySyms(n,iblk,oblk) or (IsInt(_unwrap(n)/_unwrap(iblk)) and IsInt(_unwrap(n)/_unwrap(oblk)) and
                ForAll(ipat, i->IsInt(i) and i < n/iblk) and ForAll(opat, i->IsInt(i) and i < n/oblk)),
            [_unwrap(n), 1, _unwrap(oblk), opat, _unwrap(iblk), ipat]),


        (n,k,oblk,opat,iblk,ipat) -> Checked(IsPosIntSym(n), IsIntSym(k), IsPosIntSym(oblk), IsList(opat), IsPosIntSym(iblk), IsList(ipat),
            AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
            AnySyms(n,iblk,oblk) or (IsInt(_unwrap(n)/_unwrap(iblk)) and IsInt(_unwrap(n)/_unwrap(oblk)) and
                ForAll(ipat, i->IsInt(i) and i < n/iblk) and ForAll(opat, i->IsInt(i) and i < n/oblk)),
            [_unwrap(n), When(AnySyms(n,k), k, k mod _unwrap(n)), _unwrap(oblk), opat, _unwrap(iblk), ipat])
        ],
    dims      := self >> [self.params[3]*Length(self.params[4]), self.params[5]*Length(self.params[6])],
    terminate := self >> let(size := self.params[1],
        oblk := self.params[3], opat := self.params[4], iblk := self.params[5], ipat := self.params[6],
        Tensor(Mat(List(opat, i->BasisVec(size/oblk, i))), I(oblk)).terminate() *
        DFT(size, self.params[2]).terminate() *
        Tensor(Mat(List(ipat, i->BasisVec(size/iblk, i))).transpose(), I(iblk)).terminate()),
    isReal    := self >> false,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := T_Complex(T_Real(64))
));


NewRulesFor(IOPrunedDFT, rec(
    IOPrunedDFT_base := rec(
       forTransposition := true,
       maxSize := 16,
       applicable := (self, nt) >> not nt.hasTags() and nt.params[1] <= self.maxSize and nt.params[3] = 1 and nt.params[5] = 1,
       children := nt -> [[DFT(nt.params[1], nt.params[2])]],
       apply := (nt, C, cnt) -> let(size := nt.params[1], oblk := nt.params[3], opat := nt.params[4], iblk := nt.params[5], ipat := nt.params[6],
            Tensor(Mat(List(opat, i->BasisVec(size/oblk, i))), I(oblk)).terminate() *
            C[1] *
            Tensor(Mat(List(ipat, i->BasisVec(size/iblk, i))).transpose(), I(iblk)).terminate())
    ),

    IOPrunedDFT__PrunedDFT := rec(
       forTransposition := true,
       applicable := (self, nt) >> nt.params[1] = nt.params[3] and nt.params[4] = [0],
       children := nt -> [[ PrunedDFT(nt.params[1], nt.params[2], nt.params[5], nt.params[6]).withTags(nt.getTags()) ]],
       apply := (nt, C, cnt) -> C[1]
    ),

    IOPrunedDFT__PrunedDFT_T := rec(
       forTransposition := true,
       applicable := (self, nt) >> nt.params[1] = nt.params[5] and nt.params[6] = [0],
       children := nt -> [[ PrunedDFT(nt.params[1], nt.params[2], nt.params[3], nt.params[4]).transpose().withTags(nt.getTags()) ]],
       apply := (nt, C, cnt) -> C[1]
    ),

    IOPrunedDFT__Gath_PrunedDFT := rec(
       forTransposition := true,
       maxSize := 16,
       applicable := (self, nt) >> not nt.hasTags() and nt.params[1] <= self.maxSize and
           ((nt.params[3] = 1 and not (nt.params[5] = 1)) or (nt.params[3] * nt.params[5] < nt.params[1])),
       children := nt -> [[ PrunedDFT(nt.params[1], nt.params[2], nt.params[5], nt.params[6]) ]],
       apply := (nt, C, cnt) -> let(size := nt.params[1], oblk := nt.params[3], opat := nt.params[4], iblk := nt.params[5], ipat := nt.params[6],
            Tensor(Mat(List(opat, i->BasisVec(size/oblk, i))), I(oblk)).terminate() * C[1])
    ),

    IOPrunedDFT__PrunedDFT_T_Scat := rec(
       forTransposition := true,
       maxSize := 16,
       applicable := (self, nt) >> not nt.hasTags() and nt.params[1] <= self.maxSize and
           (((not nt.params[3] = 1) and nt.params[5] = 1) or(nt.params[3] * nt.params[5] < nt.params[1])),
       children := nt -> [[ PrunedDFT(nt.params[1], nt.params[2], nt.params[3], nt.params[4]).transpose() ]],
       apply := (nt, C, cnt) -> let(size := nt.params[1], oblk := nt.params[3], opat := nt.params[4], iblk := nt.params[5], ipat := nt.params[6],
            C[1] * Tensor(Mat(List(ipat, i->BasisVec(size/iblk, i))).transpose(), I(iblk)).terminate())
    ),

    IOPrunedDFT_CT := rec(
       forTransposition := true,
       applicable := (self, nt) >> nt.params[1] > 2 and nt.params[3] * nt.params[5] >= nt.params[1]
            and not nt.hasTags()
            and not IsPrime(nt.params[1])
            and nt.params[3] > 1,
        children  := nt -> Map2(Filtered(DivisorPairs(nt.params[1]), (mn) -> IsInt(nt.params[3]/mn[2]) and IsInt(nt.params[5]/mn[1])),
            (m,n) -> [
                PrunedDFT(m, nt.params[2] mod m, nt.params[3]/n, nt.params[4]).transpose(),
                PrunedDFT(n, nt.params[2] mod n, nt.params[5]/m, nt.params[6]) ]
        ),
        apply := (nt, C, cnt) -> let(mn := nt.params[1], m := Cols(C[1]), n := Rows(C[2]),
            Tensor(C[1], I(n)) *
            Diag(fPrecompute(Tw1(mn, n, nt.params[2]))) *
            L(mn, m) * Tensor(C[2], I(m))
        ))
));
