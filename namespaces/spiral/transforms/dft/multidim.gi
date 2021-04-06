
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RemoveOnes := x -> Filtered(x, i->i<>1);

#F MDDFT(<dims>, [<exp>]) - multi-dimensional DFT non-terminal
#F   dims = [ <n_1>,.., <n_t> ] list of (positive) dimensions
#F   exp = root of unity exponent scaling (see DFT for exact definition)
#F
#F Definition : multidimensional matrix of size NxN, where N=n_1*..*n_t
#F      can also be represented as Tensor(DFT(n_1), ..., DFT(n_t))
#F
#F Example (direct)  : MDDFT([2,4,4])
#F Example (inverse) : MDDFT([2,4,4], -1)
#F
Class(MDDFT, TaggedNonTerminal, rec(
    abbrevs := [
    P     -> Checked(IsList(P), ForAll(P,IsPosInt), Product(P) > 1,
             [ RemoveOnes(P), 1, false ]),
    (P,k) -> Checked(IsList(P), ForAll(P,IsPosInt), IsInt(k), Product(P) > 1,
                     Gcd(Product(P), k)=1,
             [ RemoveOnes(P), k mod Product(P), false ]),
    (P,k,rc) -> Checked(IsList(P), ForAll(P,IsPosInt), IsInt(k), Product(P) > 1,
                     Gcd(Product(P), k)=1,
             [ RemoveOnes(P), k mod Product(P), rc ])
    ],
    dims := self >> let(n := Product(self.params[1]), When(self.isReal(), 2*[n,n], [n, n])),

    terminate := self >> let(t:=Tensor(List(self.params[1],
                                i -> DFT(i, self.params[2]).terminate())),
                            When(self.isReal(), MatAMat(RC(t).toAMat()), t)
                         ),

    transpose := self >> Copy(self),

    isReal := self >> self.params[3],

    setAB := meth(self, ab)
       self.a := ab[1];
       self.b := ab[2];
       return self;
    end,

    normalizedArithCost :=  (self) >> let(n := Product(self.params[1]),
                                        IntDouble(5 * n * d_log(n) / d_log(2)) )

#D    setpv := meth(self, pv)
#D        local s;
#D        s:= Copy(self);
#D        s.params[3] := pv;
#D        return s;
#D    end,

#D    tagpos := 3

));

# check for 1 dimensional MDDFTs and convert them to DFTs
catch1d_mddft := ch ->
    List(ch, x -> List(x, t -> When(Length(t.params[1]) > 1, t,
        DFT(Rows(t), t.params[2], t.params[3], t.params[4]))));

NewRulesFor(MDDFT, rec(
    #F RuleMDDFT_Base:  MDDFT -> DFT
    #F
    MDDFT_Base := rec(
        info := "MDDFT -> DFT",
        applicable     := nt -> Length(nt.params[1])=1,
        children       := nt -> let(P := nt.params, tags := nt.getTags(), [[ DFT(P[1][1], P[2]).withTags(tags) ]]),
        apply          := (nt, C, Nonterms) -> C[1]
    )
));

NewRulesFor(MDDFT, rec(

    #F RuleMDDFT_Tensor: MDDFT(n_1,n_2,...,n_t) = Tensor(DFT_n1, DFT_n2, ..., DFT_nt)
    #F
    MDDFT_Tensor := rec(
        switch := false,
        info :="MDDFT_n -> Tensor(DFT_n1,DFT_n2,...DFT_nt)",
        applicable := nt -> Length(nt.params[1])>1 and not nt.hasTags(),
        children  := nt -> [ List(nt.params[1],i->DFT(i, nt.params[2])) ],
        rule := (nt, C, cnt) -> Tensor(C)
#D        isApplicable := P -> Length(P[1])>1 and Length(P[3]) = 0,
#D        allChildren  := P -> [ List(P[1],i->DFT(i, P[2])) ],
#D        rule := (P,C) -> Tensor(C)
    ),

    #F RuleMDDFT_Dimless
    #F
    #F If N = RxS = n_1xn_2x...n_t and
    #F d1 = n_1* ..*n_(l-1), d2 = n_(l+1)*..*n_t, n_(l) = a*b then,
    #F MDDFT([n_1,..,n_t]) = Tensor(MDDFT([n_1,..,n_(l-1),a]), I(b), I(d2))*
    #F                       Tensor(I(d1), T(n_(l),b), I(d2))*
    #F                       Tensor(I(d1), I(a), MDDFT([b,n_(l+1),..,n_t])) *
    #F                       Tensor(I(d1), L(n_(l),a), I(d2))
    #F
    MDDFT_RowCol := rec (
        info := "MDDFT_n -> MDDFT_n/d, MDDFT_d",
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            List([1..len-1],
            i -> [ MDDFT(dims{[1..i]}, nt.params[2]), MDDFT(dims{[i+1..len]}, nt.params[2]) ])),

        apply := (nt, C, Nonterms) -> let(
            a := Last(Nonterms[1].params[1]),
            n1 := Rows(Nonterms[1])/a,
            n2 := Rows(Nonterms[2]),
            Tensor(C[1], I(n2)) *
            Tensor(I(n1), Tensor(I(a), C[2]))
        )
#D    isApplicable := P -> Length(P[1]) > 1 and Length(P[3]) = 0,
#D
#D    allChildren := P -> let(
#D        dims := P[1],
#D        len := Length(dims),
#D        List([1..len-1],
#D        i -> [ MDDFT(dims{[1..i]}, P[2]), MDDFT(dims{[i+1..len]}, P[2]) ])),
#D
#D    rule := (P,C,Nonterms) -> let(
#D        a := Last(Nonterms[1].params[1]),
#D        n1 := Rows(Nonterms[1])/a,
#D        n2 := Rows(Nonterms[2]),
#D        Tensor(C[1], I(n2)) *
#D        Tensor(I(n1), Tensor(I(a), C[2])))
    ),

    MDDFT_Dimless := rec (
        info := "MDDFT_n -> MDDFT_n/d, MDDFT_d",
        applicable := nt -> Length(nt.params[1]) > 1 and ForAny(nt.params[1], x->not IsPrime(x)) and not nt.hasTags(),

        children := function(nt)
            local ch, dims, len, simple_splits, div_splits;
            dims := nt.params[1];
            len := Length(dims);
            return Concatenation(List([1..len], i -> let(
               left := dims{[1..i-1]},
               right := dims{[i+1..len]},
               List(DivisorPairs(dims[i]), split ->
                   [ MDDFT(Concatenation(left, [split[1]]), nt.params[2]),
             MDDFT(Concatenation([split[2]], right), nt.params[2]) ]))));
        end,

        apply := (nt, C, Nonterms) -> let(
            a := Last(Nonterms[1].params[1]),
            b := Nonterms[2].params[1][1],
            n1 := Rows(Nonterms[1])/a,
            n2 := Rows(Nonterms[2])/b,

            Tensor(Tensor(C[1], I(b)), I(n2)) *
            Diag(diagTensor(fConst(n1,1), Tw1(a*b,b,nt.params[2]), fConst(n2,1))) *
            Tensor(I(n1), Tensor(I(a), C[2])) *
            Tensor(I(n1), L(a*b,a), I(n2))
        )
#D    isApplicable := P -> Length(P[1]) > 1 and ForAny(P[1], x->not IsPrime(x)) and Length(P[3]) = 0,
#D
#D    allChildren := meth(self,P)
#D        local ch, dims, len, simple_splits, div_splits;
#D        dims := P[1];
#D        len := Length(dims);
#D        return Concatenation(List([1..len], i -> let(
#D           left := dims{[1..i-1]},
#D           right := dims{[i+1..len]},
#D           List(DivisorPairs(dims[i]), split ->
#D               [ MDDFT(Concatenation(left, [split[1]]), P[2]),
#D         MDDFT(Concatenation([split[2]], right), P[2]) ]))));
#D
#D    end,
#D
#D    rule := (P,C,Nonterms) -> let(
#D        a := Last(Nonterms[1].params[1]),
#D        b := Nonterms[2].params[1][1],
#D        n1 := Rows(Nonterms[1])/a,
#D        n2 := Rows(Nonterms[2])/b,
#D
#D        Tensor(Tensor(C[1], I(b)), I(n2)) *
#D        Diag(diagTensor(fConst(n1,1), T(a*b,b), fConst(n2,1))) *
#D        Tensor(I(n1), Tensor(I(a), C[2])) *
#D        Tensor(I(n1), L(a*b,a), I(n2)))
    ),

    #   2D Vector Radix
    #   NOTE: Put citation
    MDDFT_vrdx2D := rec(
        info          := "MDDFT([mn, rs],k) -> MDDFT([m, r], k%mr), MDDFT([n, s], k%ns)",

	switch        := false,
        maxSize       := false,

        applicable := nt -> Length(nt.params[1]) = 2 and ForAll(nt.params[1], x->not IsPrime(x))
                                    and not nt.hasTags(),

        children  := nt -> let(l:=List(nt.params[1], i->DivisorPairs(i)),
            idx:=Cartesian(List([1..Length(l)], i->[1..Length(l[i])])),
            rdx := List(idx, i->List([1..2], j->List([1..2], k->l[k][i[1]][j]))),
            Map2(rdx, (m,n) -> [ MDDFT(m, nt.params[2] mod Product(m)), MDDFT(n, nt.params[2] mod Product(n)) ])
        ),

        apply := (nt, C, cnt) -> let(mn := Product(nt.params[1]), m := Rows(C[1]), n := Rows(C[2]),
            Tensor(C[1], I(n)) *
            Diag(Tw1(mn, n, nt.params[2])) *
            Tensor(I(m), C[2]) *
            L(mn, m)
        )
    )
#D    isApplicable := (self,P) >> Length(P[1]) = 2 and ForAll(P[1], x->not IsPrime(x))
#D                                and Length(P[3]) = 0,
#D
#D    allChildren  := P -> let(l:=List(P[1], i->DivisorPairs(i)),
#D            idx:=Cartesian(List([1..Length(l)], i->[1..Length(l[i])])),
#D            rdx := List(idx, i->List([1..2], j->List([1..2], k->l[k][i[1]][j]))),
#D            Map2(rdx, (m,n) -> [ MDDFT(m, P[2] mod Product(m)), MDDFT(n, P[2] mod Product(n)) ])
#D        ),
#D
#D    rule := (P,C) -> let(mn := P[1], m := Rows(C[1]), n := Rows(C[2]),
#D        Tensor(C[1], I(n)) *
#D        T(mn, n, P[2]) *
#D        Tensor(I(m), C[2]) *
#D        L(mn, m))
#D    )

));
