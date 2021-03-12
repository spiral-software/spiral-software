
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_setN := function(o) o.params[2].nested := true; return o; end;

#F Filt(<n>, <taps>, <v>, <ds>) - FIR filter (linear convolution)
#F
#F Filt is a non-terminal for a linear convolution matrix optionallly
#F downsampled by a factor of <ds>.
#F
#F For ds=1, it is a (n x n+l-1) matrix,  where l is the number of filter
#F coefficients in <taps>.
#F
#F For ds<>1, it is a (n x ds*(n-1)+l) matrix.
#F
#F <n>    - number of outputs desired
#F <taps> -  list of filter taps or a polynomial transfer function
#F <v>    - valuation, optional, but used when <taps> is not a polynomial
#F <ds>   - downsampling factor
#F
#F Examples: Filt(5, [1, 2, 3, 4, 5]) corresponds to the matrix
#F             [ [1, 2, 3, 4, 5, 0, 0, 0, 0]
#F               [0, 1, 2, 3, 4, 5, 0, 0, 0]
#F               [0, 0, 1, 2, 3, 4, 5, 0, 0]
#F               [0, 0, 0, 1, 2, 3, 4, 5, 0]
#F               [0, 0, 0, 0, 1, 2, 3, 4, 5] ]
#F
#F           Filt(3, [1, 2, 3], -2, 2) corresponds to the matrix
#F             [ [1, 2, 3, 0, 0, 0, 0]
#F               [0, 0, 1, 2, 3, 0, 0]
#F               [0, 0, 0, 0, 1, 2, 3] ]
#F
#F  Alternatively for downsampled filters valuation can be omitted:
#F   Filt(n, taps, -Length(taps)+1, ds) is equivalent to
#F   DSFilt(n, taps, ds)
#F
Class(Filt, TaggedNonTerminal, DataNonTerminalMixin, rec(
    abbrevs := [
       (n,L)   -> Checked(IsPosIntSym(n), Concat([n], toFiltFunc(L), [1])),
       (n,L,v) -> Checked(IsPosIntSym(n), Concat([n], toFiltFuncV(L,v), [1])),
       (n,L,v,ds) -> Checked(IsPosIntSym(n), IsPosIntSym(ds), Concat([n], toFiltFuncV(L,v), [ds])),
    ],
    isReal := self >> true,

    filtlen := self >> self.params[2].domain(),

    setData := meth(self, newData) self.params[2] := newData; return self; end,

    dims := self >> let(n:=self.params[1], k:=self.params[2].domain(), ds:=self.params[4],
                        [n, ds*(n-1) + k]),

    terminate := self >> let(
	n    := self.params[1], 
	coef := self.params[2].lambda().tolist(),
	ds   := self.params[4],
	k    := Length(coef),
	When(n=1, Mat([coef]),
	          RowTensor(n,k-ds,Mat([coef])))),

    HashId := self >> let(h := [ self.params[1], self.params[2].domain(), self.params[4] ],
    When(IsBound(self.tags), Concatenation(h, self.tags), h)),

    print := (self,i,is) >> Print(self.name, "(", PrintCS(self.params), ")",
    When(self.hasTags(), Print(".withTags(", self.getTags(), ")"), "")),
    hashAs := meth(self)
        local hfilt;
        hfilt := Copy(self);
        hfilt.params[2] := FUnk(self.params[2].domain()); 
        return hfilt;
    end,
    normalizedArithCost := self >> self.params[1] * (2*self.params[2].domain() - 1)
));

#F DSFilt(n, taps, ds) - creates a downsampled filter nonterminal
#F
#F This command is equivalent to Filt(n, taps, -Length(taps)+1, ds)
#F
DSFilt := (n, taps, ds) -> let(
    func := toFiltFunc(taps),
    Filt(n, func[1], func[2], ds)
);


Conv := function(arg)
    local filt, tr, v, deg;
    filt := ApplyFunc(Filt, arg).transpose();
    deg := filt.params[2].domain();
    v   := filt.params[3];
    filt.params[2] := fCompose(filt.params[2], J(deg));
    filt.params[3] := -(v+deg)+1;
    return filt;
end;

_vecList := (lst, t, vlen) -> let(l := Length(lst), Checked(l mod vlen = 0,
    List([0..l/vlen-1], i -> Value(TVect(t, vlen), lst{[1+vlen*i .. vlen*(i+1)]}))
));

_pruneVecZeros := lst -> List(lst, x->When(ForAll(x.v, v->v=0), 0, x));

#  a  b  c  d  0  0  0
#  0  a  b  c  d  0  0
#  0  0  a  b  c  d  0
#  0  0  0  a  b  c  d

#  a  b  c  d  0  0  0  0  0  0
#  0  0  a  b  c  d  0  0  0  0
#  0  0  0  0  a  b  c  d  0  0
#  0  0  0  0  0  0  a  b  c  d

_vecTapsHoriz := function(filt, vlen)
    local zeros, taps, k, vtaps, t, ds;
    t := filt.params[2].range(); # data type
    ds := filt.params[4];
# spiral> _roundup;
# (n, v) -> v * QuoInt((n + v - 1), v)
# _roundup(k + (v-1) ds,  v)
# k + v ds -ds + v - 1
# (k - ds - 1) + v (ds+1)
# (k - ds - 1) / v + v * (ds+1)
    k := filt.params[2].domain();
    zeros := Replicate(vlen*(ds + 1 + QuoInt(k-ds-1, vlen)) - k, 0);
    taps := Concatenation(List(filt.params[2].tolist(), EvalScalar), zeros);
    vtaps := List([0 .. vlen-1], blk ->
    List([0 .. k-1 + Length(zeros)], j -> taps[1 + ((j - blk*ds) mod (k + Length(zeros)))]));
    vtaps := List(vtaps, x->_vecList(x,t,vlen));
    return vtaps;
end;

_vecTaps := function(filt, vlen)
    local zeros, taps, k, vtaps, t, ds;
    t := filt.params[2].range(); # data type
    ds := filt.params[4];
    zeros := Replicate((vlen-1)*ds, 0);
    taps := Concatenation(zeros, List(filt.params[2].tolist(), EvalScalar), zeros);
    k := filt.params[2].domain();
    vtaps := List((vlen-1)*ds + [1..k+(vlen-1)*ds], i -> Value(TVect(t, vlen),
             List([0..vlen-1], j -> taps[i-ds*j])));
    return vtaps;
end;



##
## Rules
##
NewRulesFor(Filt, rec(
    #F Filt_Base: (base case)
    #F
    #F Computes convolution by definition
    #F
    Filt_Base := rec(
    requiredFirstTag := ANoTag,
    info             := "Filt -> Mat",
    forTransposition := true,
    limit            := -1, # means no limit
    applicable     := (self, t) >> When(self.limit < 0, true,
        t.params[1] <= self.limit or t.params[2].domain() <= self.limit),
    apply := (t, C, Nonterms) -> t.terminate()
    ),

    #F Filt_Nest: splits a wider filter into narrower filters, i.e. diagonal stripes
    #F
    Filt_Nest := rec(
	requiredFirstTag := ANoTag,
	forTransposition := false,

        minLength := 2,
        maxLength := 16,
        minTaps := 4,

        libApplicable := (self, t) >> let(n:=t.params[1], ntaps:=t.params[2].domain(),
            leq(self.minLength, n, self.maxLength) * leq(self.minTaps, ntaps)),

	applicable := (self, t) >> let(n:=t.params[1], ntaps:=t.params[2].domain(),
            (IsSymbolic(n) or (n >= self.minLength and n <= self.maxLength)) and 
            (IsSymbolic(ntaps) or ntaps >= self.minTaps)),

        freedoms := (self, t) >> let(ntaps := t.params[2].domain(),
	    # we split into filters with 2^k taps, to constrain the search space 
	    [ List( Filtered([1..20], i -> 2^(i+1) <= ntaps), i -> [2^i, ntaps mod 2^i] ) ]),

	child := (t, fr) -> let(
	    n  := t.params[1], 
	    k  := t.params[2].domain(),
	    ds := t.params[4], 
            d := fr[1],
	    Cond(d[2]=0, [ _setN(DSFilt(n, d[1],ds)) ],
                         [ _setN(DSFilt(n, d[1],ds)), _setN(DSFilt(n, d[2], ds)) ])),

	apply := function(t, C, Nonterms) 
	    local n,k,kk,nfilts,j,core,tot_wid,bk_wid,rem_wid,taps,
	          bk_filtlen, rem_filtlen, ds;
	    taps := t.params[2];
	    n := t.params[1]; 
	    k := t.filtlen();
	    ds := t.params[4];
	    tot_wid := ds*(n-1)+k; 
	    bk_wid  := Cols(C[1]);

	    bk_filtlen := Nonterms[1].filtlen();
	    nfilts := idiv(k, bk_filtlen); 

      	    j := Ind(nfilts);
	    Nonterms[1].setData(fCompose(taps, fAdd(k, bk_filtlen, bk_filtlen * j)));

            core := ISumAcc(j, j.range, Scat(fId(n)) * C[1] *
                                Gath(fAdd(tot_wid, bk_wid, bk_filtlen * j)));

	    if Length(C) = 1 then return core;
	    else 
		rem_wid     := Cols(Nonterms[2]);
		rem_filtlen := Nonterms[2].filtlen();
		Nonterms[2].setData(fCompose(taps, fAdd(k, rem_filtlen, bk_filtlen * nfilts)));
		return SUMAcc(core, Scat(fId(n)) * C[2] *
		                    Gath(fAdd(tot_wid, rem_wid, bk_filtlen * nfilts)));
	    fi;
	end
    ), 

    #F Filt_OverlapSave: Block convolution using Overlap-Save method.
    #F
    #F Filter_n -> Filter_n/b
    #F
    Filt_OverlapSave := rec(
    info             := "Filt(n,h) -> Filt(b,h)",
    forTransposition := true,
    switch           := true,
    applicable       := t -> t.params[1] > 2 and t.params[2].domain() > 1,
    blockRatio       := 2,
    children      := (self,t) >> let(
        n  := t.params[1],
        k  := t.params[2].domain(),
        v  := t.params[3],
        ds := t.params[4],
        tags := t.getTags(),
        newTags := Cond(tags=[], [],
                    ObjId(tags[1])=AParSMP, Drop(tags, 1),
                tags),
        divs := List( Filtered([1..20], i -> 2^i <= self.blockRatio * k and 2^i < n),
                  i-> [i, n mod 2^i]),
        List(divs, d ->
        When( d[2] = 0,
            [Filt(2^d[1], k, v, ds).withTags(newTags)], # d|n
            [Filt(2^d[1], k, v, ds).withTags(newTags), Filt(d[2], k, v, ds).withTags(newTags)]))), # not(d|n)

    forceBB := 128, # set to 32 to force reasonable unrolling

    apply := meth(self, t, C, Nonterms)
        local n, k, s, ds, bb, tags, smp, dosmp;
        Nonterms[1].setData(t.params[2]);
        n := t.params[1];
        k := t.params[2].domain();
        s := Rows(C[1]);
        ds := t.params[4];
        tags := t.getTags();
        smp := When(tags<>[] and ObjId(tags[1])=AParSMP, s -> SMP(tags[1].p, s), s->s);
        dosmp := tags<>[] and ObjId(tags[1])=AParSMP;

        # NOTE, this allows us not to use .area() for unrolling, since BB
        # explicitly forces unrolling onto small guys
        bb := When(not dosmp and self.forceBB > 0 and k*n <= self.forceBB, BB, x->x);
        if Length(C)=1 then
        return smp(bb(RowTensor(n/s, k-ds, C[1]))); # d|n
        else
        Nonterms[2].setData(t.params[2]);
        return RowDirectSum(k-ds, smp(bb(RowTensor(QuoInt(n,s),k-ds,C[1]))), C[2]); # not(d|n)
        fi;
    end
    ),

  Filt_Vect := rec(
    forTransposition := true,
    switch           := true,
    requiredFirstTag := AVecReg,
    applicable       := t -> (t.params[1] mod t.getTags()[1].v)=0 and t.params[2].domain() >= 1,
    freedoms := t -> [],

    child := (t, freedoms) -> let(
        n  := t.params[1], k := t.params[2].domain(), v := t.params[3], ds := t.params[4],
        vlen := t.getTags()[1].v,
        [ Filt(n/vlen, k+(vlen-1)*ds, v-(vlen-1)*ds, ds*vlen)
         .withTags(Drop(t.getTags(), 1)) ]),

    apply := meth(self, t, C, Nonterms)
        local vlen;
        vlen := t.getTags()[1].v;
        Nonterms[1].setData(FData(_vecTaps(t, vlen)));
        return VTensor(C[1], vlen) * VGath_dup(fId(Cols(t)), vlen);
    end
    ),

  Filt_VectHoriz := rec(
    forTransposition := true,
    switch           := true,
    requiredFirstTag := AVecReg,
    applicable       := t -> let(vlen := t.getTags()[1].v,
        t.params[1] = vlen and
        t.params[2].domain() >= 1
    ),
    freedoms := t -> [],

    child := (t, freedoms) -> let(vlen := t.getTags()[1].v,
        vtaps := _vecTapsHoriz(t, vlen),
        [ TTag(TL(vlen^2, vlen, 1, 1), AVecReg(vlen)),
          Filt(1, vtaps[1]) ]),

    apply := meth(self, t, C, Nonterms)
        local vlen, vtaps, filts, vv;
        vlen := t.getTags()[1].v;
        vtaps := _vecTapsHoriz(t, vlen);
        filts := [];
        for vv in vtaps do
            Nonterms[1].setData(vv);
        Add(filts, Copy(C[1]));
        od;

        return VTensor(RowVec(fConst(vlen, 1)), vlen) *
               C[1] *
           VTensor(VStack(filts), vlen);
    end
    ),

    # Vectorization by converting a filter into a v-way filterbank 
    #
    Filt_VectBank := rec(
 	forTransposition := true,
 	switch           := true,
 	requiredFirstTag := AVecReg,
        libApplicable    := t -> eq(imod(t.params[1], t.firstTag().v), 0),
 	applicable       := t -> let(n:=t.params[1], vlen:=t.firstTag().v,
            IsSymbolic(n) or (n mod vlen)=0),

 	freedoms := t -> [],

	child := (self,t,fr) >> let(
 	    n := t.params[1], taps := t.params[2], v := t.params[3], ds := t.params[4], vlen := t.firstTag().v,
 	    [ Filt(n/vlen, taps, v, ds*vlen).withTags(Drop(t.getTags(), 1)) ]), 

 	apply := (self, t, C, Nonterms) >> let(
 	    n := t.params[1], taps := t.params[2], ds := t.params[4], vlen := t.firstTag().v,
            #GT(C[1], HH(Cols(t), Cols(C[1]), 0, [1, ds]),
            #         HH(Rows(t), Rows(C[1]), 0, [vlen, 1]), [vlen]).withTags(new_tags)
            VTensor(C[1], vlen) * 
            VGath_u(H(Cols(t), Cols(C[1]), 0, 1), vlen)
                     ),
    ),

));

RulesFor(Filt, rec(
    #F Filter_OverlapSaveFreq: Block convolution using Overlap-Save method suited
    #F for Frequency-domain rules. It blocks so that we get powers of two for
    #F the columns -> leading to Circulant rule.
    #F
    #F Filter_n -> Filter_n/b
    #F
    Filt_OverlapSaveFreq := rec(
    info             := "Filt(n,h) -> Filt(b,h)",
    forTransposition := true,
    switch           := true,
    isApplicable     := P -> let(n:=P[1], k:= P[2].domain(), x := Log2Int(n+k-2), ds := P[4],
                                     (ds=1) and (n > 2) and (k > 1) and (k < 2^x)),
    allChildren      := P -> let(
        n  := P[1],
        k  := P[2].domain(),
         m := n + k - 1,
        divs := List( Filtered([1..20], i -> 2^(i-3) <= k and 2^i >= k and 2^i < n+k-1),
                  i-> [i, n mod (2^i-k+1)]),
        List(divs, d->
        When( d[2] = 0,
            [Filt(2^d[1]-k+1, P[2], P[3])], # d|n
            [Filt(2^d[1]-k+1, P[2], P[3]), Filt(d[2], P[2], P[3])]))), # not(d|n)

    rule := (P, C) -> let(
        n := P[1],
        k := Length(P[2]),
        s := C[1].dimensions[1],
        When(Length(C)=1,
        RowTensor(n/s, k-1, C[1]), # d|n
        RowDirectSum(k-1, RowTensor(QuoInt(n,s),k-1,C[1]), C[2]))# not(d|n)
    )
    ),

    #F Filt_OverlapAdd: block convolution using Overlap-Add method.
    #F
    #F Filt_n -> Conv_n/b, Toeplitz, Toeplitz
    #F
    Filt_OverlapAdd := rec(
    info              := "Filt(n,h) -> Conv(b,h)",
    forTransposition  := true,
    isApplicable      := P -> P[1] > 2 and P[2].domain() <= P[1]+1 and P[4]=1,
    switch := false,
    allChildren := function(P)
        local n,l,divs,d, coef,list1,list2;
        coef := P[2].tolist();
        l := Length(coef);
        n := P[1]-l+1;
        divs := List(Filtered([1..20], i-> 2^(i-1) < l and 2^i < P[1]),
                 i-> [i, n mod 2^i]);
        list1:= Reversed(Concat(Replicate(l-2,0),coef{[1..l-1]}));
        list2:= Reversed(Concat(coef{[2..l]},Replicate(l-2,0)));
        return List(divs, d->
        When(d[2] = 0,
             [ Conv(2^d[1], coef, P[3]), Toeplitz(list1), Toeplitz(list2) ], # d|n
             [ Conv(2^d[1], coef, P[3]), Conv(d[2],coef, P[3]),            # not(d|n)
                                     Toeplitz(list1), Toeplitz(list2) ]));
    end,

    rule := (P, C) -> let(
        k := P[2].domain(),
        n := P[1] - k + 1,
        s := Cols(C[1]),
        When(Length(C)=3,
        ColDirectSum(k-1, C[2], ColTensor(n/s, k-1, C[1]), C[3]), # d|n
        ColDirectSum(k-1, C[3], ColTensor(QuoInt(n,s), k-1, C[1]), C[2], C[4])) # not(d|n)
    )
    ),


    #F Filt_OverlapAdd: block convolution using Overlap-Add method.
    #F
    #F Filt_n -> Ext * Conv_n/b
    #F
    Filt_OverlapAdd2 := rec(
    info              := "Filt(n,h) -> Conv(b,h)",
    switch            := false,
    forTransposition  := true,
    isApplicable      := P -> P[1] > 2 and P[2].domain() > 1 and P[4]=1,
    allChildren := function(P)
        local n,k,divs,d, coef;
        k := P[2].domain();
        n := P[1]+k-1;
        coef := P[2];
        divs := List( Filtered([1..20], i -> 2^(i-1) < k and 2^i < P[1]),
                  i -> [i, n mod 2^i]);
        return List(divs, d->
        When( d[2] = 0,
            [Conv(2^d[1], coef, P[3])], # d|n
            [Conv(2^d[1], coef, P[3]), Conv(d[2], coef, P[3])])); # not(d|n)
    end,
    rule := (P, C) -> let(
        k := P[2].domain(),
        n := P[1] + k - 1,
        s := Cols(C[1]),
        ext := Ext(P[1],k-1,k-1).transpose(),
        When(Length(C)=1,
        ext * ColTensor(n/s, k-1, C[1]), # d|n
        ext * ColDirectSum(k-1, ColTensor(QuoInt(n,s), k-1, C[1]), C[2]))# not(d|n)
    )
    ),

    #F Filt_KaratsubaSimple:
    #F
    #F Filt_n(h) -> Filt_n(h0) dirsum Filt_n(h1) dirsum Filt_n(h0+h1)
    #F
    #F h0,h1 - even and odd downsampled h
    #F This is radix-2 Karatsuba - "transposed graph" version.
    #F Both the size and the length of the filter have to be divisible by 2
    #F
    Filt_KaratsubaSimple:= rec(
    info              := "Filt(n,h) -> Filt(b,h_e) dirsum Filt(b,h_o) dirsum Filt(b,h_e+h_o)",
    forTransposition  := false,
    switch := false,

    lowerLimit := 8,
    upperLimit := 128,

    isApplicable := (self, P) >> let(n := P[1], k := P[2].domain(), ds := P[4],
        (ds=1) and
        (n >= self.lowerLimit and n <= self.upperLimit) and (n mod 2 = 0) and
        (k >= self.lowerLimit and k <= self.upperLimit) and (k mod 2 = 0) and
        not IsBound(P[2].nested)),

    allChildren := P -> let(n := P[1], k := P[2].domain(), v := Int(P[3]/2),
        [[ Filt(n/2, k/2, v) ]]),

    rule := function( P, C, Nonterms )
        local n, m, l, m0, S, R, v0, v1, h0, h1, hsum, deg, coeffs, hds, f1,f2,f3;
        [n, l, deg] := [P[1], P[2].domain(), P[3]];
        coeffs := List(P[2].tolist(), x->x.ev());
        hds := DownsampleTwo([coeffs,deg]);
        [h0,v0] := hds[1];
        [h1,v1] := hds[2];

            if v0 <> v1 then Error("valuations of downsampled filters aren't the same"); fi;

        if (deg mod 2 = 1) then [h0,h1] := [h1,h0]; fi;
        hsum := ListSum(h0, h1);

        Nonterms[1].setData(FData(h0)); f1 := Copy(C[1]);
        Nonterms[1].setData(FData(h1)); f2 := Copy(C[1]);
        Nonterms[1].setData(FData(hsum)); f3 := Copy(C[1]);

        m0 := n + l - 1;
        m := (m0+1)/2 - 1;
        S := Tensor(Mat([[1,0,1],[0,1,1]]), I(n/2));
        # NOTE: use 3-band filter bank (instead of 3 separate filters) here to improve locality
        return L(n,n/2) * S *
               VStack(f1 * HStack(I(m), O(m,1), -I(m)),
                  f2 * HStack(O(m,1), I(m), -I(m)),
              f3 * Gath(fAdd(m0,m,m+1))) *
           BB(OS(m0,2));
        end
    ),

    #F Filt_Circulant
    #F
    #F Computes convolution by definition
    #F
    Filt_Circulant := rec(
    info             := "Filt -> Circulant",
    forTransposition := true,
    isApplicable     := P -> let(n:= P[1], k:= P[2].domain(), m:= n+k-1, ds:=P[4],
                               ds = 1 and m = 2^(Log2Int(m)) and n/k <= 8 and k/n <= 2),
    allChildren      := P -> let(
             n := P[1], l := P[3], coef := P[2], r := l+coef.domain()-1,
         [[ Circulant(n+PosInt(-l)+ PosInt(r), FUnk(coef.domain()), l) ]]),

    rule := (P,C,Nonterms) -> let(
        n := P[1], l := P[3], r := l + P[2].domain() - 1,
        nt := Nonterms[1].setData(P[2]),
        Ext(n, PosInt(-l), PosInt(r)).transpose() * C[1] )
    ),

    #F Filt_Blocking
    #F
    #F Linear Convolution (Filter) -> Blocks (Toeplitzes)
    #F
    #F Filter is partitioned into square blocks of sizes given by all
    #F proper divisors of the input size.
    #F
    Filt_Blocking := rec(
    info             := "Filt(n,h) -> Toeplitz(b,h')",
    forTransposition := false,
    switch           := true,
    isApplicable     := P -> let(n := P[1], k := P[2].domain(), ds := P[4],
        ds = 1 and n > 2 and n < k and ObjId(P[2]) in [FData, FUnk, FList]),

        # split into dense and half-sparse toeplitz, rest will be unrolled
    allChildren := P -> let(n := P[1], k := P[2].domain(), b := n,
        [[ Toeplitz(2*b-1),
           Toeplitz(2*b-1, 0, b-1),
           Toeplitz(2*b-1, b, 2*b-1) ]]),

    #When((k-1) mod b <>0, [ Toeplitz(2*((k-1) mod b) - 1) ], [ ]),

    rule := function(P, C, Nonterms)
        local n, k, b, Ls, Lsdata, S, j, rem, ofs, jj;
        n  := P[1];
        k  := P[2].domain();
        b := n;

            # first block major part
        Ls := List(P[2].tolist(), x->x.eval());
        Ls := Concat(Replicate(b-1,0), Ls, Replicate(2*b - k mod b,0));
        Lsdata := FData(Ls);

        j := Ind(Int((k-1)/b)-1); # we unroll first and last iteration
        jj := Ind(Int((k-1)/b)+1); # we unroll first and last iteration
        Nonterms[1].setData(fCompose(Lsdata, fAdd(Lsdata.domain(), 2*b-1, (j+1)*b), J(2*b-1)));
        Nonterms[2].setData(fCompose(Lsdata, fAdd(Lsdata.domain(), 2*b-1, jj*b), J(2*b-1)));
        Nonterms[3].setData(fCompose(Lsdata, fAdd(Lsdata.domain(), 2*b-1, jj*b), J(2*b-1)));
        Nonterms[3].setNonZero(b - (k mod b), 2*b - 1);

        S := SUMAcc(
        Data(jj, V(0),
            Scat(fId(b)) * C[2] * Gath(fTensor(fBase(j.range+2,jj), fId(b)))),
        ISumAcc(j, j.range,
            Scat(fId(b)) * C[1] * Gath(fTensor(fBase(j.range+2,j+1), fId(b)))),
        Data(jj, V(j.range+1),
            Scat(fId(b)) * C[3] * Gath(fTensor(fBase(j.range+2,jj), fId(b))))
        );

            # we have to add a small remaining triangle to the right
        if (k-1)mod b <> 0 then
        ofs := b * Int(((k-1)/b)+1) + 1;
        rem := Reversed( Concat( Ls{[ofs .. ofs + (k-1)mod b - 1]}, Replicate((k-1)mod b -1, 0)));
        S := ColDirectSum((b+k-1) mod b, S, toeplitz(rem));
        fi;

        return S;
    end
    ),

    Filt_KaratsubaFast := rec(
    info              := "Filt(n,h) -> Filt(b,h_e) dirsum Filt(b,h_o) dirsum Filt(b,h_e+h_o)",
    forTransposition  := false,
    switch := false,

    lowerLimit := 8,
    upperLimit := 128,

    isApplicable := (self, P) >> let(n := P[1], k := P[2].domain(), ds := P[4],
        (ds=1) and
        (n >= self.lowerLimit and n <= self.upperLimit) and (n mod 2 = 0) and
        (k >= self.lowerLimit and k <= self.upperLimit) and (k mod 2 = 0)),

    allChildren := P -> let(n := P[1], k := P[2].domain(), v := Int(P[3]/2),
        [[ Filt(n/2, k/2, v) ]]),

    rule := function( P, C, Nonterms )
        local m, m0, S, R, l, h0, h1, hsum, deg, n, t, v, coeffs, newcoeffs, hds, j, dirsum, gfunc, one, minone;
        n := P[1]; l := P[2].domain(); deg := P[3];
        coeffs := List(P[2].tolist(), x->x.ev());
        hds := DownsampleTwo([coeffs,deg]);
            if hds[1][2] <> hds[2][2] then Error("valuations of downsampled filters aren't the same"); fi;

        v := hds[1][2]; h0 := hds[1][1]; h1 := hds[2][1];
        if (deg mod 2 = 1) then t := h1; h1 := h0; h0 := t; fi;
        hsum := ListSum(h0, h1);

        newcoeffs := FData(Concat(h0, h1, hsum));
        j := Ind(3);
        gfunc := fCompose(newcoeffs, fTensor(fBase(j), fId(l/2)));
        Nonterms[1].setData(gfunc);
        dirsum := ISum(j, j.range,
            Scat(fTensor(fBase(j), fId(n/2))) * C[1] * Gath(fTensor(fBase(j), fId(Cols(C[1])))));
        m0 := n + l - 1;
        m := (m0+1)/2 - 1;
        S := Tensor(Mat([[1,0,1],[0,1,1]]), I(n/2));

        one := Diag(fConst(m,1));
        minone := Diag(fConst(m,-1));

        R := VStack(
            SUMAcc(one*Gath(fAdd(m0,m,0)), minone*Gath(fAdd(m0,m,m+1))),
            SUMAcc(one*Gath(fAdd(m0,m,1)), minone*Gath(fAdd(m0,m,m+1))),
        one*Gath(fAdd(m0,m,m+1))
        );

        return L(n,n/2) * S * dirsum * R * OS(m0,2);
        end
    )

));
