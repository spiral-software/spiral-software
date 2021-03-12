
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(TRC_stream, DFT_tSPL_Bluestein_Stream, TTensorI_Stream_Diag_Uneven, TTensorI_Stream_Perm_Uneven);

IsTwoPower := i >> 2 ^ Log2Int(i) = i;
AllRadicesInclN := function(n)
    local divisors, radices;
    divisors := Drop(DivisorsInt(n), 1);
    radices := Filtered(divisors, i -> n=(i^LogInt(n, i)));
    return radices;
end;


NewRulesFor(MDDFT, rec(
    MDDFT_tSPL_HorReuse_Stream := rec(
        info         := "tSPL MDDFT_n (2d) -> Horizontal reuse streaming",

        maxRadix     := 32,
        minRadix     := 2,

        switch       := false,
        applicable   := nt ->
            Length(nt.params[1]) = 2
            and nt.params[1][1] = nt.params[1][2]
            and nt.isTag(1, AStream),

        children     := (self, nt) >> let(
            N := nt.params[1][1],
            k := Log2Int(N),
            radices := Filtered(AllRadicesInclN(N), i -> 
                i >= self.minRadix and i <= self.maxRadix
            ),
            List(radices, rdx -> let(
                j := Ind(2),
                [ 
                    TICompose(j, 2, TCompose([
			Cond(N > nt.firstTag().bs,
                            TCompose([TTensorI(DFTDR(N, 1, rdx), N, APar, APar),
                                 TTensorI(TDR(N, rdx), N, APar, APar),
			    ]),
			    TTensorI(DFT(N, 1), N, APar, APar)
			),
                        TL(N*N, N) 
                    ])).withTags(nt.getTags())
                ]
            ))
        ),

        apply        := (nt, c, cnt) -> c[1],
#D    isApplicable := (self, P) >> (Length(P[1]) = 2) and (P[1][1] = P[1][2]) and
#D                                 IsList(P[3]) and (Length(P[3]) = 1) and
#D                     IsBound(P[3][1].bs),
#D
#D    allChildren  := (self, P) >> let(
#D                              N := P[1][1],
#D                  k := Log2Int(N),
#D                  radices := Filtered(AllRadices(N), i-> i >= self.minRadix and
#D                  i <= self.maxRadix),
#D
#D                  List(radices, rdx -> let(
#D                  j := Ind(2),
#D                  [ TICompose(j, 2, TCompose( [
#D                       TTensorI(DFTDR(N, rdx), N, APar, APar),
#D                       TTensorI(TDR(N, rdx), N, APar, APar),
#D                       TL(N*N, N) ]), P[3]) ])
#D                  )),
#D
#D    rule         := (P, C) -> C[1],
    ),

    MDDFT_tSPL_Unroll_Stream := rec(
        info         := "tSPL MDDFT_n (2d) -> Horizontal reuse streaming",

        maxRadix     := 32,
        minRadix     := 2,

        applicable   := nt ->
            Length(nt.params[1]) = 2
            and nt.params[1][1] = nt.params[1][2]
            and nt.isTag(1, AStream),

        children     := (self, nt) >> let(
            N := nt.params[1][1],
            k := Log2Int(N),
            radices := Filtered(AllRadicesInclN(N), i -> 
                i >= self.minRadix and i <= self.maxRadix
            ),
            List(radices, rdx -> let(
                j := Ind(2),
                [ 
                    TCompose([
			Cond(N > nt.firstTag().bs, TCompose([
                            TTensorI(DFTDR(N, 1, rdx), N, APar, APar),
                            TTensorI(TDR(N, rdx), N, APar, APar)]),
			    TTensorI(DFT(N, 1), N, APar, APar)),
                        TL(N*N, N),
			Cond(N > nt.firstTag().bs, TCompose([
                            TTensorI(DFTDR(N, 1, rdx), N, APar, APar),
                            TTensorI(TDR(N, rdx), N, APar, APar)]),
			    TTensorI(DFT(N, 1), N, APar, APar)),
                        TL(N*N, N)
                    ]).withTags(nt.getTags())
                ]
            ))
        ),
    
        apply := (nt, c, cnt) -> c[1],

        switch       := false

#D    isApplicable := (self, P) >> (Length(P[1]) = 2) and (P[1][1] = P[1][2]) and
#D                                 IsList(P[3]) and (Length(P[3]) = 1) and
#D                     IsBound(P[3][1].bs),
#D        allChildren  := (self, P) >> let(
#D                              N := P[1][1],
#D                  k := Log2Int(N),
#D                  radices := Filtered(AllRadices(N), i-> i >= self.minRadix and
#D                  i <= self.maxRadix),
#D
#D                  List(radices, rdx -> let(
#D                  j := Ind(2),
#D                  [ TCompose( [
#D                       TTensorI(DFTDR(N, rdx), N, APar, APar),
#D                       TTensorI(TDR(N, rdx), N, APar, APar),
#D                       TL(N*N, N),
#D
#D                       TTensorI(DFTDR(N, rdx), N, APar, APar),
#D                       TTensorI(TDR(N, rdx), N, APar, APar),
#D                       TL(N*N, N) ], P[3]) ])
#D                  )),
#D    rule         := (P, C) -> C[1],
    )
));

_MaxReImVec := function(v)
   local i, largest;

   largest := 0;
   for i in [1..Length(v)] do
       if (AbsFloat(ReComplex(v[i])) > largest) then
           largest := AbsFloat(ReComplex(v[i]));
       fi;
       if (AbsFloat(ImComplex(v[i])) > largest) then
           largest := AbsFloat(ImComplex(v[i]));
       fi;
   od;
   return largest;
end;


##########################################################################################
#   tSPL DFT -> DFT * DR rule
NewRulesFor(DFT, rec(
    DFT_tSPL_Stream   := rec (
        info             := "DFT_(n,l) -> DFTDR(n,l,r) * DR(n,r)",
        forTransposition := false,

        radix            := 2,

        applicable := (self, nt) >> nt.isTag(1, AStream) and 
                    nt.firstTag().legal_kernel(self.radix) and 
                    nt.params[1] = self.radix ^ LogInt(nt.params[1], self.radix) and
		    nt.params[1] <> self.radix,

        children := (self, nt) >> let(
            N          := nt.params[1],
            k          := Log2Int(nt.params[1]),
            streamsize := nt.firstTag().bs,
            radix      := self.radix,

            [[ TCompose([DFTDR(nt.params[1], nt.params[2], radix), TDR(nt.params[1], radix)]).withTags(nt.getTags()) ]]

        ),
        apply := (nt, c, cnt) -> c[1],

        switch := false
    ),

   DFT_tSPL_StreamFull := rec(
        forTransposition := false,
        
        applicable := (self, nt) >> nt.isTag(1, AStream) and (nt.params[1] = paradigms.stream.DFT_tSPL_Stream.radix) and (nt.params[1] = nt.firstTag().bs),
        
        children := (self, nt) >> [[ DFT(nt.params[1], nt.params[2]) ]],

        apply := (nt, c, cnt) -> c[1],

        switch := false
    ),

    DFT_tSPL_Stream_Trans   := rec (
        info             := "DFT_(n,l) -> DR(n,r) * DRDFT(n,l,r)",
        forTransposition := false,

        radix            := 2,

        applicable := (self, nt) >> nt.isTag(1, AStream) and 
                    nt.firstTag().legal_kernel(self.radix) and 
                    nt.params[1] = self.radix ^ LogInt(nt.params[1], self.radix),

        children := (self, nt) >> let(
            N          := nt.params[1],
            k          := Log2Int(nt.params[1]),
            streamsize := nt.firstTag().bs,
            radix      := self.radix,

            [[ TCompose([TDR(nt.params[1], radix), DRDFT(nt.params[1], nt.params[2], radix)]).withTags(nt.getTags()) ]]

        ),
        apply := (nt, c, cnt) -> c[1],

        switch := false
    ),
    
    DFT_tSPL_MultRadix_Stream := rec(
        forTransposition := false,
        
        applicable := (self, nt) >> let(
            n  := nt.params[1],
            rp := paradigms.stream.DFT_tSPL_Stream.radix,
            np := rp ^ LogInt(n,rp),
            r  := n/np,
            nt.isTag(1, AStream) and not paradigms.stream.DFT_tSPL_Stream.applicable(nt) and r <> 1 and 2^Log2Int(n)=n
        ),

        children := (self, nt) >> let(
            n := nt.params[1],
            rp := paradigms.stream.DFT_tSPL_Stream.radix,
            np := rp ^ LogInt(n,rp),
            r  := n/np,
            
            [[ TCompose([ 
                    TL(n, r),
                    TTensorI(DFT(r, nt.params[2]), np, APar, APar),
                    TDiag(TI(n, r, 0, nt.params[2])),
                    TL(n,np),
                    TTensorI(DFT(np, nt.params[2]), r, APar, APar),
                    TL(n,r)]).withTags(nt.getTags()) ]]
        ),

        apply := (nt, c, cnt) -> c[1],

        switch := false

    ),

    DFT_tSPL_Mixed_Radix_Stream := rec (
        forTransposition := false,
        switch := false,

        badRadices := [3, 5, 7],
        # control the good radix based on DFT_tSPL_Stream.radix, to simplify setup

        applicable := (self, nt) >> let(
            n := nt.params[1],
            badradix := Filtered(self.badRadices, r->(IsInt(n/r) and IsTwoPower(n/r))),
            goodradix := paradigms.stream.DFT_tSPL_Stream.radix,
            
            Length(badradix) = 1 and
            (n/badradix[1]) = goodradix ^ LogInt((n/badradix[1]), goodradix) and
            nt.isTag(1, AStream) and
            IsInt(nt.firstTag().bs/goodradix)
        ),
               

        children := (self, nt) >> let(
            n  := nt.params[1],
            b  := Filtered(self.badRadices, r->(IsInt(n/r) and IsTwoPower(n/r)))[1],
            r  := paradigms.stream.DFT_tSPL_Stream.radix,
            np := n/b,
            w  := nt.firstTag().bs,                       
            t  := LogInt(np, r),
                    ## streaming I(b) x DFTDR(np, goodradix, nt.params[2]) with radix goodradix, ##

            [[  TCompose(Concatenation(
#                  [ TPrm(TL(n, np, 1, 1)),
                  [ TPrm(L(n, np)),
#                    TPrm(TL(np, r, b, 1)) ],
                     TTensorI(TPrm(L(np,r)), b, APar, APar) ],

                  List([0..t-3], e -> TCompose([
                      TTensorI(DFT(r, nt.params[2]), b*(r^(t-1)), APar, APar),
                      TTensorI(TDiag(fPrecompute(Tw1(r^(t-e), r, nt.params[2]))), b*(r^e), APar, APar),
#                      TTensorI(TPrm(TL(r^(t-e), r^(t-e-1))), b*(r^e), APar, APar),
                      TTensorI(TPrm(L(r^(t-e), r^(t-e-1))), b*(r^e), APar, APar),
#                      TTensorI(TPrm(TL(r^(t-e-1), r)), b*(r^(e+1)), APar, APar)
                      TTensorI(TPrm(L(r^(t-e-1), r)), b*(r^(e+1)), APar, APar)
                  ])),

                  [ TTensorI(DFT(r, nt.params[2]), b*(r^(t-1)), APar, APar),
                    TTensorI(TDiag(fPrecompute(Tw1(r*r, r, nt.params[2]))), b*(r^(t-2)), APar, APar),
                    TTensorI(TPrm(L(r*r, r)), b*(r^(t-2)), APar, APar),
                    TTensorI(DFT(r, nt.params[2]), b*(r^(t-1)), APar, APar),
                    TTensorI(TPrm(TDR(np, r)), b, APar, APar),
                    TPrm(L(n, b)),
                    TDiag(Tw1(n, b, nt.params[2])),
                    TTensorI(DFT(b, nt.params[2]), np, APar, APar),
                    TPrm(L(n, np)) ])).withTags(nt.getTags())
            ]]
        ),

        apply := (nt, c, cnt) -> c[1],
        
        
    ),

    DFT_tSPL_Mixed_Radix_It := rec (
        forTransposition := false,
        switch := false,

        badRadices := [3, 5, 7],
        # control the good radix based on DFT_tSPL_Stream.radix, to simplify setup
        
        applicable := (self, nt) >> let(
            n := nt.params[1],
            badradix := Filtered(self.badRadices, r->(IsInt(n/r) and IsTwoPower(n/r))),
            goodradix := paradigms.stream.DFT_tSPL_Stream.radix,
            
            Length(badradix) = 1 and
            (n/badradix[1]) = goodradix ^ LogInt((n/badradix[1]), goodradix) and
            nt.isTag(1, AStream) and
            IsInt(nt.firstTag().bs/goodradix)
        ),
               

        children := (self, nt) >> let(
            n  := nt.params[1],
            b  := Filtered(self.badRadices, r->(IsInt(n/r) and IsTwoPower(n/r)))[1],
            r  := paradigms.stream.DFT_tSPL_Stream.radix,
            np := n/b,
            w  := nt.firstTag().bs,                       
            t  := LogInt(np, r),
            j  := Ind(t),

            [[  TCompose([ 
                    TPrm(TL(n, np, 1, 1)),
                    TICompose(j, j.range, TCompose([
                        TTensorI(TPrm(TL(r^t,r)), b, APar, APar),
                        TTensorI(DFT(r, nt.params[2]), b*(r^(t-1)), APar, APar),
                        TTensorI(TDiag(fPrecompute(TC(np, r, j, nt.params[2]))), b, APar, APar),
                    ])),
                    TTensorI(TPrm(TDR(np, r)), b, APar, APar),
                    TPrm(TL(n, b, 1, 1)),
                    TDiag(Tw1(n, b, nt.params[2])),
                    TTensorI(DFT(b, nt.params[2]), np, APar, APar),
                    TPrm(TL(n, np, 1, 1))]).withTags(nt.getTags())
            ]]            
        ),

        apply := (nt, c, cnt) -> c[1],
        
        
    ),

    #F DFT_tSPL_Bluestein_Stream : Convert FFT to Toeplitz matrix, then embed in a larger Circulant,
    #F     specialized for streaming

    # NB: currently, this will not work with a multiple-radix streaming design, since that relies on
    # breaking down DFT, not DFTDR.  (e.g., DFT(128) -> DFT(64) and DFT(2) to implement radix 8).
    # This can be fixed.  I would just need to make sure that the mult-radix rule can do DRDFT and DFTDR.

    DFT_tSPL_Bluestein_Stream := rec(
        minRoundup := 8,
        customFilter := True,
        goodFactors := [2],

        applicableSizes := i -> not IsTwoPower(i),

        circSizes := meth(self,N)
            local low, high, cands;
            low := 2 * N + 1;
            high := 4 * 2^Log2Int(N);
            cands := When(high < self.minRoundup, [self.minRoundup], [low..high]);
            cands := Filtered(cands, self.customFilter);
            cands := Filtered(cands, i->IsEvenInt(i) and IsSubset(Set(self.goodFactors), Set(Factors(i))));
            return cands;
        end,

        toeplitz := (N, k) ->
            List([1..2*N-1], x -> ComplexW(2*N, -(N-x)^2 * k)),

        diag := (n, m, k) ->
            Concatenation(
                List([0..m-1], x -> ComplexW(2*m, x^2 * k)),
                List([m..n-1], x -> Cplx(0,0))
            ),

        circulantToep := (toep,size) -> let(l:=Length(toep),
            Concatenation(
                List([1..(l+1)/2],   x->toep[(l+1)/2+x-1]),
                List([1..size-l],    x->0),
                List([1..(l-1)/2],   x->toep[x]))),

        switch := false,
        applicable := (self, t) >> self.applicableSizes(Rows(t)) and t.hasTags() and t.isTag(1, AStream) and IsTwoPower(t.firstTag().bs),

        children := meth(self, t)
            local P, N, k, circsize, diag, toep, circ, diagonalized_circ, sc, tags, kids, radix, maxval, scaledownby, l2max, diagConj, rl;

            radix := paradigms.stream.DFT_tSPL_Stream.radix;
            tags := t.getTags();
            P := t.params;
            N := P[1];
            k := P[2];
            toep := self.toeplitz(N,k);
            kids := [];

            for circsize in self.circSizes(N) do
                diag := fPrecompute(FData(self.diag(circsize, N,k)));
                circ := self.circulantToep(toep, circsize);
                    
                if (radix^LogInt(circsize, radix) = circsize) then

                    # maxval := _MaxReImVec(ComplexFFT(circ));
                    # l2max := log(maxval).ev() / log(2).ev();
                    # l2max := Cond(l2max = IntDouble(l2max), IntDouble(l2max), IntDouble(l2max)+1);
                    # scaledownby := 2^l2max;

                    # Print("maxval = ", maxval, "\n");
                    # Print("l2max = ", l2max, "\n");
                    # Print("scaledownby = ", scaledownby, "\n");
                    # Print("saving = ", scaledownby/circsize, "\n");
                    # diagonalized_circ := fCompose(FData(1/scaledownby * ComplexFFT(circ)), DR(circsize, radix));
                    # sc := fConst(TComplex, circsize, scaledownby/circsize);
                    # diagonalized_circ := fCompose(FData(1/circsize * List(ComplexFFT(circ), i->Cplx(ReComplex(i), -1*ImComplex(i)))), DR(circsize, radix));
                    diagonalized_circ := fPrecompute(fCompose(FData(1/circsize * ComplexFFT(circ)), DR(circsize, radix)));

                    Add(kids, [
                        TCompose([
                            TDiag(diag),
                            DFTDR(circsize, -1, radix),
                            # TDiag(sc),
                            TDiag(diagonalized_circ),
                            DRDFT(circsize, 1, radix),
                            TDiag(diag),
                        ]).withTags(tags)
                    ]);
                else
                    rl := circsize / (radix ^ LogInt(circsize, radix));
                    diagonalized_circ := fPrecompute(fCompose(
                            FData(1/circsize * ComplexFFT(circ)), 
                            L(circsize, rl),
                            fTensor(fId(rl), DR(circsize/rl, radix))
                    ));

                    Add(kids, [
                        TCompose([
                            TDiag(diag),

                            TPrm(TL(circsize, rl)),
                            TTensorI(DFT(rl, -1), circsize/rl, APar, APar),
                            TPrm(TL(circsize, circsize/rl)),
                            TDiag(Tw1(circsize, circsize/rl, -1)),
                            TTensorI(DFTDR(circsize/rl, -1, radix), rl, APar, APar),

                            TDiag(diagonalized_circ),

                            TTensorI(DRDFT(circsize/rl, 1, radix), rl, APar, APar), 
                            TDiag(Tw1(circsize, circsize/rl, 1)),
                            TPrm(TL(circsize, rl)),
                            TTensorI(DFT(rl, 1), circsize/rl, APar, APar),
                            TPrm(TL(circsize, circsize/rl)),

                           TDiag(diag),
                       ]).withTags(tags)
                    ]);
                            
                fi;

            od;

            return kids;
        end,

        apply := (t, C, Nonterms) -> let(
            P := t.params,
            w := t.getTags()[1].bs,
            n := DFT_tSPL_Bluestein_Stream.circSizes(P[1])[1],
            m := P[1],
            
            StreamPadDiscard(m, n, w, w) * C[1] * StreamPadDiscard(n, m, w, w)
        )
    ),

    #F DFT_tSPL_Bluestein_Stream : Convert FFT to Toeplitz matrix, then embed in a larger Circulant,
    #F     specialized for streaming

    DFT_tSPL_Bluestein_Stream_Rolled := rec(
        minRoundup := 8,
        customFilter := True,

#PM        goodFactors := [2,3,5], # this change restricts FFT to be 2-power sized
        goodFactors := [2],

        applicableSizes := i -> not IsTwoPower(i),
#        applicableSizes := IsPrime,
        circSizes := meth(self,N)
            local low, high, cands;
            low := 2 * N + 1;
            high := 4 * 2^Log2Int(N);
            cands := When(high < self.minRoundup, [self.minRoundup], [low..high]);
            cands := Filtered(cands, self.customFilter);
            cands := Filtered(cands, i->IsEvenInt(i) and IsSubset(Set(self.goodFactors), Set(Factors(i))));
            return cands;
        end,

        toeplitz := (N, k) ->
            List([1..2*N-1], x -> ComplexW(2*N, -(N-x)^2 * k)),

        diag := (n, m, k) ->
            Concatenation(
                List([0..m-1], x -> ComplexW(2*m, x^2 * k)),
                List([m..n-1], x -> 0)
            ),

        circulantToep := (toep,size) -> let(l:=Length(toep),
            Concatenation(
                List([1..(l+1)/2],   x->toep[(l+1)/2+x-1]),
                List([1..size-l],    x->0),
                List([1..(l-1)/2],   x->toep[x]))),

        switch := false,
        applicable := (self, t) >> self.applicableSizes(Rows(t)) and t.hasTags()  and t.isTag(1, AStream) and IsTwoPower(t.firstTag().bs),

        children := meth(self, t)
            local P, N, k, circsize, diag, toep, circ, diagonalized_circ, sc, tags, kids, radix, maxval, scaledownby, l2max, i0, j, combdiag;

            radix := paradigms.stream.DFT_tSPL_Stream.radix;
            tags := t.getTags();
            P := t.params;
            N := P[1];
            k := P[2];
            toep := self.toeplitz(N,k);
            kids := [];
            j := Ind(2);

            for circsize in self.circSizes(N) do
                diag := fPrecompute(FData(self.diag(circsize, N,k)));
                circ := self.circulantToep(toep, circsize);
                diagonalized_circ := FData(1/circsize * ComplexFFT(circ));

                Add(kids, [
                    TCompose([                        
                        TDiag(diag),
                        TICompose(j, 2,
                            TCompose([
                                DFT(circsize, 2*j-1),   # 2*j-1 --> =-1 when j=0 and =1 when j=1
                                TDiag(fPrecompute(fCompose(diagDirsum(diagonalized_circ, diag), fTensor(fBase(2, j), fId(circsize)))))
                            ])
                        )                                                
                    ]).withTags(tags)
                ]);
            od;

            return kids;
        end,

        apply := (t, C, Nonterms) -> let(
            P := t.params,
            w := t.getTags()[1].bs,
            n := DFT_tSPL_Bluestein_Stream.circSizes(P[1])[1],
            m := P[1],
            
            StreamPadDiscard(m, n, w, w) * C[1] * StreamPadDiscard(n, m, w, w)
        )
    )
));

###############################################################
NewRulesFor(DFT, rec(
    #F DFT_HW_CT: 1965
    #F   General Cooley-Tukey Rule
    #F   DFT_n = (DFT_n/d tensor I_d) * diag * (I_n/d tensor F_d) * perm
    #F
    #    Radix size guided by previous tests.
    #F Cooley/Tukey:
    #F   An Algorithm for the Machine Calculation of Complex Fourier Series.
    #F   Mathematics of Computation, Vol. 19, 1965, pp. 297--301.
    #F
    DFT_HW_CT := rec(
        info          := "DFT(mn,k) -> DFT(m, k%m), DFT(n, k%n)",

        maxSize       := false,
        forcePrimeFactor := false,

        applicable := (self, nt) >> nt.params[1] > 2
			and not DFT_CT.applicable(nt)
            and not nt.hasTags()
            and (self.maxSize=false or nt.params[1] <= self.maxSize)
            and not IsPrime(nt.params[1])
            and When(self.forcePrimeFactor, not DFT_GoodThomas.applicable(nt), true),

        children  := nt -> Cond(nt.params[1] = 16, [[DFT(4, nt.params[2] mod 4), DFT(4, nt.params[2] mod 4)]], 
	    Cond(nt.params[1] = 8, [[DFT(4, nt.params[2] mod 4), DFT(2, nt.params[2] mod 2)]], 
		Cond(nt.params[1] = 16, [[DFT(4, nt.params[2] mod 4), DFT(4, nt.params[2] mod 4)]], 
		    Cond(nt.params[1] = 32, [[DFT(4, nt.params[2] mod 4), DFT(8, nt.params[2] mod 8)]], 
			Cond(nt.params[1] = 64, [[DFT(8, nt.params[2] mod 8), DFT(8, nt.params[2] mod 8)]], 
			    Cond(nt.params[1] = 128, [[DFT(8, nt.params[2] mod 8), DFT(16, nt.params[2] mod 16)]],
				Cond(nt.params[1] = 256, [[DFT(16, nt.params[2] mod 16), DFT(16, nt.params[2] mod 16)]],
				Map2(DivisorPairs(nt.params[1]), (m,n) -> [ DFT(m, nt.params[2] mod m), DFT(n, nt.params[2] mod n) ])
				)
			    )
			)
		    )
		)
	    )
	),

        apply := (nt, C, cnt) -> let(mn := nt.params[1], m := Rows(C[1]), n := Rows(C[2]),
            Tensor(C[1], I(n)) *
            Diag(fPrecompute(Tw1(mn, n, nt.params[2]))) *
            Tensor(I(m), C[2]) *
            L(mn, m)
        )
    ),


# special case: folded CT: 2*2, stream width = 4
    DFT_tSPL_Fold_CT := rec (
        applicable := nt ->
            nt.params[1] = 4 
            and nt.isTag(1, AStream) 
            and nt.firstTag().bs = 2,

        children := (nt) -> [[
            TCompose([
                TL(4,2,1,1),
                TTensorI(DFT(2), 2, APar, APar),
                TL(4,2,1,1),
                TDiag(Tw1(4,2,1)),
                TTensorI(DFT(2), 2, APar, APar),
                TL(4,2,1,1)
            ]).withTags(nt.getTags())
        ]],
                
        apply := (nt, c, cnt) -> c[1],

        switch           := false

    )
));


###############################################################
# unrolled streaming Iterative FFT: DFT(2*4^k)
# breaks down DFT(2*4^k)into one radix 2 stage and then 
# DFT(4^k, 4).

# This rule is only used when DFT_tSPL_Stream is not applicable.

NewRulesFor(DFT, rec(
    DFT_tSPL_StreamR2R4   := rec (
        forTransposition := false,
        minSize          := 8,

        applicable := (self, nt) >> let(
            n := nt.params[1],
            streamsize := Cond(nt.isTag(1, AStream),
                nt.firstTag().bs, -1),
            nt.hasTags() and
            n >= self.minSize and
            4^LogInt(n/2,4) = n/2 and
            not(DFT_tSPL_Stream.applicable(nt))),

        children := nt -> let(
            n := nt.params[1],
            k := LogInt(n/2, 4),
            z := nt.params[2],
            [[ TCompose([TL(n,2),
                        TTensorI(DFT(2), n/2, APar, APar),
                        TDiag(TI(n, 2, 0, z)),
                        TL(n, n/2),
                        TTensorI(DFT(n/2), 2, APar, APar),
                        TL(n, 2)]).withTags(nt.getTags()) ]] ),
        
        apply := (nt, c, cnt) -> c[1],
        switch := false
        )
));


###############################################################
# unrolled streaming Iterative FFT: DFTDR
NewRulesFor(DFTDR, rec(
    DFTDR_tSPL_Stream   := rec (
        info             := "Iterative FFT: Streaming DFTDR with combined permutations",
        forTransposition := false,
        minSize          := 8,

        applicable := (self, nt) >> let(
            streamsize := Cond(
                nt.isTag(1, AStream),
                nt.firstTag().bs,
                -1
            ),
            nt.hasTags()
            and nt.params[1] >= self.minSize
            and IsInt(nt.params[1]/nt.params[3])
#            and streamsize = -1 or streamsize >= nt.params[3]
        ),

        children := nt -> let(
            k := Log2Int(nt.params[1]),
            streamsize := Cond(
                nt.isTag(1, AStream),
                nt.firstTag().bs,
                -1
            ),
            ap := Cond(
                IsInt(nt.params[1]/nt.params[3]),
                [nt.params[3]], 
                []
            ),
            t := LogInt(nt.params[1], nt.params[3]),
            List(ap, r -> [ 
                TCompose(Concatenation(
                    [TL(r^t, r)],
                    List([0..t-3], e ->
                        TTensorI(
                            TCompose([
                                TTensorI(DFT(r, nt.params[2]), r^(t-e-1), APar, APar),
                                Cond(streamsize >= nt.params[3],
                                    TDiag(TI(nt.params[1], r, e, nt.params[2])),
                                    TDiag(fPrecompute(Tw1(nt.params[1]/(r^e), r, nt.params[2])))
                                ),
                                # TDiag(TCBase(P[1]/(r^e), P[1]/(r^(e+1)), 1)),
                                TL(r^(t-e), r^(t-e-1), 1, 1),
                                TL(r^(t-e-1), r, r, 1) 
                            ]),
                            r^e, APar, APar
                        )
                    ),
                    [ 
                        TTensorI(
                            TCompose( [
                                TTensorI(DFT(r, nt.params[2]), r, APar, APar),
                                Cond(streamsize >= nt.params[3],
                                    TDiag(TI(nt.params[1], r, t-2, nt.params[2])),
                                    #TDiag(TIFold(P[1], r, t-2))
                                    TDiag(fPrecompute(Tw1(nt.params[1]/(r^(t-2)), r, nt.params[2])))
                                ),
                                # TDiag(TCBase(P[1]/(r^(t-2)), P[1]/(r^(t-1)), 1)),
                                TL(r^(2), r, 1, 1) 
                            ]),
                            r^(t-2), APar, APar
                        ) 
                    ],
                    [ TTensorI(DFT(r, nt.params[2]), r^(t-1), APar, APar) ]
                )).withTags(nt.getTags()),
            ])
        ),

        apply := (nt, c, cnt) -> c[1],

        switch := false
    ),


));

###############################################################
# unrolled streaming Iterative FFT: DRDFT
NewRulesFor(DRDFT, rec(
    DRDFT_tSPL_Stream   := rec (
        info             := "Iterative FFT: Streaming DRDFT with combined permutations",
        forTransposition := false,
        minSize          := 2,
        applicable := (self, nt) >> let(
            streamsize := Cond(
                nt.isTag(1, AStream),
                nt.firstTag().bs,
                -1
            ),
            nt.hasTags()
            and nt.params[1] >= self.minSize
            and IsInt(nt.params[1]/nt.params[3])
#            and (streamsize = -1 or streamsize >= nt.params[3])
        ),

        children := nt -> let(
            k := Log2Int(nt.params[1]),
            streamsize := Cond(nt.isTag(1, AStream), nt.firstTag().bs, -1),

            ap := Cond(IsInt(nt.params[1]/nt.params[3]) and 
                       (streamsize = -1 or streamsize >= nt.params[3]),
                    [nt.params[3]],
                    []),

            t  := LogInt(nt.params[1], nt.params[3]),
            P  := nt.params,
            ex := nt.params[2],
            n  := nt.params[1],

            List(ap, r -> [ 
                TCompose(Concatenation(
                    [ TTensorI(DFT(r, ex), r^(t-1), APar, APar) ],
                    [ 
                        TTensorI(
                            TCompose([
                                TL(r^2, r, 1, 1),
                                TDiag(TI(n, r, t-2, ex)),
                                TTensorI(DFT(r, ex), r, APar, APar)
                            ]),
                        r^(t-2), APar, APar)
                    ],
                    Reversed( List([0..t-3], e ->
                        TTensorI(
                            TCompose([
                                TL(r^(t-e-1), r^(t-e-2), r, 1),
                                TL(r^(t-e), r, 1, 1),
                                TDiag(TI(n, r, e, ex)),
                                TTensorI(DFT(r, ex), r^(t-e-1), APar, APar)
                            ]),
                            r^e, APar, APar
                        )
                    )),
                    [ TL(r^t, r^(t-1)) ]
                )).withTags(nt.getTags())
            ])   
        ),

        switch := false,
        apply := (nt, c, cnt) -> c[1],

    ),

    # Pease tSPL DR(k,r)*DFT_(k) -> Tc \Prod((I tensor F)L)
    # This is kind of a hack; the normal pease rule in dftpease.gi should accomplish this
    DRDFT_tSPL_Pease_Stream   := rec (
        forTransposition := false,
        minSize := 2,

	# set to true to wrap TC in fPrecompute (i.e., for software)
	# leave as false to allow TC to be simplified by other rules
	# (i.e., for hardware)
	precompute := false,

        applicable := (self, t) >> let(
            tags := t.getTags(), n := t.params[1], k := t.params[2], radix := t.params[3], 
            stream := Cond(Length(tags)=1 and IsBound(tags[1].bs), tags[1].bs, -1),
               Length(tags) >= 1    and 
               (IsSymbolic(n)    or ((n >= self.minSize) and IsIntPower(n, radix)))
	       and IsInt(LogInt(n, radix) / self.unroll_its)
#           and k = 1
        ),

        freedoms := t -> [[ t.params[3] ]], # radix is the only degree of freedom

	# Used in HW to have multiple stages inside the TICompose.  For "normal"
	# operation, keep this at 1.
	unroll_its := 1,

        child := (self, t, fr) >> let(
            n     := t.params[1],
            radix := fr[1],
            e     := t.params[2],
            tags  := t.getTags(),
            j     := Ind(LogInt(n, radix)/self.unroll_its),
            stage := TTensorI(DFT(radix, e), n/radix, AVec, APar),
            fPre  := Cond(self.precompute, fPrecompute, x->x), 
	    twid  := i >> TDiag(fPre(TC(n, radix, self.unroll_its*j+i, e))),
	    full_stage := List([1..self.unroll_its], t->(TCompose([stage, twid(t-1)]))),
	    
            [ TICompose(j, j.range, TCompose(full_stage)).withTags(tags).transpose() ]
        ),

        apply := (t, C, Nonterms) -> C[1]
    )


));


##########################################################################################
#   DFT(n=2*4^k, [AStream(s)]) -> L(n, 2) * (I(n/2) x F(2)) * L(n, n/2) * T(n, n/2) *
#      (I(2) x (DFTDR(n/2, 4) * R(n/2, 4))) * L(n,2)
NewRulesFor(DFT, rec(
    DFT_tSPL_Four    := rec(
        forTransposition := false,
        minSize          := 8,
        applicable := (self, nt) >> let(
            streamsize := Cond(
                nt.isTag(1, AStream),
                nt.firstTag().bs,
                -1
            ),
            nt.hasTags()
            and nt.params[1] >= self.minSize 
            and 4^LogInt(nt.params[1]/2, 4) = nt.params[1]/2
        ),

        children := (nt) -> [[
            TCompose([
                TL(nt.params[1], 2),
                TTensorI(DFT(2), nt.params[1]/2, APar, APar),
                TDiag(Tw1(nt.params[1], 2, 1)),
                TL(nt.params[1], nt.params[1]/2),
                TTensorI(
                    TCompose([
                        DFTDR(nt.params[1]/2, 4),
                        TDR(nt.params[1]/2, 4)
                    ]),
                    2, APar, APar
                ),
                TL(nt.params[1],2)
            ]).withTags(nt.getTags())
        ]],

        apply := (nt, c, cnt) -> c[1],

        switch := false

#D        isApplicable     := (self, P) >> let(
#D                           streamsize := Cond((IsList(P[3]) and
#D                              Length(P[3])=1 and IsBound(P[3][1].bs)),
#D                                 P[3][1].bs, -1),
#D
#D                           PHasTags(self.nonTerminal, P) and P[1] >= self.minSize and
#D                   4^LogInt(P[1]/2, 4) = P[1]/2
#D                        ),
#D
#D    allChildren      := (self, P) >>
#D                              [[TCompose([
#D                     TL(P[1], 2),
#D                 TTensorI(DFT(2), P[1]/2, APar, APar),
#D                 TDiag(Tw1(P[1], 2, 1)),
#D                 TL(P[1], P[1]/2),
#D                 TTensorI(TCompose([
#D                             DFTDR(P[1]/2, 4),
#D                         TDR(P[1]/2, 4)]),
#D                     2, APar, APar),
#D                 TL(P[1],2)
#D                  ], P[3])]],
#D
#D    rule := (P, C) -> C[1],
    )
));


NewRulesFor(WHT, rec(

    ######################################################################
    # tSPL Stream WHT Termination rule
    # this might not be necessary
    # WHT(s) with s ports  -> CodeBlock(WHT(s)
    WHT_tSPL_STerm := rec(
        forTransposition := false,
        switch           := false,
        applicable       := nt ->
            nt.isTag(1, AStream)
            and 2^nt.params[1] = nt.firstTag().bs,
        children := nt -> [[ WHT(nt.params[1]) ]],
        apply := (nt, C, cnt) -> CodeBlock(C[1])
    
#D    isApplicable     := P -> IsList(P[2])        and 
#D                             Length(P[2])=1      and 
#D                 IsBound(P[2][1].bs) and 
#D                 2^P[1]=P[2][1].bs,
#D
#D    allChildren      := P -> [[WHT(P[1])]],
#D    rule             := (P, C) -> CodeBlock(C[1])
    )
));


NewRulesFor(TTensorI, rec(
# new rule: TTensorI(X, 1, x, x).withTags([AStream(s)]) --> X.withTags([AStream(s)]).  
# Oh but what about the case where size of X is s.  
# ok this rule can only be applicable when IxA_stream_base isn't.
# then I think it is ok
# also need to turn off TTensorI_Streamtag

# might be missing more, though: do i still end up sometimes with stensor(stensor( ?
# need to stop and think abstarctly on this

#   I x A streaming base case
    IxA_stream_base := rec(
        info := "IxA stream base",
        forTransposition := false,

        applicable := nt -> nt.isTag(1, AStream)
            and IsParPar(nt.params)
            and Rows(nt.params[1]) = Cols(nt.params[1])
            and Rows(nt.params[1]) <= nt.firstTag().bs
            and let(
                w := nt.firstTag().bs,
                np := Rows(nt.params[1]),
                n := Cond(IsValue(np), np.ev(), np),
                IsInt(w/n)),

        children := nt -> [[ nt.params[1] ]],

        # restriction 1: n <= block size
        # restriction 2: we assume square non-terminal
        apply := (nt, C, cnt) -> let(
            P := nt.params,
            n := Rows(P[1]), 
            m := P[2], 
            bs := nt.getTags()[1].bs,
            range := m*n/bs, 
            i := Ind(range), 
            g := StreamGath(i, range, bs), 
            s := StreamScat(i, range, bs),
            block := Cond(bs > n, Tensor(I(bs/n), C[1]), C[1]),
            Cond(ObjId(block)=CodeBlock,
                STensor(block, range, bs),
                STensor(CodeBlock(block), range, bs)
            )
        )
    ),
 
#    Streaming TTensorI -> STensor
    TTensorI_Streamtag := rec(
        switch           := false,
        forTransposition := false,

        applicable := nt ->
            nt.isTag(1, AStream)
            and IsParPar(nt.params)
            and Rows(nt.params[1]) = Cols(nt.params[1])
            and Rows(nt.params[1]) > nt.firstTag().bs
            and not TTensorI_Stream_Diag_Uneven.applicable(nt)
            and not TTensorI_Stream_Perm_Uneven.applicable(nt),

        children  := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],

        apply := (nt, C, cnt) -> STensor(C[1], nt.params[2], nt.firstTag().bs)

     ),

    # Streaming (Im x Dn).withTags([AStream(w)]), where w does not divide n, but w|(mn)
    TTensorI_Stream_Diag_Uneven := rec(
        switch           := false,
        forTransposition := false,

        applicable := nt ->
           nt.isTag(1, AStream)
            and (ObjId(nt.params[1])=TDiag or ObjId(nt.params[1])=TC)
            and IsParPar(nt.params)
            and Rows(nt.params[1]) = Cols(nt.params[1])
            and let(
                np := Rows(nt.params[1]),
                n := Cond(IsValue(np), np.ev(), np),
                m := nt.params[2],
                w := nt.firstTag().bs,
                (not IsInt(n/w)) and IsInt((m*n)/w)),

        children := nt -> [[TDiag(diagDirsum(List([1..nt.params[2]], i->nt.params[1].params[1]))).withTags(nt.getTags())]],

        apply := (nt, C, cnt) -> C[1]
    ),


    # Streaming (Im x Pn).withTags([AStream(w)]), where w does not divide n, n does not divide w, but w|(mn)
    TTensorI_Stream_Perm_Uneven := rec(
        switch           := false,
        forTransposition := false,

        applicable := nt ->
           nt.isTag(1, AStream)
            and ObjId(nt.params[1])=TPrm
            and IsParPar(nt.params)
            and Rows(nt.params[1]) = Cols(nt.params[1])
            and let(
                np := Rows(nt.params[1]),
                n := Cond(IsValue(np), np.ev(), np),
                m := nt.params[2],
                w := nt.firstTag().bs,
                (not IsInt(n/w)) and IsInt((m*n)/w) and (not IsInt(w/n))),                           

        children := nt -> [[ ]],
#        apply := (nt, C, cnt) -> C[1]
        apply := (nt, C, cnt) -> let(
            np := Rows(nt.params[1]),
            w := nt.firstTag().bs,
            n := Cond(IsValue(np), np.ev(), np),
            m := nt.params[2],
            
            # a is the biggest value such that a|m and w | m*n/a
            alist := Filtered([1..m], a -> IsInt(m/a) and IsInt((m*n/a)/w)),
            a := alist[Length(alist)],
            
            StreamPerm([TPrm(Tensor(I(m/a), nt.params[1]))], a, w, 0) 
#            [[ TPrm(Tensor(I(nt.params[2]), nt.params[1])).withTags(nt.getTags()) ]],
        )

    ),

));

NewRulesFor(TTensorInd, rec(

    # I x_j A_m, streaming width w >= m
    TTensorInd_stream_base := rec(
        info := "I x_j A_m, streaming width w >= m",
        forTransposition := false,

        applicable := nt -> nt.isTag(1, AStream)
            and IsParPar(nt.params)
            and Rows(nt.params[1]) = Cols(nt.params[1])
            and Rows(nt.params[1]) <= nt.firstTag().bs,

        children := nt -> [[ nt.params[1], InfoNt(nt.params[2]) ]],

        # restriction: we assume square non-terminal
        apply := (nt, C, cnt) -> let(
            P := nt.params,
            n := Rows(P[1]), 

            j := cnt[2].params[1],
            m := j.range,
            w := nt.getTags()[1].bs,            

            it_i := Ind(m*n/w),
            it_k := Ind(w/n),

            # if w=n we don't need substvars

            block  := Cond(w > n,
		Cond(w < m*n,
		    SubstVars(Copy(C[1]), rec((cnt[2].params[1].id) := it_k+it_i*V(w/n))),
		    SubstVars(Copy(C[1]), rec((cnt[2].params[1].id) := it_k))
		    ),
                SubstVars(Copy(C[1]), rec((cnt[2].params[1].id) := it_i))),

            blocks := Cond(w > n, CodeBlock(IDirSum(it_k, w/n, block).unroll()), CodeBlock(block)),

	    Cond(m*n/w > 1,
		SIterDirectSum(it_i, m*n/w, blocks, w),
		STensor(blocks, 1, w)
	    )
        )
    ),

    # I x_j A_m, streaming width w < m
    TTensorInd_stream_prop := rec(
        info := "I x_j A_m, streaming width w >= m",
        forTransposition := false,

        applicable := nt -> nt.isTag(1, AStream)
            and IsParPar(nt.params)
            and Rows(nt.params[1]) = Cols(nt.params[1])
            and Rows(nt.params[1]) > nt.firstTag().bs,

        children := nt -> [[ nt.params[1].withTags(nt.getTags()), InfoNt(nt.params[2]) ]],

        # restriction: we assume square non-terminal
        apply := (nt, C, cnt) -> let(
            P := nt.params,
            n := Rows(P[1]), 
            j := cnt[2].params[1],
            w := nt.getTags()[1].bs,            
            m := j.range,

            SIterDirectSum(j, m, C[1], w)
        )
    ),
 
));


NewRulesFor(TDR, rec(
#   TDR -> StreamPerm(LinearBits(TDR.permBits()))
    DR_StreamPerm := rec(
        switch := false,
        info := "Place a streaming DR into a streaming permutation wrapper",
        forTransposition := false,

        applicable := nt -> nt.isTag(1, AStream),
        children := nt -> [[ ]],
        apply := (nt, C, cnt) ->
            Cond(nt.firstTag().bs = nt.params[1],
                CodeBlock(DR(nt.params[1], nt.params[2])),
                StreamPerm([DR(nt.params[1], nt.params[2])], 1, nt.firstTag().bs, 0)
#                 StreamPermBit( 
#                     LinearBits(
#                         DR(nt.params[1], nt.params[2]).permBits(), 
#                         DR(nt.params[1], nt.params[2])
#                         ), 
#                     nt.firstTag().bs
#                 )
            )
    )
                

#D    isApplicable := P -> Length(P[3]) > 0          and 
#D                         IsBound(P[3][1].isStream) and 
#D                         P[3][1].isStream,  
#D    
#D    allChildren := P -> [[]],
#D    
#D    rule := (P, C) -> StreamPermBit(
#D                         LinearBits(
#D                    DR(P[1], P[2]).permBits()
#D                 ), 
#D                 P[3][1].bs
#D                      )
#D    ),
));

NewRulesFor(TL, rec(
#   TL -> StreamPermBit(LinearBits(TL.permBits(), TL))
    L_StreamPerm := rec(
        switch := false,
        info := "Place a streaming L into a streaming permutation wrapper",
        forTransposition := false,
    
        applicable := nt -> nt.isTag(1, AStream) and nt.firstTag().bs < nt.dims()[1],
        children := nt -> [[ ]],

        apply := (nt, C, cnt) -> let(P := nt.params, 
            StreamPerm([Tensor(L(P[1], P[2]), I(P[4]))], P[3], nt.firstTag().bs, 0))

#         apply := (nt, C, cnt) -> let(P := nt.params, StreamPermBit(
#             LinearBits(
#                 TL(P[1], P[2], P[3], P[4]).withTags(nt.getTags()).permBits(),
#                 TL(P[1], P[2], P[3], P[4])
#             ),
#             nt.firstTag().bs
#         )
#         ) 

    ),

#   TL(N,n,l,r,[]) -> I_l x L(N,n) x I_r
#
#   See: J{\"a}rvinen, Salmela, Sorokin, and Takala.  Stride Permutation 
#        Networks for Array Processors.  Proc. of IEEE Int'l Conf. on 
#        Application-Specific Systems, Architectures, and Processors (ASAP04).
    L_Stream0 := rec(
        switch := false,
        info := "Streaming L decomposition, Q > N/R",
        forTransposition := false,
        applicable := nt -> 
            nt.isTag(1, AStream) 
            and let(
                P := nt.params,
                Q := nt.firstTag().bs,
                N := P[1], 
                S := P[2], 
                R := Minimum(S, N/S), 
                Q > N/R 
                and Q < N
                and P[3] = 1
                and P[4] = 1
            ),

        apply := (nt, C, cnt) -> let(
            P := nt.params,
            N := P[1], 
            S := P[2], 
            R := Minimum(S, N/S), 
            Q := nt.firstTag().bs, 
            X := N/Q,
            L1 := When(Q = R, [], [STensor(L(Q, R), N/Q, Q)]), 
            L2 := When(X = 2, [], [STensor(L(N/Q, 2), Q, Q)]),
            L3 := When(Q/N = 1, [], [STensor(L(R, R*Q/N), N/R, Q)]),
            L4 := When(Q = Q/R, [], [STensor(L(Q, Q/R), N/Q, Q)]), 
            L5 := When(N/Q = N/(2*Q), [], [STensor(L(N/Q, N/(2*Q)), Q, Q)]),
            L6 := When(Q/N = R, [], [STensor(L(R, N/Q), N/R, Q)]),

            When(S <= N/S, 
                Compose(Concat(L1, L2, List(Reversed([2..Log2Int(N/Q)]), i-> 
                    StreamIxJ(Q*2^i, N/(Q*(2^i)), Q) *
                    StreamIxJ(2^i, N/(2^i), Q)), 
                    [
                        StreamIxJ(Q*2, N/(Q*(2)), Q)
                    ], 
                    L3
                )),
                Compose(Concat(
                    L6, 
                    [StreamIxJ(Q*2, N/(Q*(2)), Q)],
                    List([2..Log2Int(N/Q)], i-> 
                    StreamIxJ(2^i, N/(2^i), Q) * 
                    StreamIxJ(Q*2^i, N/(Q*(2^i)), Q)),
                    L5,
                    L4
                ))
            )
        )
#D        isApplicable := P -> Length(P[5]) > 0 and IsBound(P[5][1].isStream) and 
#D                         P[5][1].isStream and let(Q := P[5][1].bs, N := P[1], S := P[2], 
#D                             R := Minimum(S, N/S), Q > N/R and Q < N) and P[3] = 1 and P[4] = 1, 
#D
#D        rule := (P, C) -> 
#D                let(N := P[1], S := P[2], R := Minimum(S, N/S), Q := P[5][1].bs, X := N/Q,
#D                    
#D                        let(L1 := When(Q = R, [], [STensor(L(Q, R), N/Q, P[5][1].bs)]), 
#D                            L2 := When(X = 2, [], [STensor(L(N/Q, 2), Q, P[5][1].bs)]),
#D                            L3 := When(Q/N = 1, [], [STensor(L(R, R*Q/N), N/R, P[5][1].bs)]),
#D                            L4 := When(Q = Q/R, [], [STensor(L(Q, Q/R), N/Q, P[5][1].bs)]), 
#D                            L5 := When(N/Q = N/(2*Q), [], [STensor(L(N/Q, N/(2*Q)), Q, P[5][1].bs)]),
#D                            L6 := When(Q/N = R, [], [STensor(L(R, N/Q), N/R, P[5][1].bs)]),
#D
#D                            When(S <= N/S, 
#D                                Compose(Concat(L1, L2, List(Reversed([2..Log2Int(N/Q)]), i-> 
#D                                        StreamIxJ(Q*2^i, N/(Q*(2^i)), P[5][1].bs) *
#D                                        StreamIxJ(2^i, N/(2^i), P[5][1].bs)), 
#D                                        [StreamIxJ(Q*2, N/(Q*(2)), P[5][1].bs)], L3)),
#D                                Compose(Concat(
#D                                        L6, 
#D                                        [StreamIxJ(Q*2, N/(Q*(2)), P[5][1].bs)],
#D                                        List([2..Log2Int(N/Q)], i-> 
#D                                        StreamIxJ(2^i, N/(2^i), P[5][1].bs) * 
#D                                        StreamIxJ(Q*2^i, N/(Q*(2^i)), P[5][1].bs)),
#D                                        L5,
#D                                        L4))
#D                                
#D                                )
#D                        
#D                        )
#D                )
     ),

#   TL(N,n,l,r,[]) -> I_l x L(N,n) x I_r
#
#   See: J{\"a}rvinen, Salmela, Sorokin, and Takala.  Stride Permutation 
#        Networks for Array Processors.  Proc. of IEEE Int'l Conf. on 
#        Application-Specific Systems, Architectures, and Processors (ASAP04).
    L_Stream1 := rec(
    switch := false,
        info := "Streaming L decomposition, Q <= N/R and Q >= R",
        forTransposition := false,
        applicable := nt -> nt.isTag(1, AStream) and let(
            P := nt.params,
            Q := nt.firstTag().bs,
            N := nt.params[1], 
            S := nt.params[2], 
            R := Minimum(S, N/S), 
            Q <= N/R 
            and Q >= R
            and P[3] = 1 
            and P[4] = 1
        ), 
        apply := (nt, C, cnt) -> let(
            P := nt.params,
            N := P[1], 
            S := P[2], 
            R := Minimum(S, N/S), 
            Q := nt.firstTag().bs, 
            L1 := When((N/Q) = R, [], [SIxLxI(N/Q, R, Q, 1, Q)]), 
            L2 := When(Q = R, [], [STensor(L(Q, R), N/Q, Q)]),
            L3 := When(R = 2, [], [STensor(L(R, 2), N/R, Q)]),
            L4 := When((N/Q) = R, [], [SIxLxI(N/Q, N/(Q*R), Q, 1, Q)]), 
            L5 := When(Q = R, [], [STensor(L(Q, Q/R), N/Q, Q)]),
            L6 := When(R = 2, [], [STensor(L(R, R/2), N/R, Q)]),
            When(S < N/S, 
                Compose(Concat(L1, L2, L3, List(Reversed([2..Log2Int(R)]), i-> 
                    StreamIxJ(Q*2^i, N/(Q*(2^i)), Q)
                    * StreamIxJ(2^i, N/(2^i), Q)), 
                    [StreamIxJ(Q*2, N/(Q*(2)), Q)]
                )),
                Compose(Concat(
                    [StreamIxJ(Q*2, N/(Q*(2)), Q)],
                    List([2..Log2Int(R)], i-> 
                        StreamIxJ(2^i, N/(2^i), Q)
                        * StreamIxJ(Q*2^i, N/(Q*(2^i)), Q)
                    ),
                    L6, L5, L4
                ))
            )
        )
#D        isApplicable := P -> Length(P[5]) > 0 and 
#D                             IsBound(P[5][1].isStream) and 
#D                             P[5][1].isStream and 
#D                             let(Q := P[5][1].bs, N := P[1], S := P[2], R := Minimum(S, N/S), 
#D                                Q <= N/R) and 
#D                             let(Q := P[5][1].bs, N := P[1], S := P[2], R := Minimum(S, N/S), Q >= R) and
#D                             P[3] = 1 and P[4] = 1, 
#D        rule := (P, C) -> 
#D                let(N := P[1], S := P[2], R := Minimum(S, N/S), Q := P[5][1].bs, 
#D                    
#D                        let(L1 := When((N/Q) = R, [], [SIxLxI(N/Q, R, Q, 1, P[5][1].bs)]), 
#D                            L2 := When(Q = R, [], [STensor(L(Q, R), N/Q, P[5][1].bs)]),
#D                            L3 := When(R = 2, [], [STensor(L(R, 2), N/R, P[5][1].bs)]),
#D                            L4 := When((N/Q) = R, [], [SIxLxI(N/Q, N/(Q*R), Q, 1, P[5][1].bs)]), 
#D                            L5 := When(Q = R, [], [STensor(L(Q, Q/R), N/Q, P[5][1].bs)]),
#D                            L6 := When(R = 2, [], [STensor(L(R, R/2), N/R, P[5][1].bs)]),
#D 
#D                            
#D                            When(S < N/S, 
#D                                Compose(Concat(L1, L2, L3, List(Reversed([2..Log2Int(R)]), i-> 
#D                                        StreamIxJ(Q*2^i, N/(Q*(2^i)), P[5][1].bs) *
#D                                        StreamIxJ(2^i, N/(2^i), P[5][1].bs)), 
#D                                        [StreamIxJ(Q*2, N/(Q*(2)), P[5][1].bs)])),
#D                                Compose(Concat(
#D                                        [StreamIxJ(Q*2, N/(Q*(2)), P[5][1].bs)],
#D                                        List([2..Log2Int(R)], i-> 
#D                                        StreamIxJ(2^i, N/(2^i), P[5][1].bs) * 
#D                                        StreamIxJ(Q*2^i, N/(Q*(2^i)), P[5][1].bs)),
#D                                        L6, L5, L4))
#D                                
#D                                )
#D                        
#D                        )
#D                )
     ),

#   TL(N,n,l,r,[]) -> I_l x L(N,n) x I_r
#
#   See: J{\"a}rvinen, Salmela, Sorokin, and Takala.  Stride Permutation 
#        Networks for Array Processors.  Proc. of IEEE Int'l Conf. on 
#        Application-Specific Systems, Architectures, and Processors (ASAP04).
    L_Stream2 := rec(
        switch := false,
        info := "Streaming L decomposition, Q <= N/R and Q < R",
        forTransposition := false,
        applicable := nt ->
            nt.isTag(1, AStream)
            and let(
                P := nt.params,
                Q := nt.firstTag().bs, 
                N := P[1], 
                S := P[2], 
                R := Minimum(S, N/S), 
                Q <= N/R
                and Q < R
                and P[3] = 1 
                and P[4] = 1
            ), 
        apply := (nt, C, cnt) -> let(
            P := nt.params,
            N := P[1], 
            S := P[2], 
            R := Minimum(S, N/S), 
            Q := nt.firstTag().bs, 
            L1 := When((N/Q) = R, [], [SIxLxI(N/Q, R, Q, 1, Q)]), 
            L2 := When(Q = 2, [], [STensor(L(Q, 2), N/Q, Q)]),
            L3 := When(R = R/Q, [], [SIxLxI(R, R/Q, Q, N/(R*Q), Q)]),
            L4 := When((N/Q) = (N/(Q*R)), [], [SIxLxI(N/Q, N/(Q*R), Q, 1, Q)]), 
            L5 := When(Q = Q/2, [], [STensor(L(Q, Q/2), N/Q, Q)]),
            L6 := When(R = Q, [], [SIxLxI(R, Q, Q, N/(R*Q), Q)]),
            When(S < N/S, 
                Compose(Concat(
                    L1, 
                    L2, 
                    List(Reversed([2..Log2Int(Q)]), i-> 
                        StreamIxJ(Q*2^i, N/(Q*(2^i)), Q)
                        * StreamIxJ(2^i, N/(2^i), Q)), 
                    [StreamIxJ(Q*2, N/(Q*(2)), Q)], 
                    L3
                )),
                Compose(Concat(
                    L6,
                    [StreamIxJ(Q*2, N/(Q*(2)), Q)],
                    List([2..Log2Int(Q)], i-> 
                        StreamIxJ(2^i, N/(2^i), Q) * 
                        StreamIxJ(Q*2^i, N/(Q*(2^i)), Q)),
                    L5, L4
                ))
            )
        )
#D        isApplicable := P -> Length(P[5]) > 0 and 
#D                             IsBound(P[5][1].isStream) and 
#D                             P[5][1].isStream and 
#D                             let(Q := P[5][1].bs, N := P[1], S := P[2], R := Minimum(S, N/S), 
#D                                Q <= N/R) and 
#D                             let(Q := P[5][1].bs, N := P[1], S := P[2], R := Minimum(S, N/S), Q < R) and
#D                             P[3] = 1 and P[4] = 1, 
#D        rule := (P, C) -> 
#D                let(N := P[1], S := P[2], R := Minimum(S, N/S), Q := P[5][1].bs, 
#D                    
#D                        let(L1 := When((N/Q) = R, [], [SIxLxI(N/Q, R, Q, 1, P[5][1].bs)]), 
#D                            L2 := When(Q = 2, [], [STensor(L(Q, 2), N/Q, P[5][1].bs)]),
#D                            L3 := When(R = R/Q, [], [SIxLxI(R, R/Q, Q, N/(R*Q), P[5][1].bs)]),
#D                            L4 := When((N/Q) = (N/(Q*R)), [], [SIxLxI(N/Q, N/(Q*R), Q, 1, P[5][1].bs)]), 
#D                            L5 := When(Q = Q/2, [], [STensor(L(Q, Q/2), N/Q, P[5][1].bs)]),
#D                            L6 := When(R = Q, [], [SIxLxI(R, Q, Q, N/(R*Q), P[5][1].bs)]),
#D
#D
#D                            When(S < N/S, 
#D                                Compose(Concat(L1, L2, List(Reversed([2..Log2Int(Q)]), i-> 
#D                                        StreamIxJ(Q*2^i, N/(Q*(2^i)), P[5][1].bs) *
#D                                        StreamIxJ(2^i, N/(2^i), P[5][1].bs)), 
#D                                        [StreamIxJ(Q*2, N/(Q*(2)), P[5][1].bs)], L3)),
#D                                Compose(Concat(
#D                                        L6,
#D                                        [StreamIxJ(Q*2, N/(Q*(2)), P[5][1].bs)],
#D                                        List([2..Log2Int(Q)], i-> 
#D                                            StreamIxJ(2^i, N/(2^i), P[5][1].bs) * 
#D                                            StreamIxJ(Q*2^i, N/(Q*(2^i)), P[5][1].bs)),
#D                                        L5, L4)))
#D                        )
#D                )
    ),

    L_StreamBase := rec(
        forTransposition := false,
        applicable := nt -> (nt.isTag(1, spiral.paradigms.stream.AStream) and nt.firstTag().bs = nt.dims()[1]),
        apply := (nt, c, cnt) -> let( 
            c1 := When(nt.params[3]=1, [], [I(nt.params[3])]),
            c2 := When(nt.params[4]=1, [], [I(nt.params[4])]),
            CodeBlock(Tensor(Concat(c1, [ L(nt.params[1], nt.params[2]) ], c2)))
        )
    )

));


NewRulesFor(DFT, rec(
       
    ## This rule is for the special case where one has a DFT(2) with a streaming width of 1.
    DFT_Stream1_Fold := rec(
        forTransposition := false,

        applicable := nt -> 
            nt.isTag(1, AStream) 
            and nt.firstTag().bs = 1 
            and nt.params[1] = 2,

        children := nt -> [[ ]],
        apply := (nt, C, cnt) -> Stream1DFT2(),

   )
));

NewRulesFor(TDiag, rec(

    ## Generic rule to deal with a streaming diagonal.
    TDiag_tag_stream := rec(
        switch       := false,
        applicable := nt ->
            nt.isTag(1, AStream) and
            (ObjId(nt.params[1]) <> TC or 
                (not IsTwoPower(nt.params[1].params[1]))
            ),

        children := nt -> [[ let(
            P := nt.params,
#            size  := P[1].params[1], 
            size  := Cond(IsValue(Diag(P[1]).dims()[1]), Diag(P[1]).dims()[1].ev(), Diag(P[1]).dims()[1]),

            width := nt.firstTag().bs, 
            itr   := Ind(size/width), 

            pre_term := which -> fPrecompute(fCompose(P[1], fTensor(fId(size/width), fBase(width, which)))),
            
            table_term := fCompose(diagDirsum(List([0..width-1], i->pre_term(i))), L(size, size/width)),

#            table_term := which -> fPrecompute(fCompose(P[1], fTensor(fId(size/width), fBase(width, which)))),
            block_func := fTensor(fBase(size/width, itr), fId(width)),

            SIterDirectSum(itr, size/width, 
                CodeBlock(Diag(fCompose(table_term, block_func))),
                width
            )
        
#             SIterDirectSum(itr, size/width, 
#                 CodeBlock(Diag(
#                     fCompose(
#                         diagDirsum(List([0..width-1], table_term(itr))),
#                         L(size, size/width),
#                         block_func
#                     )
#                 )), 
#                 width
#             )

        )]],

        apply := (nt, C, cnt) -> C[1]

     ),


    TDiag_TC_Pease := rec(
        info             := "Streaming TDiag(TC)",
        switch           := false,
        forTransposition := false,

        applicable := nt -> 
            ObjId(nt.params[1]) = TC and
            nt.isTag(1, AStream) and
            IsTwoPower(nt.params[1].params[1]),
         
        children := nt -> [[ let(
            P     := nt.params,
            size  := P[1].params[1], 
            width := nt.firstTag().bs, 
            itr   := Ind(size/width),
            radix := P[1].params[2], 
            it    := P[1].params[3],
           
            cond_term   := diagTensor(II(size/radix), II(radix, 0, 1)),
            const_term  := fConst(TComplex, size, 1),
            table_term  := fPrecompute(TCBase(size, radix, P[1].params[4])),
            access_func := fAccPease(size, radix, it),
            sub_func    := fCompose(fSub(size, size-size/radix, size/radix), access_func),
            block_func  := fTensor(fBase(size/width, itr), fId(width)),

            SIterDirectSum(itr, size/width, 
                Diag(fCompose(
                    fCond(cond_term, const_term, fCompose(table_term, sub_func)),
                    block_func
                )),
                width
            )
        )]],
    
        apply := (nt, C, cnt) -> C[1]

    ),

    # NB: I don't think I need this rule anymore.  I believe that it actually has the exact
    # same behavior as TDiag_tag_stream.  (PM 1/13/10)
    TDiag_TI_It := rec(
        info             := "Streaming TDiag(TI)",
        switch           := false,
        forTransposition := false,

        applicable := nt -> 
            ObjId(nt.params[1]) = TI
            and nt.isTag(1, AStream),
         
        children := nt -> [[ let(
            P     := nt.params,
            radix := P[1].params[2], 
            itr   := P[1].params[3], 
            size  := P[1].params[1], 
            lsize := size/(radix^itr),
            width := nt.firstTag().bs,
            it    := Ind(lsize/width),


            pre_term    := which -> fPrecompute(fCompose(
                                       P[1], 
                                       fTensor(fId(lsize/width), fBase(width, which)))),

            table_term  := fCompose(diagDirsum(List([0..width-1], i-> pre_term(i))),
                                    L(lsize, lsize/width)),
            block_func  := fTensor(fBase(lsize/width, it), fId(width)),

            SIterDirectSum(it, lsize/width,
                Diag(fCompose(
                    table_term,
                    block_func
                )), 
                width
            ) 
        )]],
    
        apply := (nt, C, cnt) -> C[1]
    ),

    TDiag_TIFold_It := rec(
        info             := "Streaming TDiag(TIFold)",
        switch           := false,
        forTransposition := false,

        applicable := nt -> 
            ObjId(nt.params[1]) = TIFold
            and nt.isTag(1, AStream),
         
        children := nt -> [[ let(
            P     := nt.params,
            size  := P[1].params[1], 
            width := nt.firstTag().bs,
            radix := P[1].params[2], 
            itr   := P[1].params[3], 
            it    := Ind(size/((radix^itr)*width)),

            table_term := fPrecompute(TCBase(size/(radix^itr), radix)),
            sub_func   := fSub(size/(radix^itr), size/(radix^(itr+1)), size/(radix^(itr+1))),
            block_func := fTensor(fBase(size/((radix^itr)*width), it), fId(width)),

            SIterDirectSum(it, size/((radix^itr)*width), 
                CodeBlock(
                    Diag(fCompose(
                        table_term, 
                        sub_func,
                        block_func
                    ))
                ), 
                width
            )
        )]],
    
        apply := (nt, C, cnt) -> C[1]

    ),

    TDiag_base := rec(
        forTransposition := false,
        applicable := nt -> not nt.hasTags(),
        apply := (nt, C, cnt) -> Diag(fPrecompute(nt.params[1]))
    )
));

                 

########################################################################
#   TRC rules
NewRulesFor(TRC, rec(


#   Streaming TRC base case
    TRC_stream := rec(
        info             := "TRC stream",
        forTransposition := false,
        applicable       := nt -> nt.isTag(1, AStream),
        children         := nt -> [[ nt.params[1].withTags(
            [ AStream(nt.firstTag().bs / 2) ]
        )]],

        apply := (nt, C, cnt) -> RC(C[1]),

        switch := false
    )
));


NewRulesFor(TPrmMulti, rec(

    # TPrmMulti([P0, P1, ...]).withTags([AStream(w)]) --> StreamPerm([P0, P1, ...], 1, w)
    TPrmMulti_stream := rec(
        info             := "Streaming permutation",
        forTransposition := false,
        
        applicable := nt -> nt.isTag(1, AStream),

        children := nt -> [[ ]],

        apply := (nt, C, cnt) -> let(
            p := nt.params[1],
            w := nt.firstTag().bs,
            StreamPerm(p, 1, w, nt.params[2])
        ),
                            
    ),
        

));




############################################################
# TPrm rules

NewRulesFor(TPrm, rec(

    # TPrm(P).withTags([AStream(w)]) --> StreamPerm(p, 1, w)
    TPrm_stream := rec(
        info             := "Streaming permutation",
        forTransposition := false,
        
        applicable := nt -> nt.isTag(1, AStream),

        children := nt -> [[ ]],

        apply := (nt, C, cnt) -> let(
            p := nt.params[1],
            w := nt.firstTag().bs,
            StreamPerm([p], 1, w, 0)
        ),
                            
    ),
       
    # TPrm(P) no streaming tag --> CodeBlock(P)
    TPrm_flat := rec(
        info             := "Permutation, not streaming",
        forTransposition := false,
        
        applicable := nt -> not nt.isTag(1, AStream),

        children := nt -> [[ ]],

#  I think sometimes I need the CB, sometimes I don't.  I need to figure 
#  this out.
#        apply := (nt, C, cnt) -> CodeBlock(nt.params[1])
        apply := (nt, C, cnt) -> nt.params[1]
                            
    ),


    # TPrm(P).withTags([AStream(w)]), where P streamed with w is representable as a bit matrix
    # See: - Pueschel, Milder, Hoe.  "Permuting Streaming Data Using RAMs."  Journal of the
    #           ACM, 2009.
    # NB: Deprecated.  Use the TPrm_stream rule.
    TPrm_stream_bit_perm := rec(
        info             := "Streaming permutation, linear on bit representation",
        forTransposition := false,

        applicable := nt -> nt.isTag(1, AStream) and 
                            PermMatrixToBits(MatSPL(nt.params[1])) <> -1 and
                            2 ^ Log2Int(nt.getTags()[1].bs) = nt.getTags()[1].bs,

        children := nt -> [[ ]],

        apply := (nt, C, cnt) -> let(
            p := nt.params[1],
            w := nt.firstTag().bs,
            l := LinearBits(PermMatrixToBits(MatSPL(p)), p),
            StreamPermBit(l, w)
        ),

    ),

    # TPrm(P).withTags([AStream(w)]), where P streamed with w is not representable as a bit matrix
    # See: Milder, Hoe, Pueschel.  "Automatic Generation of Streaming Datapaths for Arbitrary 
    #       Fixed Permtuations."  Design, Automation and Test in Europe, 2009.
    # NB: Deprecated.  Use the TPrm_stream rule.
    TPrm_stream_not_bit_perm := rec(
        info             := "Streaming permutation, not linear on bit representation",
        forTransposition := false,
        
        applicable := nt -> nt.isTag(1, AStream) and
                            IsInt(nt.params[1].dimensions[1] / nt.getTags()[1].bs) and
                            PermMatrixToBits(MatSPL(nt.params[1])) = -1,

        children := nt -> [[ ]],

        apply := (nt, C, cnt) -> let(
            p := nt.params[1],
            w := nt.firstTag().bs,
            StreamPermGen(p, w)
        ),
                            
    ),

));


##################################################
# This is an ugly hack to use RulesFuncSimp without using
#   two specific rules (ComposePrecompute, RCDataPrecompute).
 
Class(KillComposePrecompute, RuleSet);

RewriteRules(KillComposePrecompute, rec(
     ComposePrecompute := Rule(@(1, DFT).cond(x->false), i->I(1)),
));

SumRulesFuncSimp := MergedRuleSet(RulesFuncSimp, KillComposePrecompute);

StreamRulesRC := RulesRC;

# end ugly hack
##################################################



Class(RulesStreamCleanup, RuleSet);

RewriteRules(RulesStreamCleanup, rec(
    ## SIterDirectSum(i, m, A, w), where A does not depend on i
    ##        --> STensor(A, m, w)

#! NB: This is commented out as a workaround for a bug we have in sorting, when w=1.
#!    SIterDirectSumNoIndex := Rule(
#!        @(1, SIterDirectSum, i -> not i.var in i.child(1).free()), z ->
#!            STensor(@(1).val.child(1), @(1).val.var.range, @(1).val.bs)),

));

Class(RulesStream, StreamRulesRC);

RewriteRules(RulesStream, rec(    

    RemoveGrp := Rule([@(1, Grp), @(2)], z->z.child(1)),

    # SIterDirectSum(i, m, SIterDirectSum(j, k, A(i,j))) --> SIterDirectSum(l, k*m, A(idiv(l,k), imod(l,k)))
    # (only makes sense when k is a two power)
     SIterDirectSumSIterDirectSumRule := Rule(
        [@(1, SIterDirectSum), @(2, SIterDirectSum, z->z.var.range = 2^Log2Int(z.var.range))], z-> let(
            i := @(1).val.var,
            m := @(1).val.var.range,
            w := @(1).val.bs,
            j := @(2).val.var,
            k := @(2).val.var.range,
            A := @(2).val.child(1),
            l := Ind(k*m),
            newI := idiv(l,k),
            newJ := imod(l,k),
            newA := SubstVars(Copy(A), rec((i.id) := newI)),
            newA2 := SubstVars(newA, rec((j.id) := newJ)),
            
            SIterDirectSum(l, k*m, newA2, w)
        )
    ),


    # SIterDirectSum(i, r, StreamPerm(p, k, w)) --> StreamPerm(DirectSum(Tensor(I(k), p)), 1, w)
    SIterDirectSumStreamPerm := Rule(
        [@(1, SIterDirectSum), @(2, StreamPerm)], i ->
           let(
               c := @(1).val.unroll().children(),
               k := @(2).val.child(2),
# old               l := List([1..Length(c)], a->Tensor(I(k), c[a].child(1))),

               # c is the list of children of the direct sum
               # a iterates over each of the multiple perms
               # b iterates over c
               l2 := List([1..Length(c[1].child(1))], a -> List(c, b -> Tensor(I(k), b.child(1)[a]))),
               l3 := List(l2, a -> DirectSum(a)),


               # If the DirSum's index is a free variable in the perm, then we need to
               # use the direct sum.  Otherwise, we can just use StreamPerm's built-in tensor
               # capability.
               Cond(@(1).val.var in @(2).val.free(),
                   StreamPerm(l3, 1, @(2).val.child(3), @(2).val.child(4)),
                   StreamPerm(@(2).val.child(1), @(1).val.var.range * k, @(1).val.bs, @(2).val.child(4))
               )

           )
    ),

    # STensor(StreamPerm(p, k, w), l, w) --> StreamPerm(p, k*l, w)
    STensorStreamPerm := Rule(
       [@(1, STensor), @(2, StreamPerm), @(3), @(4)], i ->
              StreamPerm(@(2).val.child(1), @(1).val.p * @(2).val.child(2), @(1).val.bs, @(2).val.child(4))
   ),


    # RC(StreamPerm) --> RCStreamPerm
    RCStreamPermRule := Rule(
        [@(1, RC), @(2, StreamPerm)], i ->
           RCStreamPerm(@(2).val.child(1), @(2).val.par, @(2).val.streamSize, @(2).val.it)
    ),

    # SIterDirectSum(i, r, RCStreamPerm(p, k, w)) --> RCStreamPerm(DirectSum(Tensor(I(k), p)), 1, w)
    # Essentially the same as SIterDirectSumStreamPerm rule above.
    SIterDirectSumRCStreamPerm := Rule(
        [@(1, SIterDirectSum), @(2, RCStreamPerm)], i ->
           let(
               c := @(1).val.unroll().children(),
               k := @(2).val.par,
               l := List([1..Length(c)], a->Tensor(I(k), c[a].child(1))),

               l2 := List([1..Length(c[1])], a -> List(c, b -> Tensor(I(k), b[1]))),
               l3 := List(l2, a -> DirectSum(a)),

               Cond(@(1).val.var in @(2).val.free(),
                   RCStreamPerm(l3, 1, @(1).val.bs, @(2).val.it),
                   RCStreamPerm(@(2).val.child(1), @(1).val.var.range * k, @(1).val.bs, @(2).val.it)
               )

           )
    ),

    # STensor(RCStreamPerm(p, k, w), l, w) --> RCStreamPerm(p, k*l, w)
    STensorRCStreamPerm := Rule(
       [@(1, STensor), @(2, RCStreamPerm), @(3), @(4)], i ->
              RCStreamPerm(@(2).val.child(1), @(1).val.p * @(2).val.par, @(1).val.bs/2, @(2).val.it)
    ),


        
    # This rule converts SIterDirectSum(i, r, Compose([c0, c1, ... ], w)) to
    #      Compose(SIterDirectSum(i0, r, c0, w), SIterDirectSum(i1, r, c1, w), ...

    SIterDirectSumComposeRule := Rule(
       [@(1, SIterDirectSum), @(2, Compose)],  i ->
        
        let(
            oldvar := @(1).val.var,
            rng    := oldvar.range,
            w      := @(1).val.bs,
            newv   := List([1..Length(@(2).val.children())], c->Ind(rng)),
            
            terms  := List([1..Length(@(2).val.children())], c->
                SIterDirectSum(newv[c], rng, 
                    SubstVars(Copy(@(2).val.child(c)), rec((oldvar.id) := newv[c])), w)),
            
            Compose(terms)
        )
    ),

    # STensor(Compose([c0, c1, ...], r, w) --> Compose(STensor(c0, r, w), ...)
    STensorComposeRule := Rule(
        [@(1, STensor), @(2, Compose, z->
((ObjId(z.child(1)) <> StreamPadDiscard) and 
            (ObjId(z.child(1)) <> RC) or ObjId(z.child(1).child(1)) <> StreamPadDiscard)), @(3), @(4)], i ->
        Compose(List([1..Length(@(2).val.children())], c ->
            STensor(@(2).val.child(c), @(1).val.p, @(1).val.bs)))
    ),

    ## StreamPerm(P(n/m), m, w) * StreamPerm(Q(n/k), k, w) --> 
    ##        StreamPerm(Tensor(I(m/l), P) * Tensor(I(k/l), Q), l, w), where l = min(m,k)
     ComposeStreamPermRule := ARule(Compose, [@(1, StreamPerm), @(2, StreamPerm, z -> z.it=@(1).val.it)],
        x -> let(
            m := @(1).val.child(2),
            k := @(2).val.child(2),
            l := Minimum(m,k),
            P := @(1).val.child(1),
            Q := @(2).val.child(1),
            w := @(1).val.streamSize,

            prms := List([1..Length(P)], i -> Tensor(I(m/l), P[i]) * Tensor(I(k/l), Q[i])),
            [ StreamPerm(prms, l, w, @(1).val.it) ]

#            [StreamPerm([Tensor(I(m/l), P[1]) * Tensor(I(k/l), Q[1])], l, w, 0)]
    )),


    ## RCStreamPerm(P(n/m), m, w) * RCStreamPerm(Q(n/k), k, w) --> 
    ##        RCStreamPerm(Tensor(I(m/l), P) * Tensor(I(k/l), Q), l, w), where l = min(m,k)
     ComposeRCStreamPermRule := ARule(Compose, [@(1, RCStreamPerm), @(2, RCStreamPerm, z -> z.it=@(1).val.it)],
        x -> let(
            m := @(1).val.par,
            k := @(2).val.par,
            l := Minimum(m,k),
            P := @(1).val.child(1),
            Q := @(2).val.child(1),
            w := @(1).val.streamSize,

            prms := List([1..Length(P)], i -> Tensor(I(m/l), P[i]) * Tensor(I(k/l), Q)),

            [ RCStreamPerm( prms, l, w, @(1).val.it) ]

#            [RCStreamPerm([Tensor(I(m/l), P) * Tensor(I(k/l), Q)], l, w, 0)]
    )),
    

     # StreamPerm(P(n/m), m, w) * RCStreamPerm(Q(n/(2*k)), k, w), if m <= k--> StreamPerm(P * Tensor(I(k/m), Q, I(2)), m, w)
     # The condition on m <= k is so that we don't replace a very small "non RC" perm with a big general one.
     # This condition might need to be tweaked.
     ComposeStreamPermRCStreamPermRule := ARule(Compose, [@(1, StreamPerm), @(2, RCStreamPerm, i->i.par >= @(1).val.par and i.ir = @(1).val.it)],
         x -> let(
             m := @(1).val.par,
             k := @(2).val.par,
             P := @(1).val.child(1),
             Q := @(2).val.child(1),
             w := @(1).val.streamSize,

             prms := List([1..Length(P)], i -> P[i] * Tensor(I(k/m), Q[i], I(2))),

             [ StreamPerm(prms, m, w, @(1).val.it) ]
             
#             [StreamPerm(P * Tensor(I(k/m), Q, I(2)), m, w)]
     )),
     
     # NOTE: Replicate above rule with terms flipped
        


##    STensor(STensor(A, i, w), j, w) --> STensor(A, i*j, w)
    STensorSTensorRule := Rule(
	[@(1, STensor), @(2, STensor), @(3), @(4)], i ->
       STensor(@(2).val.child(1), @(1).val.p * @(2).val.p, @(1).val.bs)),
    

    ## This rule is an ugly rule to deal with the stream 1 DFT2
    STensorRCStream1DFT2Rule := Rule(
    [@(1, STensor), [RC, Stream1DFT2], @(3), @(4)],
    
    x-> StreamPerm([L(4,2)], @(1).val.p, 2, 0) *
        STensor(CodeBlock(F(2)), 2*@(1).val.p, 2) * 
	StreamPerm([L(4,2)], @(1).val.p, 2, 0)

#   x ->
#         StreamPermBit(
#             LinearBits(
#                 TL(4, 2, @(1).val.p, 1).permBits(), 
#                 TL(4, 2, @(1).val.p, 1)
#             ), 
#         2) * 
#         STensor(CodeBlock(F(2)), 2*@(1).val.p, 2) * 
#         StreamPermBit(
#             LinearBits(
#                 TL(4,2, @(1).val.p, 1).permBits(),
#                 TL(4,2, @(1).val.p, 1)
#             ), 
#         2)


    ),


    ## Overwriting the rcdiag rule in sigma/rc_rules.gi
    ## RC(Diag(D)) -> RCDiag((fReal(D) + fImag(D)) o L(2n, n))
    RC_Diag := Rule([RC, @(1,Diag)], 

    e -> RCDiag(fCompose(
        diagDirsum(fReal(@(1).val.element), fImag(@(1).val.element)), 
            L(@(1).val.dimensions[1]*2, @(1).val.dimensions[1])))
    ),

    STensor_one := Rule([@(1, STensor).cond(e -> e.p = 1), @(2)],
    e -> @(2).val),



    ComposeStreamPermRuleNew := ARule(Compose, [@(1, StreamPermBit), @(2, StreamPermBit)],
        x -> [StreamPermBit(LinearBits(@(1).val.child(1).child(1) * @(2).val.child(1).child(1), @(1).val.child(1).child(2) * @(2).val.child(1).child(2)), @(1).val.streamSize)]),


    Stensor_perm_compat := ARule(Compose, [[@(1, STensor), [RC, @(2, StreamPermBit)], @(5), @(6)], 
                                           [@(3, STensor).cond(x -> x.p = @(1).val.p),
                          [RC, @(4, StreamPermBit)], @(7), @(8)]],
    e -> [STensor(RC(StreamPermBit(LinearBits(@(2).val.child(1).child(1) * @(4).val.child(1).child(1)),
                           @(2).val.streamSize)), @(1).val.p, @(1).val.bs)]),

    ComposeStreamPermRule2 := ARule(Compose, [[@(4, RC), @(1, StreamPermBit)], [@(5, RC), [@(2, STensor), @(3, StreamPermBit), @(6), @(7)]]], 
    x->[RC(StreamPermBit(LinearBits(@(1).val.child(1).child(1) * AddTopLeftOnes(@(3).val.child(1).child(1), Log2Int(@(2).val.dimensions[1]))), 
         @(1).val.streamSize))]),

    ComposeRCStreamPermCompose := ARule(Compose, [[@(4, RC), @(1, StreamPermBit)], 
        [@(2, STensor), [@(3, Compose).cond(e -> ObjId(e.children()[1]) = RC and ObjId(e.children()[1][1]) = StreamPermBit)], @(5), @(6)]], 

    x -> let(ch := @(3).val.children(),
        [RC(StreamPermBit(LinearBits(@(1).val.child(1).child(1) * AddTopLeftOnes(ch[1].child(1).child(1).child(1), Log2Int(@(2).val.dimensions[1]))),
          @(1).val.streamSize)) * STensor(Compose(Drop(ch, 1)), @(2).val.p, @(1).val.streamSize)
         ])),


    ComposeStreamPermRule4 := ARule(Compose, [[@(2, STensor), [@(5, RC), @(3, StreamPermBit)], @(6), @(7)], [@(4, RC), @(1, StreamPermBit)]], 
    x-> let( totalBits := Log2Int(@(2).val.dimensions[1]),
             incrBits  := totalBits - Log2Int(@(5).val.dimensions[1]),
#             bits0 := AddTopLeftOnes(@(3).val.child(1).child(1), Log2Int(@(3).val.dimensions[1])),
             bits0 := AddTopLeftOnes(@(3).val.child(1).child(1), incrBits),
             bits1 := @(1).val.child(1).child(1),
             bits  := bits0 * bits1,
             p     := @(4).val * @(2).val,

             [ RC(StreamPermBit(LinearBits(bits, p), @(1).val.streamSize))])),


#    ComposeStreamPermRule3 := ARule(Compose, [[@(4, RC), @(1, StreamPermBit)], [@(5, STensor), [@(2, RC), @(3, StreamPermBit)]]], 
#    x-> let( bits0 := @(1).val.child(1).child(1),
#             bits1 := AddTopLeftOnes(@(3).val.child(1).child(1), Log2Int(@(3).val.dimensions[1])),
#             bits  := bits0 * bits1,
#             p     := @(4).val * @(5).val,
#
#             [ RC( StreamPermBit( LinearBits( bits, p), @(1).val.streamSize))])),



#!  NOTE (?)       
  RCData_fCompose := Rule([RCData, @(1, fCompose).cond(e -> IsInt(fCompose(Drop(e.children(), 1)).range()))],
      e -> let(ch := @(1).val.children(),
        fCompose(RCData(ch[1]), fTensor(fCompose(Drop(ch, 1)), fId(2))))),


  fReal_fCompose := Rule([fReal, @(1, fCompose).cond(e -> IsInt(fCompose(Drop(e.children(), 1)).range()))],
      e -> let(ch := @(1).val.children(),
         fCompose(fReal(ch[1]), Drop(ch, 1))
      )
  ),

  fImag_fCompose := Rule([fImag, @(1, fCompose).cond(e -> IsInt(fCompose(Drop(e.children(), 1)).range()))],
      e -> let(ch := @(1).val.children(),
         fCompose(fImag(ch[1]), Drop(ch, 1))
#        fCompose(fImag(ch[1]), Drop(ch, 1), fTensor(fId(@(1).val.domain()), fBase(2,1)))
      )
  ),

  fRealImag_fCond := Rule([@(1).cond(e -> ObjId(e) = fReal or ObjId(e) = fImag), @(2, fCond)],
      e -> Cond(ObjId(@(1).val) = fReal,
            fCond(@(2).val.params[1], fReal(@(2).val.params[2]), fReal(@(2).val.params[3])),
            fCond(@(2).val.params[1], fImag(@(2).val.params[2]), fImag(@(2).val.params[3]))
      )
  ),
                  

  fReal_diagDirsum := Rule([fReal, @(1, diagDirsum)],
      e -> let(ch := @(1).val.children(),
           diagDirsum(List([1..Length(ch)], i -> fReal(ch[i]))))
  ),

  fImag_diagDirsum := Rule([fImag, @(1, diagDirsum)],
      e -> let(ch := @(1).val.children(),
           diagDirsum(List([1..Length(ch)], i -> fImag(ch[i]))))
  ),

   fReal_fPrecompute := Rule([fReal, [fPrecompute, @(1)]],
       e -> fPrecompute(fReal(@(1).val))),

   fImag_fPrecompute := Rule([fImag, [fPrecompute, @(1)]],
       e -> fPrecompute(fImag(@(1).val))),

 
    RCData_fCondRule := Rule([RCData, @(1, fCond)],
    x -> fCond(diagTensor(@(1).val.params[1], fConst(TInt, 2, 1)), RCData(@(1).val.params[2]), 
         RCData(@(1).val.params[3]))),

    RCSTensorIRule := Rule([@(1, STensor), @(2, I), @(3), @(4)], x -> I(@(2).val.params[1] * @(1).val.p)),

    ITensorIRule := Rule([Tensor, @(1, I), @(2, I)], x -> I(@(1).val.params[1] * @(2).val.params[1])),

    RCIComposeRule := Rule([RC, [@(1, ICompose), @(2)]], x -> ICompose(@(1).val.var, 
                                                              @(1).val.domain, RC(@(2).val))),  

    ComposeStreamPermBitRule := ARule(Compose, [[@(1, RC), StreamPermBit], [@(2, RC), StreamPermBit]], 
        x->[RC(StreamPermBit(LinearBits(@(1).val.child(1).child(1).child(1) * @(2).val.child(1).child(1).child(1),
                                        @(1).val.child(1).child(1).child(2) * @(1).val.child(1).child(1).child(2)
                             ), 
                             @(1).val.child(1).streamSize
                            ))
           ]
    ),

    ComposeStreamPermRule := ARule(Compose, [[@(1, RC), StreamPerm], [@(2, RC), StreamPerm]], 
        x->[RC(StreamPermBit(LinearBits(@(1).val.child(1).child(1).child(1) * @(2).val.child(1).child(1).child(1),
                                        @(1).val.child(1).child(1).child(2) * @(1).val.child(1).child(1).child(2)
                             ), 
                             @(1).val.child(1).streamSize
                            ))
           ]
    ),

    # Broken!
    ComposeStreamPermRule2 := ARule(Compose, [[@(4, RC), @(1, StreamPermBit)], [@(5, RC), [@(2, STensor), @(3, StreamPermBit), @(6), @(7)]]], 
    x->[RC(StreamPermBit(LinearBits(@(1).val.child(1).child(1) * AddTopLeftOnes(@(3).val.child(1).child(1), Log2Int(@(2).val.p))), 
         @(1).val.streamSize))]),

    ComposeStreamPermRule2b := ARule(Compose, [[@(4, RC), @(1, StreamPermBit)], [@(2, STensor), [@(5, RC), @(3, StreamPermBit)], @(6), @(7)]], 
    x->[RC(StreamPermBit(LinearBits(@(1).val.child(1).child(1) * AddTopLeftOnes(@(3).val.child(1).child(1), Log2Int(@(2).val.p)),
                                    @(1).val.child(1).child(2) * TTensorI(@(3).val.child(1).child(2), @(2).val.p, APar, APar)
                        ), 
                        @(1).val.streamSize))]),

    ComposeStreamPermRule3 := ARule(Compose, [[@(2, STensor), [@(5, RC), @(3, StreamPermBit)], @(6), @(7)], [@(4, RC), @(1, StreamPermBit)]], 
    x->[RC(StreamPermBit(LinearBits(AddTopLeftOnes(@(3).val.child(1).child(1), Log2Int(@(2).val.p)) * @(1).val.child(1).child(1)), 
         @(1).val.streamSize))]),


    STensorComposePerm := Rule([@(1, STensor), @(2, Compose).cond(
            i->ObjId(i.child(Length(i.children()))) = RC and
               ObjId(i.child(Length(i.children())).child(1)) = StreamPermBit), @(3), @(4)],
    x->let(len := Length(@(2).val.children()),
           sp := @(2).val.child(len).child(1),

           Compose(STensor(Compose(Take(@(2).val.children(), len-1)), @(1).val.p, @(1).val.bs),
                   STensor(RC(sp), @(1).val.p, @(1).val.bs)))),

    RCTensorPermRule := Rule([RC, [@(1, Tensor), @(2, I), @(3).cond(x->ObjId(x) in [L, Prm, SIxLxI, StreamIxJ])]],
    x -> Tensor(Tensor(@(2).val, @(3).val), I(2))),

    RCSTensorRule := Rule([RC, @(1, STensor)], 
    x->STensor(RC(@(1).val.child(1)), @(1).val.p, 2*@(1).val.bs)),

    # I forget why I added this rule, but it is a problem.
    # RCTensorRule := Rule([RC, @(1, Tensor)],
    #     x->Tensor(@(1).val, I(2))), 

    RCSIterDirectSum := Rule([RC, [@(1, SIterDirectSum), @(2)]], 
    x->SIterDirectSum(@(1).val.var, @(1).val.domain, RC(@(2).val), 2*@(1).val.bs)),

    RCCodeBlock := Rule([RC, @(1, CodeBlock)], 
    x->CodeBlock(RC(@(1).val.child(1)))),

    SIterDirSumDiagRule := Rule([@(1, SIterDirectSum), RCDiag],
    x -> SIterDirectSum(@(1).val.var, @(1).val.domain, CodeBlock(@(1).val.child(1)), @(1).val.bs)),

    STensorDiagRule := Rule([@(1, STensor), RCDiag, @(2), @(3)],
    x -> STensor(CodeBlock(@(1).val.child(1)), @(1).val.p, @(1).val.bs)),

    STensorIComposeRule := Rule([@(1, STensor), @(2, ICompose), @(3), @(4)],
    x -> ICompose(@(2).val.var, @(2).val.domain, STensor(@(2).val.child(1), @(1).val.p, @(1).val.bs))),
        
));


Class(RulesDiagStream, RulesDiag);

RewriteRules(RulesDiagStream, rec(

      
#    STensorSTensorRule := ARule(Compose, [@(1, STensor), @(2, STensor).cond(x -> @(1).val.bs = x.bs and @(1).val.p = x.p)],
#   e->[ STensor(@(1).val.child(1) * @(2).val.child(1), @(1).val.p, @(1).val.bs) ]),

# Originally, the idea here was to pull the diagonal into the same IterDirectSum as the RC(DFT(2)) kernel.  I'm not sure
# if there's any reason to actually do this, anyway.

#     STensorSDirSumRule := ARule(Compose, [@(1, STensor), @(2, SIterDirectSum).cond(x -> 
#                        @(1).val.bs = x.bs and @(1).val.child(1).dims() = x.child(1).dims())],
#         e->[ SIterDirectSum(@(2).val.var, @(2).val.domain, @(1).val.child(1) * @(2).val.child(1), @(1).val.bs)]),

#     SDirSumSTensorRule := ARule(Compose, [@(1, SIterDirectSum), @(2, STensor).cond(x -> 
#                @(1).val.bs = x.bs and @(1).val.child(1).dims() = x.child(1).dims())], 
#         e->[ SIterDirectSum(@(1).val.var, @(1).val.domain, @(1).val.child(1) * @(2).val.child(1), @(1).val.bs)]),


    ## STensor(Diag(f)) -> SIterDirectSum(v, Diag(f o fTensor(fBase(v), fId)
    TensorDiagRule := ARule(Compose, [@(1, STensor), @(2, Diag), @(3), @(4)], e -> 
    let(
           a := @(1).val.child(1),
           f := @(2).val.element,
           p := @(1).val.p,
           v := Ind(p),
           m := Rows(a),
        
       [SIterDirectSum(v, p, a * Diag(
           fCompose(f, fTensor(fBase(p, v), fId(m)))
        ), @(1).val.bs)]
    )
    ),


    ## CodeBlock(c) * Diag(d) -> CodeBlock(c*Diag(d))
    CodeBlockDiagRule := ARule(Compose, [@(1, CodeBlock), @(2, Diag)], e -> let(
    a := @(1).val.child(1),
    f := @(2).val.element,
    [CodeBlock(a * Diag(f))])),


    ## Diag(d) * CodeBlock(c) -> CodeBlock(Diag(d)*c)
    CodeBlockDiagRule2 := ARule(Compose, [@(1, Diag), @(2, CodeBlock)], e -> let(
    a := @(2).val.child(1),
    f := @(1).val.element,
    [CodeBlock(Diag(f) * a)])),
    
));


RewriteRules(RulesStrengthReduce, rec(

    ## {a,b} & {c,d}, where width(b) = width(d) -> {a&c, b&d}
    AndConcatRule := Rule(
        [bin_and, 
           [concat, @(1), @(2), @(3,Value)], 
           [concat, @(4), @(5), @(6,Value, e->(@(3).val.v = e.v))]
        ], 

        e -> concat(bin_and(@(1).val, @(4).val), bin_and(@(2).val, @(5).val), @(3).val)
    ),
    

    ## {a,b} & {c,d} rule, width(b) > width(d)
    AndIncompatConcatRule := Rule(
        [bin_and, 
        [concat, @(1), @(2), @(3,Value)], 
        [concat, @(4), @(5), @(6,Value, e->(@(3).val.v > e.v))]
        ], 

        e -> bin_and(
            concat(
               concat(
                  @(1).val, 
              bin_and(@(2).val, 2^(@(6).val)-1), 
              @(3).val - @(6).val
               ), 
               idiv(@(2).val, 2^(@(6).val)), 
               @(6).val
            ), 
            
            concat(@(4).val, @(5).val, @(6).val)
         )
    ),


    ## rCylicShift(concat(a, b)) with compatible bit widths -> concat(b,a)
    CircShiftConcatRule := Rule(
        [rCyclicShift, 
        [concat, @(1), @(2), @(3, Value)],
        @(5, Value, e->(@(3).val >= e.v)), 
        @(6, Value)
        ],
        
        e -> concat(
            concat(
               bin_and(@(2).val, (2^@(5).val) - 1), 
               @(1).val, 
               @(6).val - @(3).val
            ),
            idiv(@(2).val, (2^@(5).val)), @(3).val - @(5).val
         )
     ),

    ## leq(0, x, y), where range(x) tells us x<=y -> V(1)
    LeqRangeTrueRule := Rule(
        [@(1, leq), 
            @(2, Value, x->(x.v = 0)), 
                @(3), 
        @(4, Value, x -> 
            IsBound(@(3).val.range) and 
            x.v+1 >= @(3).val.range
        )
        ], 
        
        y -> V(1)
    ),

    ## leq(a, concat(x,y,z), b), where concat(x,y,z) forces expression false -> V(0)
    LeqFalseConcatRule := Rule(
        [leq, @(1, Value), 
          [concat, @(2, Value), @(3), @(4)], 
          @(5, Value, e -> 
              (mul(@(2).val, pow(2,@(4).val).ev()).ev() > e.v)
          )
        ], 
        
        e -> V(0)
    ),

    ## leq(a, concat(x,y,z), b), where concat(x,y,z) forces expression true -> V(1)
    LeqTrueConcatRule := Rule(
        [leq, @(1, Value), 
          [concat, @(2, Value), @(3), @(4)], 
          @(5, Value, e -> 
              ((mul(@(2).val, pow(2,@(4).val)).ev() >= @(1).val) and 
               (sub(add(mul(@(2).val, 2^(@(4).val)), pow(2, @(4).val)), 1).ev() <= e.v)
              )
          )
        ],
        
        e -> V(1)
    ),

    ## leq(V(a), V(b)*x, V(c)) -> leq(floor(a/b), x, floor(c,b))
# NOTE: YSV -- this rule is invalid when the interval [@(1), @(4)] does not have any multiples of @(2)
#        for example 1 <= 4x <= 2 is not a satisfiable condition, but below rule will simplify it to
#        0 <= x <= 0 which is satisfiable
#    MulLeqRule := Rule([leq, @(1, Value), [mul, @(2, Value), @(3)], @(4, Value)], 
#        e -> leq(idiv(@(1).val,@(2).val), @(3).val, idiv(@(4).val, @(2).val))),


    ## leq(V(a), x+V(b), V(c)) -> leq(V(a-b), x, V(c-b))
    AddLeqRule := Rule([leq, @(1, Value), [add, @(2), @(3, Value)], @(4, Value)], 
        e -> leq(floor(@(1).val - @(3).val), @(2).val, floor(@(4).val - @(3).val))),


    ## leq(V(a), V(b)+x, V(c)) -> leq(V(a-b), x, V(c-b))
    AddLeqRule2 := Rule([leq, @(1, Value), [add, @(3, Value), @(2)], @(4, Value)], 
        e -> leq((@(1).val - @(3).val), @(2).val, (@(4).val - @(3).val))),


    ## (V(a)+x)-V(b) -> V(a-b)+x
    AddSubRule := Rule([sub, [add, @(1, Value), @(2)], @(3, Value)],
        e -> add(V(@(1).val.v - @(3).val.v), @(2).val)),

    ## assign(assign, val) -> assign(val)
    AssignAssignValue := Rule([assign, @(1), [@(2, assign), @(3, Value)]],
                    e->assign(@(1).val, @(3).val)),


    ## {A, B} (where B is 0 bits wide) -> A
    ConcatZeroRule := Rule([concat, @(1), @(2), @(3, Value, e -> e.v = 0)], e->@(1).val),

    ## I(1) x L x I(1) -> L
    TensorIdRule := Rule([Tensor, @(1, I, e->e.params[1]=1), @(2,L), @(3, I, e->e.params[1]=1)], 
        e -> @(2).val),

    ## (x+v)-v -> x
    SubAddRule := Rule([sub, [add, @(1), @(2, Value)], @(3, Value, e->e.v=@(2).val.v)],
        e -> @(1).val),

    ## (v*x)/v -> x
    DivMulRule := Rule([div, [mul, @(1, Value), @(2)], @(3, Value, e->e.v=@(1).val.v)],
        e -> @(2).val),
    
    ## (v*x)/v -> x
    iDivMulRule := Rule([idiv, [mul, @(1, Value), @(2)], @(3, Value, e->e.v=@(1).val.v)],
        e -> @(2).val),

    ## bin_and(x, 0) -> 0
    AndZeroRule := Rule([bin_and, @(1), @(2, Value, e->e.v=0)],
        e -> V(0)),

    ## bin_and(0, x) -> 0
    AndZeroRule2 := Rule([bin_and, @(1, Value, e->e.v=0), @(2)],
        e -> V(0)),

    ## add(a, stickyNeg(b)) -> sub(a,b)
    AddNegRule := Rule([add, @(1), [stickyNeg, @(2)]],
        e -> sub(@(1).val, @(2).val)),

    ## bin_shr(x, y), where 2^y > x -> 0
    RShiftZeroRule := Rule([bin_shr, @(1, Value), @(2, Value, e->2^(e.v) > @(1).val.v)],
        e -> V(0)),

));


#   Old, used in original "Id" based RDFT implementation

# Class(RulesIdRDFT, RulesId);

# RewriteRules(RulesIdRDFT, rec(

#    realDFT4UnkId := ARule(Compose, 
#       [@(1, realDFT4Unk), 
#        @(2, Id).cond(e -> ObjId(e.element) = idConst and
#                           e.element.params[1] = 4)
#       ], x ->
#       let(r := @(1).val,
#           m := r.params[1],
#           k := r.params[2],
#           l := @(2).val.element.params[2],
          
#           [realDFT4(m, k, l)]
#       )
#    ),

#    realDFT4UnkId2 := ARule(Compose, 
#       [@(1, realDFT4Unk), 
#        @(2, Id).cond(e -> ObjId(e.element) = fTensor)
#       ], x ->
#       let(r := @(1).val,
#           m := r.params[1],
#           k := r.params[2],
#           l := @(2).val.element.at(0),
          
#           [realDFT4(m, k, l)]
#       )
#    ),


#    SDirSumrDFTLIdRule := Rule(
#       [ @(1, SIterDirectSum), 
#         @(2, Compose).cond(x -> 
#          Length(x.children()) = 3      and
#              ObjId(x.child(1))=realDFT4Unk and 
#              x.child(2)=TL(4,2)            and 
#              x.child(3)=Id(idConst(4, @(1).val.var))) ],
#       x -> let( var  := @(1).val.var,
#        dom  := @(1).val.domain,
#        bs   := @(1).val.bs,
#        rd   := @(2).val.child(1),
#            str  := @(2).val.child(2),
#            idnt := @(2).val.child(3),
#        chld := Compose( [rd, idnt, str] ),

#        SIterDirectSum(var, dom, chld, bs)
#       )
#    ),

#    SDirSumIdRule := ARule(Compose,
#       [ @(1, SIterDirectSum),
#     @(2, Id) ],

#       x -> let(
#       var := @(1).val.var,
#       dom := @(1).val.domain,
#       blk := @(1).val.child(1),
#       n   := @(1).val.bs,
#       fna := @(2).val.element,
#       fnb := fTensor(fBase(dom, var), fId(n)),
#       idn := Id(fCompose(fna, fnb)),

#       [SIterDirectSum(var, dom, Compose(blk, idn), n)]
#       )),

#    DirSumIdRule := ARule(Compose,
#       [ @(1, IterDirectSum),
#     @(2, Id) ],

#       x -> let(
#       var := @(1).val.var,
#       dom := @(1).val.domain,
#       blk := @(1).val.child(1),
#       n   := @(1).val.child(1).dimensions[1],
#       fna := @(2).val.element,
#       fnb := fTensor(fBase(dom, var), fId(n)),
#       idn := Id(fCompose(fna, fnb)),

#       [IterDirectSum(var, dom, Compose(blk, idn))]
#       )),

#    TLIRule := ARule(Compose,
#      [ @(1, TL).cond(e->e=TL(4,2)),
#        @(2, Id).cond(e->e.domain()=4) ],
#      x -> [Compose(@(2).val, @(1).val)]),

#    PrmRule := ARule(Compose,
#      [ @(1, Prm).cond(e->e.size=4),
#        @(2, Id).cond(e->e.domain()=4) ],
#      x -> [Compose(@(2).val, @(1).val)]),


#    CodeBlockIDRule := ARule(Compose,
#      [ @(1, CodeBlock).cond(e->Length(e.children())=1),
#        @(2, Id)],
#      x -> [CodeBlock(Compose(@(1).val.child(1), @(2).val))]),
   
#    STensorIdRule := ARule(Compose,
#      [ @(1, STensor), @(2, Id) ],
#      x -> [ let(its := @(1).val.p,
#             blk := @(1).val.child(1),
#         bs  := @(1).val.bs,
#         j   := Ind(its),
        
#         Compose(SIterDirectSum(j, its, blk, bs), 
#             @(2).val)
#         )]),

#    TensorIdBlkRule := ARule(Compose,
#      [ @(1, Tensor).cond(e->ObjId(e.child(1)) = I), @(2, Id) ],
#      x -> [ let(its := @(1).val.child(1).params[1],
#             blk := @(1).val.child(2),
#         j   := Ind(its),
        
#         Compose(IterDirectSum(j, its, blk), 
#             @(2).val)
#         )]),

#     fTensorFBaseFBase := Rule(
#       [ @(1, fTensor), @(2, fBase), @(3, fBase) ],
#       x -> fBase(@(2).val.params[1]*@(3).val.params[1], @(2).val.params[2] * @(3).val.params[1] + @(3).val.params[2])),

#     # idConst(n, x) o L(n,y) -> idConst(n,x)
#     idConstPerm := Rule(
#        [ @(1, fCompose), @(2, idConst), @(3, L).cond(e -> e.size = @(2).val.size) ],
#        x -> @(2).val),


# ));

RuleRDFTStrategy := [MergedRuleSet(RulesSums, RulesFuncSimp, RulesStrengthReduce, StreamRulesRC, RulesDiagStream, SumRulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStream, RulesStreamCleanup)];




# include ESReduce?

SumStreamStrategy := [

    MergedRuleSet(RulesDiagStream, RulesSums, SumRulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, StreamRulesRC, RulesStream),

];




StreamStrategy := [

    
    MergedRuleSet(RulesDiagStream, RulesSums, SumRulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, StreamRulesRC, RulesStream, RulesStreamCleanup),
    RulesStream,
    MergedRuleSet(StreamRulesRC, RulesStream),
    MergedRuleSet(RulesDiagStream, RulesSums, SumRulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, StreamRulesRC, RulesStream),

    s -> SubstTopDown(s, CodeBlock, e -> e.sums()), 
#    FixLocalize,
    MergedRuleSet(RulesDiagStream, RulesSums, SumRulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce),
    MergedRuleSet(StreamRulesRC, RulesStream),
    MergedRuleSet(RulesStream, RulesDiagStream, RulesSums, SumRulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce),
    RulesStream,
    MergedRuleSet(RulesStream, RulesDiagStream, RulesSums, SumRulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce)



];
