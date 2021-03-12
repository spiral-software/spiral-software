
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Cross-compatibility
#


#######################################################################################
#   tSPL DFT rules
NewRulesFor(DFT, rec(
    #F DFT_CT: 1965
    #F   General Cooley-Tukey Rule
    #F   DFT_n = (DFT_n/d tensor I_d) * diag * (I_n/d tensor F_d) * perm
    #F
    #F Cooley/Tukey:
    #F   An Algorithm for the Machine Calculation of Complex Fourier Series.
    #F   Mathematics of Computation, Vol. 19, 1965, pp. 297--301.
    #F
    DFT_tSPL_CT := rec(
    info          := "tSPL DFT(mn,k) -> DFT(m, k%m), DFT(n, k%n)",

    maxSize       := false,
    filter := e->true,

    applicable    := (self, nt) >> nt.params[1] > 2
        and (self.maxSize = false or nt.params[1] <= self.maxSize)
        and not IsPrime(nt.params[1])
        and nt.hasTags(),

    children      := (self, nt) >> Map2(Filtered(DivisorPairs(nt.params[1]), self.filter), (m,n) -> [
        TCompose([
            TGrp(TCompose([
                TTensorI(DFT(m, nt.params[2] mod m), n, AVec, AVec),
                TTwiddle(m*n, n, nt.params[2])
            ])),
            TGrp(TTensorI(DFT(n, nt.params[2] mod n), m, APar, AVec))
        ]).withTags(nt.getTags())
    ]),

    apply := (nt, c, cnt) -> c[1],

#D    isApplicable := (self,P) >> P[1] > 2 and
#D        (self.maxSize=false or P[1] <= self.maxSize) and not IsPrime(P[1]) and PHasTags(self.nonTerminal, P),
#D    allChildren  := P -> Map2(DivisorPairs(P[1]),
#D        (m,n) -> [ TCompose([TGrp(TCompose([TTensorI(DFT(m, P[2] mod m), n, AVec, AVec), TDiag(fPrecompute(Tw1(m*n, n, P[2])))])),
#D                   TGrp(TTensorI(DFT(n, P[2] mod n), m, APar, AVec))], P[3]) ]),
#D    rule := (P,C,nt) -> C[1],
    switch := false
    )
));


NewRulesFor(DFT, rec(
    DFT_TTensorI_CT := rec(
        switch := false,
        maxSize := false,
        minSize := false,

        applicable := (self, t) >> let(
            n := Rows(t),
            n > 2
            #and nt.hasTags() --> fixed: dpickem 08/03/09
            and t.hasTags()
            and (self.maxSize=false or n <= self.maxSize)
            and (self.minSize=false or n >= self.minSize)
            and not IsPrime(n)
        ),

        children := t -> let(
            tags := t.params[3],
            Map2(DivisorPairs(Rows(t)), (m,n) -> [
                TTensorI_GT(GT(DFT(m, t.params[2] mod m), XChain([0, 1]), XChain([0, 1]), [n]).withTags(tags)),
                TTensorI_GT(GT(DFT(n, t.params[2] mod n), XChain([0, 1]), XChain([1, 0]), [m]).withTags(tags))
            ])
        ),

        apply := (t, C, Nonterms) ->
            Grp(C[1] * Diag(fPrecompute(Tw1(Rows(t), Rows(Nonterms[2].params[1]), t.params[2])))) * C[2]
    ),

    #F DFT_Rader: Rader's Algorithm for Prime size DFT
    #F
    #F   DFT(p) -> P(g)' * (1 dirsum DFT(p-1)) * Tp * (1 dirsum DFT(p-1)) * P(g)",
    #F   P(g) = perm
    #F   Tp   = [[1,1],[1,-1/(p-1)]] dirsum diag
    #F
    DFT_tSPL_Rader := rec(
        minSize := 3,
        useSymmetricAlgorithm := true,
        avoidSizes := [],
        switch := false,

        applicable := (self, t) >> let(
            n := Rows(t),
            n >= self.minSize
            and IsPrime(n)
            and (not n in self.avoidSizes)
            and t.hasTags()
        ),

        children := (self, t) >> let(
            N := Rows(t),
            tags := t.getTags(),
            When(self.useSymmetricAlgorithm,
                [[
                    TDirectSum(I(1), DFT(N-1, -1)).withTags(tags),
                    TRaderMid(N, t.params[2], PrimitiveRootMod(N)).withTags(tags)
                ]],
                [[
                    TCompose([
                        TDirectSum(I(1), DFT(N-1, -1)),
                        TRaderMid(N, t.params[2], PrimitiveRootMod(N)),
                        TDirectSum(I(1), DFT(N-1, -1))
                    ]).withTags(tags)
                ]]
            )
        ),

    apply := (self, t, C, Nonterms) >> let(
        N := Rows(t), k := t.params[2], root := PrimitiveRootMod(N),
        When(self.useSymmetricAlgorithm,
            RR(N, 1, root).transpose()
            * C[1].transpose() * C[2] * C[1]
            * RR(N, 1, root),
            RR(N, 1, root).transpose()
            * C[1]
            * RR(N, 1, root)
        ))
    ),

    #F DFT_Bluestein : Convert FFT to Toeplitz matrix, then embed in a larger Circulant
    #F
    #F    DFT_n -> diag * Toeplitz * diag
    #F
    # DFT on inputs, some of which are 0:
    #
    DFT_tSPL_Bluestein := rec(
        minRoundup := 8,
        customFilter := True,
        goodFactors := [2,3,5],
        applicableSizes := IsPrime,
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

        diag := (N, k) ->
            List([0..N-1], x -> ComplexW(2*N, x^2 * k)),

        circulantToep := (toep,size) -> let(l:=Length(toep),
            Concatenation(
                List([1..(l+1)/2],   x->toep[(l+1)/2+x-1]),
                List([1..size-l],    x->0),
                List([1..(l-1)/2],   x->toep[x]))),

        switch := false,
        applicable := (self, t) >> self.applicableSizes(Rows(t)) and t.hasTags(),

        children := meth(self, t)
            local P, N, k, circsize, dvar, diag, toep, circ, diagonalized_circ, tags, kids;

            tags := t.getTags();
            P := t.params;
            N := P[1];
            k := P[2];
            diag := FData(self.diag(N,k));
            toep := self.toeplitz(N,k);
            kids := [];
            for circsize in self.circSizes(N) do
                dvar := Dat(TArray(TComplex, circsize));
                circ := self.circulantToep(toep, circsize);
                diagonalized_circ := FData(1/circsize * ComplexFFT(circ));
                Add(kids, [
                    TCompose([
                        TRDiag(circsize/2, N, diag).transpose(),
                        PrunedDFT(circsize, -1, circsize/2, [0]).transpose(),
                        TDiag(diagonalized_circ),
                        PrunedDFT(circsize, 1, circsize/2, [0]),
                        TRDiag(circsize/2, N, diag)
                    ]).withTags(tags)
                ]);
            od;

            return kids;
        end,

        apply := (t, C, Nonterms) -> C[1]
    ),


    #F DFT_GoodThomas : Prime Factor FFT
    #F
    #F     DFT_n*k -> perm * (DFT_n_a tensor DFT_k_b) * perm
    #F     when gcd(n,k) = 1
    #F
    DFT_tSPL_GoodThomas := rec(
        applicable     := (t) ->
            t.params[1] > 2
            and DivisorPairsRP(t.params[1]) <> []
            and t.hasTags(),

        children := (self, t) >> let(
            tags := t.getTags(),
            mn := t.params[1],
            k := t.params[2],
            Map2(DivisorPairsRP(mn), (m,n) -> [
                TTensor(DFT(m, k*n mod m), DFT(n, k*m mod n)).withTags(tags)
            ])
        ),

        apply := (t, C, Nonterms) -> let(
            P := t.params,
            r := Rows(Nonterms[1].params[1]),
            s := Rows(Nonterms[1].params[2]),
            alpha := 1 / s mod r,
            beta  := 1 / r mod s,
            # CRT should not be pulled into sums before DelayedDirectSums are terminated,
            # otherwise PFA inside Rader for n =vq breaks as VectRaderDiag rules can't match
            DelayedPrm(CRT(r,s,1,1)).transpose() * C[1] * DelayedPrm(CRT(r,s,1,1))
        )
    )
));

#--------------------------------------------------------------------------------------

NewRulesFor(TTensorI, rec(

    # Vector recursion for the DFT
    DFT_vecrec := rec(
        forTransposition := false,
        applicable := nt ->
            ObjId(nt.params[1]) = DFT
            and IsParVec(nt.params)
            and let(p1 := nt.params[1].params[1],
                p1 > 2 and not IsPrime(p1)
            )
            and nt.hasTags(),

        children := nt -> let(
            f := nt.params[1],
            k := nt.params[2],
            Map2(DivisorPairs(f.params[1]), (m,n) -> let(
                r := When(
                    nt.hasTags() and nt.firstTagIs(spiral.paradigms.vector.AVecReg),
                    VWrapTRC(nt.firstTag().v),
                    VWrapId
                ),
                [
                    TGrp(TCompose([
                        TTensorI(DFT(m, f.params[2] mod m), n, AVec, AVec),
                        TDiag(fPrecompute(Tw1(m*n, n, f.params[2])))
                    ])).withTags(nt.getTags()), <# strange #>
                    TTensorI(
                        TTensorI(DFT(n, f.params[2] mod n), m, APar, AVec),
                        k, APar, AVec
                    ).withTags(nt.getTags()) <# strange #>
                ]
            ))
        ),

        apply := (nt, c, cnt) -> Tensor(I(nt.params[2]), c[1]) * c[2],

        switch := false

#D    isApplicable := (self,P) >> ObjId(P[1])=DFT and P[3].isPar and P[4].isVec and
#D                                    let(P1:=P[1].params[1], P1 > 2 and not IsPrime(P1)) and PHasTags(self.nonTerminal, P),

#D        allChildren  := P -> let(k:=P[2],
#D        Map2(DivisorPairs(P[1].params[1]),
#D        (m,n) -> let( r:=When(Length(P[5])>0 and P[5][1].isReg, VWrapTRC(P[5][1].v), VWrapId),
#D            [ TGrp(TCompose([TTensorI(DFT(m, P[1].params[2] mod m), n, AVec, AVec),
#D                     TDiag(fPrecompute(Tw1(m*n, n, P[1].params[2])))]), P[5]).setWrap(r),

#D              TTensorI(TTensorI(DFT(n, P[1].params[2] mod n), m, APar, AVec),
#D                  k, APar, AVec, P[5]).setWrap(r) ]))),

#D    rule := (P,C,nt) -> Tensor(I(P[2]), C[1]) * C[2],
    ),

    # Transposed vector recursion for the DFT
    DFT_vecrec_T := rec(
        forTransposition := false,

        applicable := nt ->
            ObjId(nt.params[1]) = DFT
            and IsVecPar(nt.params)
            and let(
                p1 := nt.params[1].params[1],
                p1 > 2
                and IsPrime(p1)
            )
            and nt.hasTags(),

        children := nt -> let(
            k := nt.params[2],
            Map2(DivisorPairs(nt.params[1].params[1], (m,n) -> let(
                r := When(nt.hasTags() and nt.isTag(1, spiral.paradigms.vector.AVecReg),
                    VWrapTRC(nt.firstTag().v),
                    VWrapId
                ),
                [
                    TTensorI(
                        TTensorI(DFT(n, P[1].params[2] mod n), m, AVec, APar),
                        k, AVec, APar
                    ).withTags(nt.getTags()).setWrap(r), <# strange #>
                    TGrp(TCompose([
                        TDiag(fPrecompute(Tw1(m*n, n, P[1].params[2]))),
                        TTensorI(DFT(m, P[1].params[2] mod m), n, AVec, AVec)
                    ])).withTags(nt.getTags()).setWrap(r) <# strange #>
                ]
            )))
        ),

        apply := (nt, c, cnt) -> c[1] * Tensor(I(nt.params[2]), c[2]),

        switch := false

#D        isApplicable := (self,P) >> ObjId(P[1])=DFT and P[3].isVec and P[4].isPar and
#D                                    let(P1:=P[1].params[1], P1 > 2 and not IsPrime(P1)) and PHasTags(self.nonTerminal, P),
#D        allChildren  := P -> let(k:=P[2],
#D            Map2(DivisorPairs(P[1].params[1]),
#D        (m,n) -> let(r:=When(Length(P[5])>0 and P[5][1].isReg, VWrapTRC(P[5][1].v), VWrapId), [
#D            TTensorI(TTensorI(DFT(n, P[1].params[2] mod n), m, AVec, APar),
#D                                        k, AVec, APar, P[5]).setWrap(r),
#D
#D                    TGrp(TCompose([TDiag(fPrecompute(Tw1(m*n, n, P[1].params[2]))),
#D                                   TTensorI(DFT(m, P[1].params[2] mod m), n, AVec, AVec)
#D                  ]), P[5]).setWrap(r)
#D                            ]))),
#D
#D    rule := (P,C) -> C[1] * Tensor(I(P[2]), C[2]),

    )
));



#######################################################################################################
#   tSPL rule
NewRulesFor(MDDFT, rec(
    MDDFT_tSPL_RowCol := rec(
        info := "tSPL MDDFT_n -> MDDFT_n/d, MDDFT_d",

        applicable := (self, t) >> Length(t.params[1]) > 1,
        freedoms := t -> [ [1..Length(t.params[1])-1] ],

        child := (t, fr) -> let(
            newdims := SplitAt(t.params[1], fr[1]),
            rot := t.params[2],
            [ TTensor(
                MDDFT(newdims[1], rot),
                MDDFT(newdims[2], rot)
            ).withTags(t.getTags())]
        ),

        apply := (t, C, Nonterms) -> C[1],
        switch := false
    ),
));


AllRadices := function(n)
    local divisors, radices;
    divisors := DropLast(Drop(DivisorsInt(n), 1), 1);
    radices := Filtered(divisors, i -> n=(i^LogInt(n, i)));
    return radices;
end;



##########################################################################################
#   tSPL Pease DFT rule
NewRulesFor(DFT, rec(
    DFT_tSPL_Pease   := rec (
        info             := "Pease tSPL DFT_(k) -> (\Prod(L(I tensor F)Tc))*DR(k,r)",
#        forTransposition := false,
        forTransposition := true,
        maxRadix         := 32,
        minRadix         := 4,

        applicable := (self, nt) >>
            nt.hasTags()
            and nt.params[2] = 1
            and let(
                r := AllRadices(nt.params[1]),
                Length(r) > 0
                and ForAny(r, i ->
                    i >= self.minRadix
                    and i <= self.maxRadix
                    #and nt.firstTag().legal_kernel(i)
                )
            ),

        children := (self, nt) >> let(
            N := nt.params[1],
            m := nt.params[2],
            radices := Filtered(AllRadices(N), i ->
                i >= self.minRadix and i <= self.maxRadix
                #and nt.firstTag().legal_kernel(i)
            ),
            j := var("j"),

            List(radices, rdx -> [
                TCompose([
                    TICompose(j, LogInt(N, rdx),
                        TCompose([
                            TTensorI(DFT(rdx, m), N/rdx, AVec, APar),
                            TDiag(fPrecompute(TC(N, rdx, j, m)))
                        ])
                    ),
                    TDR(N, rdx)
                ]).withTags(nt.getTags()),
            ])
        ),

        apply := (nt, c, cnt) -> c[1],

        switch := false

#D    isApplicable     := (self, P) >> PHasTags(self.nonTerminal, P) and P[2] = 1 and
#D                                     let(r := AllRadices(P[1]),
#D                                        Length(r) > 0 and
#D                                        ForAny(r, i-> i >= self.minRadix and i <= self.maxRadix and PGetFirstTag(self.nonTerminal, P).legal_kernel(i))
#D                                     ),
#D
#D    allChildren      := (self, P) >> let(
#D                            N := P[1],
#D                            m := P[2],
#D                            radices := Filtered(AllRadices(N), i-> i >= self.minRadix and i <= self.maxRadix and PGetFirstTag(self.nonTerminal, P).legal_kernel(i)),
#D                            j := var("j"),
#D
#D                            List(radices, rdx ->
#D                                [
#D                                    AddTag(
#D                                        TCompose([
#D                                            TICompose(j, LogInt(N, rdx),
#D                                                TCompose([
#D                                                    TTensorI(DFT(rdx, m), N/rdx, AVec, APar),
#D                                                    TDiag(fPrecompute(TC(N, rdx, j, m)))
#D                                                ])),
#D                                            TDR(N, rdx)
#D                                        ]),
#D                                    PGetTags(self.nonTerminal, P))
#D                                ]
#D                            )),
#D
#D    rule := (P, C) -> C[1],
    )
));

##########################################################################################
#   tSPL Stockham DFT rule
NewRulesFor(DFT, rec(
    DFT_tSPL_Stockham   := rec (
        info             := "Stockham tSPL DFT_(k) -> (\Prod(DFT tensor I)*Diag*(L tensor I))",
        forTransposition := false,
        maxRadix         := 2,
        minRadix         := 2,

        #one dimensional transform --> nt.params[2] = 1
        applicable := (self, nt) >>
            nt.hasTags()
            and nt.params[2] = 1
            and let(
                r := AllRadices(nt.params[1]),
                Length(r) > 0
                and ForAny(r, i ->
                    i >= self.minRadix
                    and i <= self.maxRadix
                    #and nt.firstTag().legal_kernel(i)
                )
            ),

        children := (self, nt) >> let(
            N := nt.params[1],
            m := nt.params[2],
            radices := Filtered(AllRadices(N), i ->
                i >= self.minRadix and i <= self.maxRadix and (When(IsBound(nt.firstTag().legal_kernel), nt.firstTag().legal_kernel(i), true))
            ),
#need the tag for TICompose to be parallelized, but then the DFT(r) in TTensorI will not be broken down
#if i use no tag for TICompose, no parallelization will take place ...
#solution: drop tags in GT_Par's children function --> rule GT_Par_drop implemented in paradigms/gpu/breakdown.gi
            List(radices, rdx -> let(j := var.fresh("j", TInt, LogInt(N, rdx)), tags := Drop(t.getTags(), 1),
                [TICompose(j, LogInt(N, rdx),
                        TCompose([
                            TTensorI(DFT(rdx, m), N/rdx, AVec, AVec),
                            #Diag(fPrecompute(diagTensor(Stockham_radix.gen(rdx, LogInt(N, rdx)-j-1), fConst(TComplex, rdx^j, 1)))),
                            #Tensor(Diag(fPrecompute(Stockham_radix.gen(rdx, LogInt(N, rdx)-j-1))), I(rdx^j)),
                            #TDiag(fPrecompute(diagTensor(TC5(rdx, LogInt(N, rdx)-j-1), fConst(TComplex, rdx^j, 1)))),
                            TTwiddle_Stockham(N, N/rdx, rdx, j),
                            Tensor(L(N/rdx^j, rdx), I( rdx^j))
                        ])
                    ).withTags(nt.getTags()),
                ])
            )
        ),

        apply := (nt, c, cnt) -> c[1],

        switch := false
    )
));

NewRulesFor(DFT, rec(
    DFT_tSPL_Stockham_split   := rec (
        info             	:= "Stockham tSPL DFT_(k) -> (\Prod(DFT tensor I)*Diag*(L tensor I)), first n iterations split off",
        forTransposition 	:= false,
        minRadix         	:= 4,
        maxRadix         	:= 4,

        # one dimensional transform --> nt.params[2] = 1
        # first parameter of AGenericTag indicates how many iterations to split off
        applicable := (self, nt) >>
            nt.hasTags()
#            and nt.firstTag().kind() = AGenericTag
#            and IsPosInt(nt.firstTag().params[1])
            and nt.params[2] = 1
            and let(
                r := AllRadices(nt.params[1]),
                Length(r) > 0
                and ForAny(r, i ->
                    i >= self.minRadix
                    and i <= self.maxRadix
                    #and nt.firstTag() is GPU tag
                    #and nt.firstTag().legal_kernel(i)
                )
            ),

        children := (self, nt) >> let(
            N := nt.params[1],
            m := nt.params[2],
# number of iterations to peel off is n
#            n := nt.firstTag().params[1],
            n := 1,
            radices := Filtered(AllRadices(N), i ->
                i >= self.minRadix and i <= self.maxRadix and (When(IsBound(nt.firstTag().legal_kernel), nt.firstTag().legal_kernel(i), true))
            ),
            List(radices, rdx -> let(j := var.fresh("j", TInt, LogInt(N, rdx)-n), i := var.fresh("i", TInt, n),
            	# first TICompose deals with the iterations [n..LogInt(N,rdx)-1], these are the critical write-out stages

            	[TCompose([
                	TICompose(j, LogInt(N, rdx)-n,
                    	    TCompose([
                        	    TTensorI(DFT(rdx, m), N/rdx, AVec, AVec),
                            	#TDiag(fPrecompute(diagTensor(Stockham_radix.gen(rdx, LogInt(N, rdx)-j-1), fConst(TComplex, rdx^j, 1)))),
                            	#Diag(fPrecompute(diagTensor(TTwiddle_Stockham(rdx^(LogInt(N, rdx)-j), rdx^(LogInt(N, rdx)-j-1), 1).terminate(), fConst(TComplex, rdx^j, 1)))),
	                            TTwiddle_Stockham(N, N/rdx, rdx, j),
	                            TL(N/rdx^j, rdx, 1, rdx^j)
    	                    ])
        	            ).withTags(nt.getTags()),
            	     # second TICompose has to range from [0..n-1]
                	 TICompose(i, n,
	                        TCompose([
    	                        TTensorI(DFT(rdx, m), N/rdx, AVec, AVec),
        	                    #TDiag(fPrecompute(diagTensor(Stockham_radix.gen(rdx, n-i-1), fConst(TComplex, N/rdx^(n-i), 1)))),
                            	#Tensor(TTwiddle(rdx^(n-j), rdx^(n-j-1)), I(rdx^j)),
	                            #Diag(fPrecompute(diagTensor(TTwiddle_Stockham(rdx^(LogInt(N, rdx)-j), rdx^(LogInt(N, rdx)-j-1), 1).terminate(), fConst(TComplex, rdx^j, 1)))),
								TTwiddle_Stockham(N, N/rdx, rdx, j),
            	                TL(rdx^(n-i), rdx, 1, N/rdx^(n-i))
                	        ])
	                    ).withTags(nt.getTags()),
					])
                ])
            )
        ),

        apply := (nt, c, cnt) -> c[1],

        switch := false
    )
));
