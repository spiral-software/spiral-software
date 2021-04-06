
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(paradigms.common, paradigms.vector, paradigms.vector.sigmaspl);

_bDivisorPairs := N -> Concatenation(DivisorPairs(N), [[N, 1]]);

VBfly1 := VBase(F(2), 2);
VBfly3 := VBase(Mat([[1,E(4)],[1,-E(4)]]), 2);

Class(v_lo2, Exp);
Class(v_hi2, Exp);
Class(v_rev2, Exp);
Class(v_revhi2, Exp);
Class(v_neg01, Exp);
Class(v_neg03, Exp);
Class(v_neg23, Exp);
Class(v_neg2, Exp);
Class(v_mul1j, Exp);

VBaseHash := HashTableSPL();

VectorCodegen.VBase := (self, o, y, x, opts) >> let(
    lkup := HashLookup(VBaseHash, o),
    Cond(lkup <> false,
         lkup.code(o, y, x, opts),
         Error("VBase implementation of <o> not found in VBaseHash")));

# NOTE: increase vlen to 4 and RC the matrices
HashAdd(VBaseHash, VBase(L(4,2), 2),
    rec(
        spl := VBase(L(4,2), 2),
        code := (o, y, x, opts) -> chain(
            assign(vvref(y,0,2), v_lo2(vvref(x,0,2), vvref(x,1,2))),
            assign(vvref(y,1,2), v_hi2(vvref(x,0,2), vvref(x,1,2))))
    ));

HashAdd(VBaseHash, VBfly1,
    rec(
        spl := VBfly1,
        code := (o, y, x, opts) -> let(
            hi := TempVar(TVect(x.t.t, 2)),
            lo := TempVar(TVect(x.t.t, 2)),
            chain(
                assign(lo, v_lo2(vvref(x,0,2), vvref(x,0,2))),
                assign(hi, v_hi2(vvref(x,0,2), vvref(x,0,2))),
                assign(hi, v_neg03(hi)),
                assign(vvref(y, 0, 2), hi+lo)))
    ));

HashAdd(VBaseHash, VBfly3,
    rec(
        spl := VBfly3,
        code := (o, y, x, opts) -> let(
            hi := TempVar(TVect(x.t.t, 2)),
            lo := TempVar(TVect(x.t.t, 2)),
            chain(
                assign(lo, v_lo2(vvref(x,0,2), vvref(x,0,2))),
                assign(hi, v_revhi2(vvref(x,0,2), vvref(x,0,2))),
                assign(hi, v_neg03(hi)),
                assign(vvref(y, 0, 2), hi+lo)))
    ));

HashAdd(VBaseHash, VBase(J(2), 2),
    rec(
        spl := VBase(J(2), 2),
        code := (o, y, x, opts) ->
            assign(vvref(y, 0, 2), v_rev2(vvref(x,0,2)))
    ));

HashAdd(VBaseHash, VBase(I(2), 2),
    rec(
        spl := VBase(J(2), 2),
        code := (o, y, x, opts) ->
            assign(vvref(y, 0, 2), vvref(x,0,2))
    ));


# NOTE: Pull out twiddles, for 2-way nothing else needs to be done! (1-way cplx)
RulesFor(PSkewDFT3, rec(
   PSkewDFT3_toSkewDFT := rec(
       isApplicable := P -> IsEvenInt(P[1]),
       allChildren := P -> [[ SkewDFT(P[1]/2, P[2]) ]],
       rule := (P, C) -> let(n := P[1],
           M(n, n/2) *
           L(n,2) *
           Tensor(I(n/2), Mat([[1, E(4)], [1, -E(4)]])) *
           RC(C[1]) *
           L(n,n/2))
   )
));

RulesFor(BSkewDFT3, rec(
   BSkewDFT3_Base2 := rec(
       isApplicable := P -> P[1]=2,
       rule := (P, C) -> let(
       D := Dat1d(TReal, 2),
       re := nth(D, 0), im := nth(D, 1),
       Data(D, fPrecompute(FList(TReal, [cospi(2*P[2]), sinpi(2*P[2])])),
       F(2)*Mat([[1,re],
             [0,E(4)*im]])))
   ),

   BSkewDFT3_Base4 := rec(
       isApplicable := P -> P[1]=4,
       allChildren := P -> let(n:=P[1], a:=P[2], aa:=[a/2, (1-a)/2],
       [[ BSkewDFT3(n/2, aa[1]), BSkewDFT3(n/2, aa[2]), BRDFT3(4,a) ]]),
       rule := (P, C) ->
           K(P[1],P[1]/2) * DirectSum(C[1], C[2]) * C[3]
   ),

   BSkewDFT3_Fact := rec(
       switch := false,
       isApplicable := P -> P[1] = 4,
       allChildren := P -> let(n:=P[1], a:=P[2], aa:=[a/2, (1-a)/2],
       [[ BSkewDFT3(n/2, aa[1]), BSkewDFT3(n/2, aa[2]) ]]),
       rule := (P, C) -> K(P[1],P[1]/2) * DirectSum(C) * bruun2(P[1],P[2])
   ),

   BSkewDFT3_Decomp := rec(
       isApplicable := P -> P[1] > 4,
       allChildren := P -> let(N:=P[1], a:=P[2],
       Map2(DivisorPairs(N/2), (m,n) -> let(i := Ind(n),
           [BSkewDFT3(2*m, fr(2*n,i,2*a)), BRDFT3(2*n,a), InfoNt(i)]))),

       rule := (P, C, Nonterms) -> let(N := P[1], m := Cols(C[1])/2,
       K(N,2*m) * IterDirectSum(Nonterms[3].params[1], C[1]) * Tensor(C[2], I(m)))
   )
));

RulesFor(BSkewDFT4, rec(
   BSkewDFT4_Base2 := rec(
       isApplicable := P -> P[1]=2,
       rule := (P, C) -> let(w:=E(2*Denominator(P[2]))^Numerator(P[2]), cc:=Global.Conjugate,
       Mat([[w, w^3],
        [-cc(w), -cc(w)^3]]))
   ),

   BSkewDFT4_Base4 := rec(
       isApplicable := P -> P[1]=4,
       allChildren := P -> let(n:=P[1], a:=P[2], aa:=[a/2, (1-a)/2],
       [[ BSkewDFT4(n/2, aa[1]), BSkewDFT4(n/2, aa[2]), BRDFT3(4,a) ]]),
       rule := (P, C) ->
           K(P[1],P[1]/2) * DirectSum(C[1], C[2]) * C[3]
   ),

   BSkewDFT4_Decomp := rec(
       isApplicable := P -> P[1] > 4,
       allChildren := P -> let(N:=P[1], a:=P[2],
       Map2(DivisorPairs(N/2), (m,n) -> Concatenation(
           List([0..n-1], i->BSkewDFT4(2*m, fr(2*n,i,2*a).ev())),
           [BRDFT3(2*n,a)]))),
       rule := (P, C) -> let(N := P[1], m := Cols(C[1])/2,
       K(N,2*m) * DirectSum(DropLast(C,1)) * Tensor(Last(C), I(m)))
   )
));

RulesFor(BRDFT1, rec(
   BRDFT1_Base2 := rec(
       isApplicable := P -> P[1]=2,
       rule := (P, C) -> F(2)),

   BRDFT1_Base3 := rec(
       isApplicable := P -> P[1]=3,
       rule := (P, C) -> BRDFT1(P[1], P[2]).terminate()),

   BRDFT1_Decomp := rec(
       # NOTE: add support for odd sizes
       isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]),
       allChildren := P -> let(N:=P[1],
       Map2(_bDivisorPairs(N/2), (m,n) -> let(i := Ind(n-1),
           When(n=1,
           [BRDFT1(m), BRDFT3(m), BRDFT1(2*n)],
           [BRDFT1(m), BRDFT3(m), BRDFT1(2*n), InfoNt(i), BRDFT3(2*m, (1+i)/(2*n))])))),

       rule := (P, C, Nonterms) -> let(N := P[1], m := Cols(C[1]), n := N/m/2,
       RC(Scat(IP(N/2, OS(n, -1)))) * RC(L(N/2, m)) * # RC(K'(N/2,m)) *
       DirectSum(RC(L(m, m/2)) * DirectSum(C[1], C[2]),
                 When(n=1, [], IterDirectSum(Nonterms[4].params[1], C[5]))) *
       Tensor(C[3], I(m)))
   )
));

# URDFT_Decomp:
# MatSPL(Kp(16,4))^-1 * MatSPL(DFT(16))*MatSPL(Tensor(URDFT(8), I(2)))^-
#
NewRulesFor(DFT, rec(
    DFT_Bruun_Decomp := rec(
        switch := false,
        # NOTE: add support for odd sizes
        # only apply if there are no tSPL tags.
        applicable := nt -> nt.params[1] > 2 and IsEvenInt(nt.params[1]) and not nt.hasTags(),

        children := nt -> let(
            N := nt.params[1],
            Map2(_bDivisorPairs(N/2), (m,n) -> let(
                i := Ind(n-1),
                When(n=1,
                    [DFT(m), BSkewDFT3(m), BRDFT1(2*n)],
                    [DFT(m), BSkewDFT3(m), BRDFT1(2*n), InfoNt(i), BSkewDFT3(2*m, (1+i)/(2*n))]
                )
            ))
        ),

        apply := (nt, C, nonterms) -> let(
            N := nt.params[1], 
            m := Cols(C[1]), 
            n := N/m/2,

            Scat(IP(N, OS(n, -1))) 
            * L(N, 2*m) 
#            * RC(K'(N/2,m)) 
            * DirectSum(
                L(2*m, m) 
                * DirectSum(C[1], C[2]),
                When(n=1, 
                    [], 
                    IterDirectSum(Nonterms[4].params[1], C[5])
                )
            ) 
            * Tensor(C[3], I(m))
        )

#D       isApplicable := (self, P) >> P[1] > 2 and IsEvenInt(P[1]) and PHasNoTags(self.nonTerminal, P),
#D       allChildren := P -> let(N:=P[1],
#D       Map2(_bDivisorPairs(N/2), (m,n) -> let(i := Ind(n-1),
#D           When(n=1,
#D           [DFT(m), BSkewDFT3(m), BRDFT1(2*n)],
#D           [DFT(m), BSkewDFT3(m), BRDFT1(2*n), InfoNt(i), BSkewDFT3(2*m, (1+i)/(2*n))])))),
#D
#D       rule := (P, C, Nonterms) -> let(N := P[1], m := Cols(C[1]), n := N/m/2,
#D       Scat(IP(N, OS(n, -1))) * L(N, 2*m) * # RC(K'(N/2,m)) *
#D       DirectSum(L(2*m, m) * DirectSum(C[1], C[2]),
#D                 When(n=1, [], IterDirectSum(Nonterms[4].params[1], C[5]))) *
#D       Tensor(C[3], I(m)))
    )
));

Class(VWrapRC, VWrapBase, rec(
    __call__ := self >> self,
    wrap := (self, r, t, opts) >> @_Base(RC(t), r),
    twrap := (self, t, opts) >> RC(t)
));

NewRulesFor(DFT, rec(
   DFT_URDFT_Decomp := rec(
       switch := false,

       # NOTE: add support for odd sizes
       applicable := (self, t) >> let(n:=t.params[1],
          n > 4 and IsEvenInt(n) and not IsPrime(n/2)) and t.getTags() = [],

       maxRadix := 128,

       children := (self, t) >> let(N:=t.params[1], maxrad := self.maxRadix, rot := t.params[2],
               Map2(Filtered(DivisorPairs(N/2), x -> x[1] <= maxrad/2),
                   (m,n) -> [ DFT(2*m, rot), DFT(m, rot).addWrap(VWrapRC()),
                              GT(URDFT1(2*n, rot), GTVec, GTVec, [m]).withTags(t.getTags()) ])),

       apply := (t, C, Nonterms) -> let(
           N := t.params[1], m := Cols(C[2]), n := N/m/2, rot := t.params[2],
           j := Ind(n-1),

           Scat(Refl0_u(n, 2*m)) *
       DirectSum(
               BB(C[1]),
               IterDirectSum(j, BB(
                   #condM(2*m, m) * 
                   L(2*m, 2) *
                   Tensor(I(m), Mat([[1, E(4)], [1, -E(4)]])) *
                   RC(C[2]) *
                   RC(DirectSum(I(1), Diag(fPrecompute(
                                   fCompose(Twid(N, m, rot, 0, 0, j+1), fAdd(m, m-1, 1)))))) *
                   L(2*m, m))
                   # can replace by jj * b * (rdft(a) x I2)
                   # b = b' * (I x L^4_2)
               )
           ) *
       C[3] #Tensor(C[3], I(m))
       )
   ),

   DFT_Base4_VecCx := rec(
       switch := false,
       requiredFirstTag := AVecRegCx,
       applicable := (self, t) >> t.params[1]=4 and t.firstTag().v = 2,
       apply := (t, C, Nonterms) ->
           VTensor(F(2), 2) *
           VDiag(Tw1(4,2,t.params[2]), 2) *
           VBase(L(4,2), 2) *
           VTensor(F(2), 2)
   ),

   DFT_URDFT_Decomp_VecCx := rec(
       switch := false,
       requiredFirstTag := AVecRegCx,

       # NOTE: add support for odd sizes
       applicable := (self, t) >> let(n:=t.params[1], k:=t.params[2],
           k = 1 and n > 4 and IsEvenInt(n)), # and t.getTags() = []),

       maxRadix := 32,

       children := (self, t) >> let(N:=t.params[1], maxrad := self.maxRadix, tags := t.getTags(),
               Map2(Filtered(DivisorPairs(N/2), x -> x[1] <= maxrad/2),
                   (m,n) -> let(j := Ind(n-1),
                       [ DFT(2*m).withTags(tags),
                         rDFT(2*m, (j+1)/(2*n), 1).withTags(tags),
                         GT(URDFT1(2*n), GTVec, GTVec, [m]).withTags(tags),
                         InfoNt(j) ]))),

       apply := (t, C, Nonterms) -> let(
           N := t.params[1], m := Cols(C[2])/2,
           j := Nonterms[4].params[1],
           v := t.getTags()[1].v,

           condKp(N, 2*m) *
       DirectSum(C[1],
               IDirSum(j, K(2*m, 2) * Tensor(I(m), VBfly3) * C[2])) *
       C[3]
       )
   )
));

RulesFor(PkRDFT1, rec(
    PkRDFT1_Base2 := rec(
    isApplicable := P -> P[1]=2,
    rule := (P, C) -> F(2)
    ),

    PkRDFT1_Bruun_Decomp := CopyFields(BRDFT1_Decomp, rec(
       allChildren := P -> let(N:=P[1],
       Map2(_bDivisorPairs(N/2), (m,n) -> let(i := Ind(n-1),
           When(n=1,
           [PkRDFT1(m), BSkewPRDFT(m), BRDFT1(2*n), InfoNt(i)],
           [PkRDFT1(m), BSkewPRDFT(m), BRDFT1(2*n), InfoNt(i), BSkewPRDFT(2*m, (1+i)/(2*n))])))),
       )),
));


RulesFor(BRDFT3, rec(
   BRDFT3_Base2 := rec(
       isApplicable := P -> P[1]=2,
       rule := (P, C) -> I(2)),

   BRDFT3_Base3 := rec(
       isApplicable := P -> P[1]=3,
       rule := (P, C) -> ApplyFunc(BRDFT3, P).terminate()),

   BRDFT3_Base4 := rec(
       isApplicable := P -> P[1]=4,
       rule := (P, C) -> let(
       a := P[2], D := Dat1d(TReal, 2), q := nth(D, 0), q2 := nth(D, 1),
       Data(D, fPrecompute(FList(TReal, [2*cospi(a), 4*cospi(a)^2 - 1])),
           L(4,2) * VStack(
           F(2)*Mat([[1, 0,-1, 0],
                 [0, 0, 0,-q]]),
           F(2)*Mat([[0, 1, 0,q2],
                 [0, 0, q, 0]]))))
   ),

   BRDFT3_Fact := rec(
       switch := false,
       isApplicable := P -> P[1] > 2,
       allChildren := P -> let(n:=P[1], a:=P[2], [[ BRDFT3(n/2, a/2), BRDFT3(n/2, (1-a)/2) ]]),
       rule := (P, C) -> K(P[1],P[1]/2) * DirectSum(C) * bruun2(P[1],P[2])
   ),

   BRDFT3_Trig := rec(
       switch := false,
       isApplicable := P -> P[1] > 2 and (P[1] mod 2) = 0,
       allChildren := P -> let(n:=P[1], a:=P[2], [[ PolyDTT(SkewDTT(DST3(n/2), 2*a)) ]]),
       rule := (P, C) -> let(n := P[1], nn := n/2-1,
       b := Diag(BHD(n/2, 1, -1)) *
            VStack(
           HStack(DirectSum(I(1), -I(nn-1)),  O(nn,1), -J(nn),   O(nn,1)),
           RowVec(Replicate(nn,0), -1, Replicate(nn+1, 0)),
           HStack(O(nn,1), I(nn),  O(nn,1), J(nn)),
           RowVec(Replicate(nn+1,0), 1, Replicate(nn, 0))),
       M(P[1],P[1]/2) * Tensor(C[1], I(2)) * b)
   ),

   BRDFT3_Decomp := rec(
       isApplicable := P -> P[1] > 4,
       allChildren := P -> let(N:=P[1], a:=P[2], rot := P[3],
       Map2(DivisorPairs(N/2), (m,n) -> let(i := Ind(n),
           [BRDFT3(2*m, fr(2*n,i,2*a), rot), BRDFT3(2*n,a,rot), InfoNt(i)]))),

       rule := (P, C, Nonterms) -> let(N := P[1], m := Cols(C[1])/2,
       RC(K(N/2,m)) * IterDirectSum(Nonterms[3].params[1], C[1]) * Tensor(C[2], I(m)))
   )
));

NewRulesFor(rDFT, rec(
    rDFT_Base4 := rec(
       requiredFirstTag := ANoTag,
       applicable := t -> t.params[1]=4,
       apply := (t, C, Nonterms) -> let(
       a := t.params[2],
           Diag(1,1,1,-1) *
           Tensor(F(2), I(2)) *
           DirectSum(I(2), RCDiag(fPrecompute(FList(TReal, [cospi(a), sinpi(a)])))) *
           L(4,2))
   ),

   rDFT_Base4_Vec := rec(
       requiredFirstTag := AVecRegCx,
       applicable := t -> t.params[1]=4 and t.firstTag().v = 2,
       apply := (t, C, Nonterms) -> let(
       a :=t.params[2], D := Dat1d(TReal, 2),
           VTensor(F(2), 2) *
           DirectSum(VBase(I(2), 2), VBase(J(2), 2)*VDiag(FList(TReal, [1,-1]), 2)) *
           VRCDiag(fPrecompute(VData(FList(TReal, [1, cospi(a), 0, -sinpi(a)]), 2)), 2)
       )  # for some reason its not 1 c 0 s ...
   ),

   rDFT_Decomp := rec(
       applicable := t -> t.params[1] > 4,
       children := t -> let(N := t.params[1], a := t.params[2], tags := t.getTags(),
       Map2(DivisorPairs(N/2), (m,n) -> let(i := Ind(n),
           [ rDFT(2*m, fr(2*n,i,2*a)).withTags(tags),
                 GT(rDFT(2*n, a), GTVec, GTVec, [m]).withTags(tags),
                 InfoNt(i)]))),

       apply := (t, C, Nonterms) -> let(N := t.params[1], m := Cols(C[1])/2,
       RC(K(N/2,m)) * IterDirectSum(Nonterms[3].params[1], C[1]) * C[2]),
   )
));
#s:=SPLRuleTree(ExpandSPL(rDFT(4).withTags([AVecRegCx(SSE_4x32f)]), SpiralDefaults)[2]);


# packed fmt
RulesFor(BSkewPRDFT, rec(
   BSkewPRDFT_Decomp := CopyFields(BRDFT3_Decomp, rec(
       allChildren := P -> let(N:=P[1], a:=P[2],
       Map2(DivisorPairs(N/2), (m,n) -> let(i := Ind(n),
           [BSkewPRDFT(2*m, fr(2*n,i,2*a)), BRDFT3(2*n,a), InfoNt(i) ]))),
   )),

   BSkewPRDFT_Base2 := rec(
       isApplicable := P -> P[1]=2,
       rule := (P, C) -> I(2)
   ),

   BSkewPRDFT_Base4 := rec(
       isApplicable := P -> P[1]=4,
       rule := (P, C) -> let(
       a := P[2], D := Dat1d(TReal, 6), d := Diag(1,-1),
       Data(D, fPrecompute(FList(TReal, [cospi(a), cospi(2*a), cospi(3*a),
                                             sinpi(a), sinpi(2*a), sinpi(3*a) ])),
           L(4,2) * VStack(
           F(2)*Mat([[1,        0,   nth(D,1),          0 ],
                 [0,  nth(D,0),         0,  nth(D, 2) ]]),
         d*F(2)*Mat([[0,        0,   nth(D,4),          0 ],
                 [0,  nth(D,3),         0,  nth(D, 5) ]]))))
   )
));

# algebraic fmt
RulesFor(BSkewRDFT, rec(
   BSkewRDFT_Decomp := rec(
       isApplicable := P -> P[1] > 4,
       allChildren := P -> let(N:=P[1], a:=P[2],
       Map2(DivisorPairs(N/2), (m,n) -> Concatenation(
           List([0..n-1], i->BSkewRDFT(2*m, fr(2*n,i,2*a).ev())),
           [BRDFT3(2*n,a)]))),
       rule := (P, C) -> let(N := P[1], m := Cols(C[1])/2,
       K(N,2*m) * DirectSum(DropLast(C,1)) * Tensor(Last(C), I(m)))
   )
));

RulesFor(DCT4, rec(
   DCT4_BSkew_Decomp := rec(
       switch := false,
       isApplicable := P -> P > 2,
       allChildren := P -> let(N:=P, a:=1/2,
       Map2(DivisorPairs(N), (m,n) -> Concatenation(
           List([0..n-1], i->SkewDTT(DCT4(m), fr(n,i,a).ev())),
           [BRDFT3(2*n,a/2)]))),

       rule := (P, C) -> let(N := P, m := Cols(C[1])/2, n:=Cols(Last(C))/2,
       i := Ind(m),
       K(N,2*m) *
       DirectSum(DropLast(C,1)) *
       K(N, 2*n) *
       ISum(i,
           Scat(H(N,2*n,(2*n)*i,1)) * Last(C) *
           DirectSum(I(n+1), -I(n-1)) * OS(2*n,-1) * Gath(BH(N,2*N-1,2*n,i,2*m))))
   )
));

# right := (m,n) -> let(N:=2*m*n, i:=Ind(m),
#     ISum(i,
#   Scat(H(N,2*n,(2*n)*i,1)) * BRDFT3(2*n,1/4) *
#   DirectSum(I(n), -I(n)) * Gath(BH1(2*N,2*n,i,2*m))));

# RC.toAMat := self >> AMatMat(RCMatCyc(MatSPL(self.child(1))));

# bb:=(n,a)->MatSPL(BRDFT3(n,a));;
# ts:=TensorProductMat;;
# ii:=IdentityMat;;
# ms:=MatSPL;
# rms := x->RCMatCyc(MatSPL(x));

# BRDFT3(4,a) -> 6a+3m
# BRDFT3(4,1/4) -> 6a+2m

# costs:
#         4, 6+2   = 8
# PRDFT3, 8, 22+10 = 32
#        16, 66+30 = 96
#
#           4, 8(1/4) or 9(other)
# BRDFT3, 8, 34
#           16, 104
