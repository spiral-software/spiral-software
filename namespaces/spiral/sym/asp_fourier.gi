
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(paradigms.vector); 

Declare(ASPF);

Class(flipBot, PermClass, rec(
    def := k -> rec(size := 2*k),
    lambda := self >> let(k := self.params[1],
	i := Ind(2*k),
	Lambda(i, cond(leq(i, k-1), i, i + 1 - 2*imod(i,2))))
));

Class(ASPF, TaggedNonTerminal, rec(
   _short_print := true,
   abbrevs := [ (alg, tbasis, fbasis) -> Checked(
           IsASPAlgebra(alg), IsASPTimeBasis(tbasis), IsASPFreqBasis(fbasis),
           [alg, tbasis, fbasis]) ],

   isReal := self >> self.params[3].dim() > 1, 

   dims := self >> let(n:=self.params[1].n, [n, n]),

   print := (self, i, is) >> let(base_print := NonTerminal.print, 
       Print(base_print(self, i, is), 
           When(self.tags<>[], Print(".withTags(", self.tags, ")")))),

   hashAs := self >> let(p := self.params,
       t := ObjId(self)(p[1].hashAs(), p[2], p[3].hashAs()).withTags(self.getTags()),
       When(self.transposed, t.transpose(), t)),

   HashId := self >> [ # NOTE: why is this function necessary?
       [ObjId(self.params[1]), self.params[1].n], self.params[2], ObjId(self.params[3]), 
        self.getTags(), self.transposed ],


   norm := self >> CopyFields(self, rec(params := 
                       [self.params[1], self.params[2].norm(), self.params[3].norm()])),

   _scale12 := (self, s1, s2) >> let(n := self.params[1].n, Cond(
       n<=2,         s1,
       IsEvenInt(n), Diag(diagDirsum(fConst(TReal, 2, s1), fConst(TReal, n-2, s2))),
       <# else #>    Diag(diagDirsum(fConst(TReal, 1, s1), fConst(TReal, n-1, s2))))),

   _scale34 := (self, s1, s2) >> let(n := self.params[1].n, Cond(IsEvenInt(n),
       s2,
       Diag(diagDirsum(fConst(TReal, n-1, s2), fConst(TReal, 1, s1))))),

   terminate := self >> let(  
       A := self.params[1], T := self.params[2], F := self.params[3], s1 := F.scale1d, 
       n := EvalScalar(A.n),
       res := CondPat(self, 
           [ASPF, XN_min_1,  Time_TX, Freq_1,  ...], s1 * DFT (n, A.rot), 
           [ASPF, XN_min_1,  Time_TX, Freq_1H, ...], s1 * DFT2(n, A.rot), 
           [ASPF, XN_plus_1, Time_TX, Freq_1,  ...], s1 * DFT3(n, A.rot), 
           [ASPF, XN_plus_1, Time_TX, Freq_1H, ...], s1 * DFT4(n, A.rot), 

           [ASPF, XN_skew,  @,  @(1, [Freq_1,Freq_1H]), ...], let(
               aa := EvalScalar(A.a), 
               p := When(aa > 1/2, J(n), I(n)) * LIJ(n).transpose(),
               p * s1 * Cond(ObjId(F)=Freq_1, BSkewDFT3(n, aa, A.rot), BSkewDFT4(n, aa, A.rot)) * T.toX(A)),

           [ASPF, XN_min_1, ...],
               DirectSum(List(A.rspectrum(), a -> F.from1X(a))) * BRDFT1(n, A.rot) * T.toX(A), 
           [ASPF, XN_min_1U, ...],
               DirectSum(List(A.rspectrum(), a -> F.from1X(a))) * UBRDFT1(n, A.rot) * T.toX(A), 
           [ASPF, XN_plus_1, ...],
               DirectSum(List(A.rspectrum(), a -> F.from1X(a))) * BRDFT3(n, 1/4, A.rot) * T.toX(A), 
           [ASPF, XN_skew, ...], let(aa := EvalScalar(A.a), 
               p := When(aa > 1/2, RC(J(n/2)), RC(I(n/2))) * RC(LIJ(n/2).transpose()),
               DirectSum(List(A.rspectrum(), s -> F.from1X(s))) * p * BRDFT3(n, aa, A.rot) * T.toX(A)), 

           Error("not implemented")),
       Cond(self.transposed, TerminateSPL(res).transpose(), TerminateSPL(res))),

   transpose := self >> Cond(
       ObjId(self.params[1]) = XN_min_1 and self.params[2] = Time_TX and self.params[3]=Freq_1(1),
           self, # DFT
       CopyFields(self, rec(
               transposed := not self.transposed,
               dimensions := Reversed(self.dimensions) ))),

   conjTranspose := self >> Cond(
       self.isReal(),
           self.transpose(),
       CopyFields(self, rec(
	       params := [self.params[1].conj(), self.params[2], self.params[3]],
               transposed := not self.transposed,
               dimensions := Reversed(self.dimensions) ))),

   normalizedArithCost := self >> let(n := self.params[1].n,
       nlogn := n * log(n) / log(2),
       When(self.params[3].dim() = 1, floor(5*nlogn), floor(2.5*nlogn)))
));


Class(ASP, rec(
  rot := 1,
  __call__ := (self, rot) >> WithBases(self, rec(rot := rot, operations := PrintOps)),
  print := self >> Print(self.__name__, "(", self.rot, ")"),

  DFT    := (self, n) >> ASPF(XN_min_1(n,self.rot),  Time_TX, Freq_1(1)),
  RDFT   := (self, n) >> ASPF(XN_min_1(n,self.rot),  Time_TX, Freq_E(1,1)),
  URDFT  := (self, n) >> ASPF(XN_min_1U(n,self.rot), Time_TX, Freq_E(1,1)),
  DHT    := (self, n) >> ASPF(XN_min_1(n,self.rot),  Time_TX, Freq_H(1,1)),
  UDHT   := (self, n) >> ASPF(XN_min_1U(n,self.rot), Time_TX, Freq_H(1,1)),

  BRDFT  := (self, n) >> ASPF(XN_min_1(n,self.rot),  Time_TX, Freq_T(1,1)),
  MBRDFT := (self, n) >> ASPF(XN_min_1(n,self.rot),  Time_SX, Freq_S(1,1)), # Murakami BRDFT variant

  IRDFT  := (self, n) >> ASPF(XN_min_1(n,self.rot),  Time_TX, Freq_E(1,2)).transpose(),
  IURDFT := (self, n) >> ASPF(XN_min_1U(n,self.rot), Time_TX, Freq_E(1,2)).transpose(),

  DFT2   := (self, n) >> ASPF(XN_min_1(n,self.rot), Time_TX,  Freq_1H(1)),
  RDFT2  := (self, n) >> ASPF(XN_min_1(n,self.rot), Time_TX,  Freq_EH(1,1)),
  DHT2   := (self, n) >> ASPF(XN_min_1(n,self.rot), Time_TX,  Freq_HH(1,1)),
  BRDFT2 := (self, n) >> ASPF(XN_min_1(n,self.rot), Time_TX,  Freq_TH(1,1)),
  SBRDFT2:= (self, n) >> ASPF(XN_min_1(n,self.rot), Time_TX,  Freq_THU(1,1)),

  DFT3   := (self, n) >> ASPF(XN_plus_1(n,self.rot), Time_TX, Freq_1(1)),
  RDFT3  := (self, n) >> ASPF(XN_plus_1(n,self.rot), Time_TX, Freq_E(1,1)),
  DHT3   := (self, n) >> ASPF(XN_plus_1(n,self.rot), Time_TX, Freq_H(1,1)),
  BRDFT3 := (self, n) >> ASPF(XN_plus_1(n,self.rot), Time_TX,  Freq_T(1,1)),
  MBRDFT3:= (self, n) >> ASPF(XN_plus_1(n,self.rot), Time_SX, Freq_S(1,1)), # Murakami BRDFT variant

  BDFT   := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_EX, Freq_1(1)),
  rDFT   := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_EX, Freq_E(1,1)),
  rDFTII := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_EX, Freq_EH(1,1)),
  rDHT   := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_HX, Freq_H(1,1)),
  rDHTII := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_HX, Freq_HH(1,1)),

  brDFT   := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_SX, Freq_E(1,1)),
  brDFTII := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_SX, Freq_EH(1,1)),
  brDHT   := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_SX, Freq_H(1,1)),
  brDHTII := (self, n,a) >> ASPF(XN_skew(n, a,self.rot), Time_SX, Freq_HH(1,1)),

  bRDFT3 := (self, n,a) >> ASPF(XN_skew(n,a,self.rot),  Time_SX, Freq_S(1,1)), 

  skewSS := (self, n,a) >> ASPF(XN_skew(n,a,self.rot), Time_SX, Freq_S(1,1)), 
  skewTT := (self, n,a) >> ASPF(XN_skew(n,a,self.rot), Time_TX, Freq_T(1,1)), 

  DFT4   := (self, n) >> ASPF(XN_plus_1(n,self.rot), Time_TX, Freq_1H(1)),
  RDFT4  := (self, n) >> ASPF(XN_plus_1(n,self.rot), Time_TX, Freq_EH(1,1)),
  DHT4   := (self, n) >> ASPF(XN_plus_1(n,self.rot), Time_TX, Freq_HH(1,1)),

  BRDFT4 := (self, n,a) >> ASPF(XN_skew(n,a, self.rot), Time_TX, Freq_TH(1,1)),
  SBRDFT4:= (self, n,a) >> ASPF(XN_skew(n,a, self.rot), Time_TX, Freq_THU(1,1)),
));

Class(TCodelet, Tagged_tSPL_Container);

_mid := F -> When(ObjId(F)=Freq_1, Freq_E, ObjId(F));

Freq_H.fixbot := k -> Prm(flipBot(k));
Freq_E.fixbot := k -> Diag(BHD(k, 1.0, -1.0));
Freq_1.fixbot := k -> I(2*k); 
Freq_S.fixbot := k -> I(2*k);
Freq_T.fixbot := k -> I(2*k);
# NOTE: what is it for others?
# NOTE: explain fixbot

nsFiltered := (lst, func) -> Cond(
    IsValue(lst), Filtered(lst.v, func),
    IsSymbolic(lst), lst, 
    Filtered(lst, func)
);

ASPF.isType1 := self >> ObjId(self.params[1]) in [XN_min_1, XN_min_1U];
ASPF.isType3 := self >> ObjId(self.params[1]) in [XN_plus_1];
ASPF.isSkew  := self >> ObjId(self.params[1]) in [XN_skew];
ASPF.size := self >> self.params[1].n;


ASPF_Breakdown_Rule := rec(
    a := rec(
        extraLeftTags := [],
        extraRightTags := [],
        inplaceTag := false,
	maxRadix := -1,
    ),

    apply    := (t, C, N) -> C[1],
    inplace := (self, x) >> When(self.getA("inplaceTag"), Inplace(x), x),
    leftTags := (self, t) >> Concatenation(t.tags, self.getA("extraLeftTags", [])),
    forTransposition := true,    
);

Class(even, AutoFoldExp, rec(
    ev := self >> let(a := self.args[1].ev(), 
        Cond(not IsInt(a), Error("even(<n>) works only with integer <n>"), 
             IsEvenInt(a), 1, 
             0)),
    computeType := self >> TInt
));

Class(odd, AutoFoldExp, rec(
    ev := self >> let(a := self.args[1].ev(), 
        Cond(not IsInt(a), Error("odd(<n>) works only with integer <n>"), 
             IsOddInt(a), 1, 
             0)),
    computeType := self >> TInt
));


Class(firstEltRft, FuncClass, rec(
    def := (n,f) -> Checked(IsIntSym(n), IsFunction(f), let(d:=f.domain(), r:=f.range(), 
                   rec(n := n, N := 2*r-odd(n)))),
    domain := self >> self.params[1],
    range  := self >> 2*self.params[2].range() - odd(self.params[1]),

    lambda := self >> let(
        i := Ind(self.domain()), n := self.params[1], f := self.params[2], 
        fst := 1+even(n), o := odd(n),
        Lambda(i, cond(lt(i,fst), i,  imod(i+o, 2) + 2*f.at(idiv(i+o,2)) - o))),

    transpose := self >> self.__bases__[1](self.params[1], self.self.params[2].transpose())
));

Class(ASPF_CT1_DFT_Mat, TaggedNonTerminal, rec(
    isAuxNonTerminal := true,
    abbrevs := [ k -> Checked(IsPosIntSym(k), [k]) ],
    dims := self >> [2*self.params[1], 2*self.params[1]],
    isReal := self >> true,
    terminate := self >> let(k := self.params[1],
        mat := Mat([[1,1],[-E(4), E(4)]]),
        res := L(2*k, 2) * RC(Tensor(I(k/2), mat) * MM(2,k/2)) * Diag(BHD(k,1.0,-1.0)),
        Cond(self.transposed, res.transpose(), res)
    )
));

NewRulesFor(ASPF_CT1_DFT_Mat, rec(
    ASPF_CT1_DFT_Mat_terminate := rec(
        applicable := t -> true,
        freedoms := t -> [],
        child := (t, fr) -> [],
        apply := (self, t, C, Nonterms) >> t.terminate()
    )
));

NewRulesFor(ASPF, rec(
    # Type 1
    #
    ASPF_CT1_URFT := CopyFields(ASPF_Breakdown_Rule, rec(
        applicable := t -> let(n := t.size(), 
            logic_and(t.isType1(), logic_and(eq(n mod 2, 0), logic_neg(isPrime(idiv(n,2)))))),

        freedoms := (self, t) >> let(maxR := self.getA("maxRadix", -1), 
            [ nsFiltered(divisorsIntNonTriv(t.size()/2), x -> (maxR < 0) or (x <= maxR/2)) ]),

        child := (self, t, fr) >> let(
            ltags := self.leftTags(t), rtags := self.getA("extraRightTags", []), 
            A := t.params[1],  F    := t.params[3],  T    := t.params[2],
            k := fr[1],        Fmid := _mid(F),      Tmid := Fmid.timeBasis(),  
            m := A.n / (2*k),  j    := Ind(m-1),     aj   := fdiv(j+1, 2*m),   

            C1 := ASPF(ObjId(A) (2*k,     A.rot),  Time_TX, F),
            C2 := ASPF(XN_skew  (2*k, aj, A.rot),  Tmid,    F),
            C3 := ASPF(XN_min_1U(2*m,     A.rot),  T,       Fmid(1,1)),
            
	    [ GT(C1.withTags(ltags), fId(Cols(C1)), fId(Rows(C1)), []),
	      C2.withTags(ltags), 
	      GT(C3.withTags(rtags), GTVec, GTVec, [k]).withTags(t.tags), 
	      InfoNt(j) ]
        ),

	apply := (self, t, C, Nonterms) >> let(
            A := t.params[1],  F := t.params[3], j := Nonterms[4].params[1],
	    k := Cols(Nonterms[1])/2, m := Cols(Nonterms[3].params[1])/2,
            When(F.dim()=1, Scat(Refl0_u(m, 2*k)),      # if transform is complex
                            RC(Scat(Refl0_u(m, k)))) *  # if it is real
            DirectSum(
                C[1],
		IDirSum(j, F.fixbot(k) * C[2])) * 
            self.inplace(C[3])
        )
    )),

    ASPF_CT1Prm_URFT := CopyFields(~.ASPF_CT1_URFT, rec(
        child := (self, t, fr) >> let(
            ltags := self.leftTags(t), rtags := self.getA("extraRightTags", []), 
            A := t.params[1],    T := t.params[2],    F := t.params[3],
            k := fr[1],          Fmid := _mid(F),     Tmid := Fmid.timeBasis(),  
            m := div(A.n, 2*k),  j := Ind(m-1),       aj := fdiv(j+1, 2*m),   
            
            C1 := ASPF(ObjId(A) (2*k,     A.rot),  Time_TX, F),
            C2 := ASPF(XN_skew  (2*k, aj, A.rot),  Tmid,    F),
            C3 := ASPF(XN_min_1U(2*m,     A.rot),  T,       Fmid(1,1)),

	    [ GT(C1.withTags(ltags), fId(Cols(C1)), fId(Rows(C1)), []),
	      C2.withTags(ltags), 
	      GT(C3.withTags(rtags), GTVec, GTPar, [k]).withTags(t.tags), 
	      InfoNt(j) ]
	),
        apply := (self, t, C, Nonterms) >> let(
            A := t.params[1],  F := t.params[3], j := Nonterms[4].params[1],
            N := t.params[1].n,          k := div(Cols(C[1]), 2),   tr := Tr(k, 2),
            j := Nonterms[4].params[1],  m := div(N, 2*k),
            fixbot := t.params[3].fixbot(k), 

            When(F.dim()=1, Scat(Refl0_u(m, 2*k)),      # if transform is complex
                            RC(Scat(Refl0_u(m, k)))) *  # if it is real
	    DirectSum(
		C[1] * tr, 
		IDirSum(j, F.fixbot(k) * C[2] * tr)) *

            RC(Tr(k, m)) * C[3]
        )
    )),

    ASPF_CT1_DFT := CopyFields(ASPF_Breakdown_Rule, rec(
        applicable := t -> let(n := t.size(), 
            logic_and(t.isType1(), logic_and(eq(n mod 4, 0), logic_neg(isPrime(idiv(n,4)))))),

        freedoms := (self, t) >> let(maxR := self.getA("maxRadix", -1), 
            [ nsFiltered(divisorsIntNonTriv(t.size()/4), x -> (maxR < 0) or (x <= maxR/2)) ]),

        child := (self, t, fr) >> [let(
            ltags := self.leftTags(t), 
            A := t.params[1],  F    := t.params[3],  T    := t.params[2],
            k := 2*fr[1],      Fmid := _mid(F),      Tmid := Fmid.timeBasis(),  
            m := A.n / (2*k),  j    := Ind(m-1),     aj   := fdiv(j+1, 2*m),   

            mat := Mat([[1,1],[-E(4), E(4)]]),
            C1 := ASPF(ObjId(A) (2*k,     A.rot),  Time_TX, F),
            C2 := ASPF(XN_skew  (2*k, aj, A.rot),  Tmid,    CopyFields(F, rec(scale2d:=1/2*F.scale2d))),
            C3 := UDFT(2*m, A.rot), 
                          
            P := When(F.dim()=1, Scat(Refl0_u(m, 2*k)),      # if transform is complex
                                 RC(Scat(Refl0_u(m, k)))),   # if it is real
            Pt := When(F.dim()=1, Prm(Refl0_u(m, 2*k)),      # if transform is complex
                                 RC(Prm(Refl0_u(m, k)))),   # if it is real
            self.inplace(
                P * 
                DirectSum(
                    C1.withTags(ltags) * RC(L(k, 2)), 
                    IDirSum(j, F.fixbot(k) * C2.withTags(ltags) * PushL(ASPF_CT1_DFT_Mat(k)))
                ) *
                Pt
            ) * 
            RC(Tensor(I(k/2), C3) * 
               Tr(2*m,k/2))
        )]
    )),

    ASPF_CT1Odd_RFT := CopyFields(~.ASPF_CT1_URFT, rec(
        applicable := t -> let(n := t.size(), 
            logic_and(t.isType1(), logic_and(logic_neg(isPrime(n)), hasOddDivisors(n)))),

        freedoms := (self, t) >> let(maxR := self.getA("maxRadix", -1), 
            [ nsFiltered(oddDivisorsIntNonTriv(t.size()), x -> (maxR < 0) or (n/x <= maxR/2)) ]),

        # m is odd here
        child := (self, t, fr) >> [let(
            ltags := self.leftTags(t), 
            A := t.params[1],  T    := t.params[2],  F    := t.params[3],       N  := A.n,
            m := fr[1],        Fmid := _mid(F),      Tmid := Fmid.timeBasis(),  mf := (m-1)/2,
            k := N / m,        j    := ind(mf,1),    aj   := fdiv(j+1, m),      Nc := idiv(N+1, 2),
            kc:= idiv(k+1,2),
            C1 := ASPF(ObjId(A) (  k,     A.rot), Time_TX, F).withTags(ltags),
            C2 := ASPF(XN_skew  (2*k, aj, A.rot), Tmid,    F).withTags(ltags),
            C3 := ASPF(XN_min_1 (  m,     A.rot), T,       Fmid(1,1)),
            fst := 1 + even(k),
            cpx := F.dim()=1,
#           rp := fDirsum(fId(1+even(k)), fTensor(Refl0_odd(mf, k, 1), fId(2))), 
#           When(F.dim()=1, Scat(Refl0_odd(mf, 2*k, 0)), Scat(rp)) *
            SUM(
                Scat(Cond(cpx, HH(N,k,0,[m]), firstEltRft(k, HH(Nc,kc,0,[m])))) * C1 * Gath(HH(N, k, 0, [1])),

                GT(F.fixbot(k) * C2,   # reflect = N-2  in BH compensates for fAdd shift
                   HH(N, 2*k, k, [1, 2*k]),
                   Cond(cpx, BHH(N, 2*k, 1, [m,1], 2*N),
                             fCompose(HH(N, N-fst, fst, [1]), fTensor(BHH(Nc-1,k,0,[m,1], N-2), fId(2)))),
                   [mf])) *
            self.inplace(
                GT(C3, GTVec, GTVec, [k]).withTags(t.tags))
        )]
    )),

#     ASPF_CT1Inp_URFT := CopyFields(~.ASPF_CT1_URFT, rec(
#         child := (self, t, fr) >> let(
#             A := t.params[1],    T := t.params[2],    F := t.params[3],         r := A.rot,  
#             k := fr[1],          Fmid := _mid(F),     Tmid := Fmid.timeBasis(),  
#             m := div(A.n, 2*k),  j := Ind(m-1),       aj := fdiv(j+1, 2*m),   
            
#             [ ASPF(ObjId(A) (2*k, r),     Time_TX, F).withTags([ANoRecurse()]),
#               ASPF(XN_skew  (2*k, aj, r), Tmid,    F).withTags([ANoRecurse()]),
#               ASPF(XN_min_1U(2*m, r),     T,       Fmid(1,1)),
#               InfoNt(j) ]),

#         apply := (self, t, C, Nonterms) >> let(
#             N := t.params[1].n,         k := div(Cols(C[1]), 2), 
#             j := Nonterms[4].params[1], m := div(N, 2*k),

#             RC(Scat(Refl0_u(m, k))) * 
#             DirectSum(BB(C[1]*Tr(k,2)), 
#                       IDirSum(j, BB(Diag(BHD(k,1,-1))*C[2]*Tr(k,2)*RC(MM(2,k/2))))) *
#             RC(Gath(Refl0_u(m, k))) * 
#             RC(condIOS(k*m, m))*
#             Tensor(I(k), C[3]) * Tr(2*m, k)
#         )
#     )),

    # Type 3
    #
    ASPF_CT3_RFT := CopyFields(ASPF_Breakdown_Rule, rec(
        aj := (j, a, m) -> fdiv(j+1/2, 2*m), 

        libApplicable := t -> logic_and(eq(imod(t.params[1].n, 2),0), logic_neg(isPrime(div(t.params[1].n,2)))),
        applicable := t -> let(A := ObjId(t.params[1]), n := t.params[1].n,
            A = XN_plus_1 and (IsSymbolic(n) or (n > 4 and IsEvenInt(n)))),

        freedoms := (self, t) >> let(n := t.params[1].n, maxR := self.getA("maxRadix", -1), 
            [ When(IsSymbolic(n), divisorsIntNonTriv(div(n,2)), 
                                  Filtered(divisorsIntNonTriv(div(n,2)).ev(), x -> (maxR < 0) or (x <= maxR/2))) ]),

        child := (self, t, fr) >> let(
            A := t.params[1],     T := t.params[2],  F := t.params[3],         r := A.rot,  
            k := fr[1],           Fmid := _mid(F),   Tmid := Fmid.timeBasis(), a := A.a, 
            m := div(A.n, 2*k),   j := Ind(m),       aj := self.aj(j, a, m), 

            C1 := ASPF(XN_skew(2*k, aj, r),                 Tmid, F),
            C2 := ASPF(CopyFields(A, rec(n:=_unwrap(2*m))), T,    Fmid(1,1)),

            [ C1.withTags(t.tags), GT(C2, GTVec, GTVec, [k]).withTags(t.tags), InfoNt(j) ]),

        apply := (self, t, C, Nonterms) >> let( 
            N := t.params[1].n,         k := div(Cols(C[1]), 2), 
            j := Nonterms[3].params[1], m := div(N, 2*k),  
            When(t.params[3].dim()=1, 
                 Scat(Refl1(m, 2*k)) *   IDirSum(j,                   C[1]) * C[2],
                 RC(Scat(Refl1(m, k))) * IDirSum(j, Diag(BHD(k,1.0,-1.0))*C[1]) * C[2]))
    )),

    # Skew
    #
    ASPF_CTSkew_RFT := CopyFields(~.ASPF_CT3_RFT, rec(
        aj := (j, a, m) -> fdiv(j+a, m), 

        applicable := t -> let(A := ObjId(t.params[1]), n := t.params[1].n,
            A = XN_skew and (IsSymbolic(n) or (n > 4 and IsEvenInt(n)))),

        apply := (self, t, C, Nonterms) >> let( 
            N := t.params[1].n,         k := div(Cols(C[1]),2), 
            j := Nonterms[3].params[1], m := div(N, 2*k),  
            When(t.params[3].dim()=1, 
                 Tr(m, 2*k)   * IDirSum(j, C[1]) * C[2],
                 RC(Tr(m, k)) * IDirSum(j, C[1]) * C[2]))
    ))
));

NewRulesFor(ASPF, rec(
    ASPF_CT1_URFT_LftInplace := CopyFields(ASPF_Breakdown_Rule, rec(
        switch := false,

        applicable := t -> let(A := ObjId(t.params[1]), n := t.params[1].n,
            A in [XN_min_1, XN_min_1U] and
            (IsSymbolic(n) or (n > 2 and IsEvenInt(n))) and
            ObjId(t.params[3]) = Freq_1),

        freedoms := (self, t) >> let(n := t.params[1].n, maxR := self.getA("maxRadix", -1), 
            [ Filtered(DivisorsIntNonTrivSym(div(n,2)), x -> (maxR < 0) or (x <= maxR/2)) ]),

        child := ASPF_CT1_URFT.child,

        apply := (self, t, C, Nonterms) >> let(
            N := t.params[1].n, k := Cols(C[1])/2, j := Nonterms[4].params[1], 
	    m := div(N, 2*k), 
            cplx := ObjId(t.params[3])=Freq_1, 
	    jj := Ind( Int(j.range/2) ),
	    pp := When(IsOddInt(m), DirectSum(I(1), condM(m-1, (m-1)/2)),
		                    condMp(m, m/2)),
            # NOTE:  add a GT_IJ' transform (ie with IJ' on either side), and an inplace rule for it
            #        also  GT_IJ transforms (ie with IJ on either side), and an inplace rule for it
	    When(not cplx, Error("not implemented"), #  RC(condKp(div(n,2), k)) 
                Inplace(
		    DirectSum(
		        C[1], 
		        When(IsOddInt(m), [], Data(j, V( (j.range-1)/2), C[2])),
		        When(jj.range = 0, [],
			    IDirSum(jj, 
			        condM(4*k, 2)*L(4*k,2*k) * 
			        DirectSum(Data(j, jj, C[2]), Data(j, j.range-1-jj, C[2])))))
		    ^ (L(N, m) * Tensor(I(2*k), pp))) *
		Grp(L(N, 2*k) * C[3])))
    ))
));

#
# Base cases
#
NewRulesFor(ASPF, rec(
    ASPF_Base2 := rec(
        requiredFirstTag := ANoTag,
        forTransposition := true,
        applicable := t -> t.params[1].n = 2, 
        apply := (t, C, Nonterms) -> t.terminate()
    ),
    
    # This rules provides the base case for any complex transform (ie for any of the algebras)
    # via the real transform.
    #
    # The trouble here, is that in different cases there are some 1d spectral components
    # in the real transform already, and that makes permutations different.
    #
    ASPF_SmallCpx := rec(
        requiredFirstTag := ANoTag,
        forTransposition := true,
        a := rec(maxSize := 512),
        applicable := (self, t) >> ObjId(t.params[3]) = Freq_1 and t.params[1].n <= self.getA("maxRadix", -1), 
        children := t -> let(F := t.params[3], s := F.scale1d, 
            [[ ASPF(t.params[1], t.params[2], Freq_E(s, s)) ]]),

        apply := (t, C, Nonterms) -> let(
            A := t.params[1], n := EvalScalar(A.n), 
            bcK := nn -> K(nn, 2) * Tensor(I(nn/2), Mat([[1, E(4)], [1, -E(4)]])), 
            bcL := nn -> L(nn, 2) * Tensor(I(nn/2), Mat([[1, E(4)], [1, -E(4)]])), 
                 # x^n-1
            Cond(ObjId(A) = XN_min_1 and IsEvenInt(n),
		   When(n=2, I(2), DirectSum(I(1), Z(n/2, 1), I(n/2-1)) * DirectSum(I(2), bcK(n-2))) * C[1],
                 ObjId(A) = XN_min_1 and IsOddInt(n),
                   DirectSum(I(1), bcK(n-1)) * C[1],
                 # x^n+1
                 ObjId(A) = XN_plus_1 and IsEvenInt(n), 
                   bcK(n) * C[1],
                 ObjId(A) = XN_plus_1 and IsOddInt(n),
                   DirectSum(I((n-1)/2), Z((n+1)/2, -1)) * DirectSum(bcK(n-1), I(1)) * C[1],
                 # skew 
                 ObjId(A) = XN_skew,  bcL(n) * C[1]
            ))
    ),

    ASPF_NonSkew_Base_VecN := rec(
       requiredFirstTag := [AVecReg, AVecRegCx],
       forTransposition := false,
       applicable := t -> let(v:=t.firstTag().v, alg := t.params[1],
	   ObjId(alg) in [XN_min_1, XN_min_1U, XN_plus_1] and 
	   2 <= alg.n and alg.n <= 2*4*v
       ),
       apply := (t, C, Nonterms) -> VectorizedMatSPL(t.firstTag().isa, t)
    ),
    # need a separate rule because VectorizedMatSPL returns a non-transposeable result,
    ASPF_NonSkew_Base_VecN_tr := rec(
       requiredFirstTag := [AVecReg, AVecRegCx],
       forTransposition := false,
       transposed := true,
       applicable := t -> let(v:=t.firstTag().v, alg := t.params[1],
	   ObjId(alg) in [XN_min_1, XN_min_1U, XN_plus_1] and 
	   2 <= alg.n and alg.n <= 2*4*v
       ),
       apply := (t, C, Nonterms) -> VectorizedMatSPL(t.firstTag().isa, t)
    ),

    ASPF_BRDFT3_Base4 := rec(
        requiredFirstTag := ANoTag,
        forTransposition := true,
        applicable := t -> t.hashAs() = ASP.bRDFT3(4,1/16),  # skew versionm but rule is probably invalid up to a permutation
        apply := (t, C, Nonterms) -> let(
	   rot := t.params[1].rot, a := t.params[1].a, D := Dat1d(TReal, 2), 
           s := t.params[3].scale2d, dd := When(s=1, I(2), Diag(s, 1)), 
	   Data(D, fPrecompute(FList(TReal, [2*s*cospi(rot*a), s*(4*cospi(rot*a)^2 - 1)])), 
           L(4,2) * VStack(
	       F(2) * dd * Mat([[1,   0,   -1,        0    ],
		                [0,   0,    0,    -nth(D,0)]]),
	       F(2)   *    Mat([[0,   s,    0,     nth(D,1)],
		                [0,   0,  nth(D,0),   0    ]]))))
    ),

    ASPF_URDFT_Base4 := rec(
        requiredFirstTag := ANoTag,
        forTransposition := true,
        applicable := t -> t.hashAs() = ASP.URDFT(4),
        apply := (t, C, Nonterms) -> let(rot := EvalScalar(t.params[1].rot) mod 4,
            #NoPull
            BB(
                Cond(rot=1, I(4), Diag(1,1,1,-1)) * 
                Tensor(t.params[3].scale2d * F(2), I(2))))
    ),

    ASPF_URDFT_Base4_Vec2 := rec(
        requiredFirstTag := [AVecReg, AVecRegCx],
        forTransposition := true,
        applicable := t -> t.hashAs() = ASP.URDFT(4).withTags(t.tags) and t.firstTag().v = 2,
        apply := (t, C, Nonterms) -> 
            When(EvalScalar(t.params[1].rot)=1, 
                VTensor(t.params[3].scale2d * F(2), 2),
                DirectSum(VBase(I(2), 2), VDiag(FList(TReal, [1,-1]), 2)) *
                VTensor(t.params[3].scale2d * F(2), 2))
    ),

    ASPF_RDFT1_Base4 := rec(
        requiredFirstTag := ANoTag, 
        forTransposition := true,
        applicable := t -> t.hashAs() = ASP.RDFT(4),
        apply := (t, C, Nonterms) -> let(rot := t.params[1].rot,
            DirectSum(F(2), When(rot=1, I(2), Diag(1,-1))) * 
            Tensor(Diag(t.params[3].scale1d, t.params[3].scale2d) * F(2), I(2)))
    ),

    ASPF_RDFT_toPRDFT := rec(
        requiredFirstTag := ANoTag, 
        forTransposition := true,
        a := rec(maxSize := false), 

        applicable := (self, t) >> let(n:=t.params[1].n, 
            not Is2Power(n) and 
            (self.getA("maxSize")=false or n <= self.getA("maxSize")) and 
            When(IsEvenInt(n), t.hashAs() in [ASP.RDFT(n), ASP.URDFT(n)], 
                               t.hashAs() = ASP.RDFT(n))), 

        freedoms := (self, t) >> [],
        child := (self, t, fr) >> [ PRDFT(EvalScalar(t.params[1].n), EvalScalar(t.params[1].rot)) ],

        apply := (t, C, Nonterms) -> let(n := EvalScalar(t.params[1].n),
            rdft := Perm_CCS(n) * C[1],
            When(IsEvenInt(n) and t.hashAs() = ASP.URDFT(n), 
                 DirectSum((1/2)*F(2), I(n-2)) * rdft,
                 rdft))
    ),

    ASPF_RDFT1_Base4_Vec2 := rec(
        requiredFirstTag := [AVecReg, AVecRegCx],
        forTransposition := true,
        applicable := t -> t.hashAs() = ASP.RDFT(4).withTags(t.tags) and t.firstTag().v = 2,
        apply := (t, C, Nonterms) -> let(rot := t.params[1].rot, vt := TVect(TReal, 2),
            DirectSum(VBlk( [[ vt.value([1, -1]), vt.value([1,1]) ]], 2) * _VVStack([VTensor(I(1), 2), VIxJ2(2)], 2),
                      When(rot=1, VBase(I(2), 2), VDiag(FList(TReal, [1,-1]), 2))) * 
            VTensor(Diag(t.params[3].scale1d, t.params[3].scale2d) * F(2), 2))
    ),

    ASPF_RDFT1_toTRDFT := rec(
        requiredFirstTag := [AVecReg, AVecRegCx],
        forTransposition := true,
	a := rec(maxSize := 512),
        applicable := (self, t) >> let(n:=t.params[1].n, 
	    t.hashAs() = ASP.RDFT(n).withTags(t.tags) and n <= self.getA("maxSize", 512)),
	freedoms := (self, t) >> [],
	child := (self, t, fr) >> [ TRDFT(t.params[1].n, t.params[1].rot).withTags(t.tags) ],
        apply := (t, C, Nonterms) -> let(
	    n := t.params[1].n,
            s  := Double(t.params[3].scale1d),
            s2 := Double(t.params[3].scale2d),
	    Cond(s=1 and s2=1, C[1],
		 IsEvenInt(n), Diag(diagDirsum(fConst(TReal, 2, s), fConst(TReal, n-2, s2))) * C[1],
		 IsOddInt(n),  Diag(diagDirsum(fConst(TReal, 1, s), fConst(TReal, n-1, s2))) * C[1]
	    )
	)
    ),

    ASPF_rDFT_Base4 := rec(
       requiredFirstTag := ANoTag,
       forTransposition := true,
       applicable := t -> t.hashAs() = ASP.rDFT(4,1/16),
       apply := (t, C, Nonterms) -> let(
	   a := t.params[1].a,
           rot := t.params[1].rot,
           s := t.params[3].scale2d,
           NoPull(           #Diag(1,1,1,-1) * 
               Tensor(F(2), I(2)) *
               DirectSum(s*I(2), RCDiag(fPrecompute(FList(TReal, [s*cospi(rot*a), s*sinpi(rot*a)])))) *
               L(4,2)))
    ),

    ASPF_rDFT_toSkewDFT := rec(
        requiredFirstTag := [ANoTag, ATwidOnline], 
        forTransposition := true,
        a := rec(maxSize := 512),
        applicable := (self, t) >> let(n:=t.params[1].n, 
            IsEvenInt(n) and n >= 4 and n <= self.getA("maxSize", 512) and t.hashAs().setTags([]) = ASP.rDFT(n,1/16)),

        freedoms := (self, t) >> [],
        child := (self, t, fr) >> let(alg := t.params[1], 
            [ SkewDFT(alg.n/2, alg.a, alg.rot).withTags(t.getTags()) ]), 

        apply := (t, C, Nonterms) -> let(n:=t.params[1].n, s := t.params[3].scale2d,
            RC(s * C[1]) * L(n, n/2))
    ),

    ASPF_rDFT_BaseN := rec(
       requiredFirstTag := ANoTag,
       forTransposition := true,
       a := rec(maxSize := 512),
       applicable := (self, t) >> let(n:=t.params[1].n, 
           IsEvenInt(n) and n >= 4 and n <= self.getA("maxSize", 512) and t.hashAs() = ASP.rDFT(n,1/16)),

       freedoms := (self, t) >> [],
       child := (self, t, fr) >> [ DFT(EvalScalar(t.params[1].n/2), EvalScalar(t.params[1].rot)) ], 
       apply := (t, C, Nonterms) -> let(
           n := t.params[1].n, 
	   a := t.params[1].a,
           rot := t.params[1].rot,
           s := Double(t.params[3].scale2d), # NOTE: if Double() is not used, vector code uses _mm_set1_epi32(..) intead of _ps(..)
           j := Ind(n-2), 
           exp := rot*a*(fdiv(4,n)), 
           twid := Lambda(j, cond(neq(imod(j, 2),0), s*sinpi(exp*idiv(j+2,2)), s*cospi(exp*idiv(j+2,2)))),
           #NoPull
	   (
               RC(C[1]) * 
               # XXX NOTE: Buf below delays sucking in, terrible hack, 
               # due to VJamData performance problems, fix it right now
               # NOTE: fPrecompute inside diagDirsum causes twiddles to be vpacked on the fly in the code
               Buf(RCDiag(diagDirsum(fConst(TReal, 1, s), fConst(TReal, 1, 0.0), #FList(TReal, [s,0]),
                                      fPrecompute(twid)))) * 
               L(n,n/2)))
    ),

    ASPF_rDHT_BaseN := rec(
 #      requiredFirstTag := ANoTag,
       forTransposition := true,
       a := rec(maxSize := 512),
       applicable := (self, t) >> let(n:=t.params[1].n, 
           IsEvenInt(n) and n >= 4 and n <= self.getA("maxSize", 512) and t.setTags([]).hashAs() = ASP.rDHT(n,1/16)),

       freedoms := (self, t) >> [],
       child := (self, t, fr) >> [ DFT(EvalScalar(t.params[1].n/2), EvalScalar(t.params[1].rot)) ], 
       apply := (t, C, Nonterms) -> let(
           n := t.params[1].n, 
	   a := t.params[1].a,
           rot := t.params[1].rot,
           s := t.params[3].scale2d,
           j := Ind(n-2), 
           exp := rot*a*(fdiv(4,n)),
           twid := Lambda(j, cond(neq(imod(j, 2),0), s*sinpi(exp*idiv(j+2,2)), s*cospi(exp*idiv(j+2,2)))),

           NoPull(
               Tensor(I(n/2), Diag(1,-1)) *
               RC(C[1]) * 
               DirectSum(s*I(2), RCDiag(fPrecompute(twid))) *
               L(n,n/2) * Diag(diagDirsum(fConst(TReal, n/2, 1), fConst(TReal, n/2, -1)))))
    ),

    ASPF_DHT1_Base4 := rec(
#        requiredFirstTag := ANoTag, 
        forTransposition := true,
        applicable := t -> t.setTags([]).hashAs() = ASP.DHT(4),
        apply := (t, C, Nonterms) -> let(rot := t.params[1].rot, 
            DirectSum(F(2), F(2)*When(rot=1, I(2), Diag(1,-1))) *
            Tensor(Diag(t.params[3].scale1d, t.params[3].scale2d) * F(2), I(2)))
    ),

    ASPF_UDHT1_Base4 := rec(
#        requiredFirstTag := ANoTag, 
        forTransposition := true,
        applicable := t -> t.setTags([]).hashAs() = ASP.UDHT(4),
        apply := (t, C, Nonterms) -> let(rot := t.params[1].rot, 
            NoPull(
                DirectSum(I(2), F(2)*When(rot=1, I(2), Diag(1,-1))) * 
                Tensor(Diag(t.params[3].scale1d, t.params[3].scale2d) * F(2), I(2))))
    ),

    ASPF_rDHT_Base4 := rec(
       requiredFirstTag := ANoTag,
       forTransposition := true,
       applicable := t -> t.hashAs() = ASP.rDHT(4,1/16),
       apply := (t, C, Nonterms) -> let(
	   a := t.params[1].a,
           rot := t.params[1].rot,
           s := t.params[3].scale2d,
           NoPull(Diag(1,-1,1,-1) * 
               Tensor(F(2), I(2)) *
               DirectSum(s*I(2), RCDiag(fPrecompute(FList(TReal, [s*cospi(rot*a), s*sinpi(rot*a)])))) *
               L(4,2)*Diag(1,1,-1,-1)))),

    ASPF_rDFT_Base4_Vec2 := rec(
       requiredFirstTag := [AVecReg, AVecRegCx],
       forTransposition := true,
       applicable := t -> t.hashAs() = ASP.rDFT(4,1/16).withTags(t.tags) and t.firstTag().v = 2, 
       freedoms := t -> [],
       child := (t, fr) -> [TL(4,2,1,1).withTags(t.getTags())],
       apply := (t, C, Nonterms) -> let(
	   a := t.params[1].a, 
           rot := t.params[1].rot,
           s := t.params[3].scale2d,
           #VDiag(FList(TReal, [1,1,1,-1]), 2) * 
           VTensor(F(2), 2) * 
           C[1] * 
           VRCDiag(fPrecompute(VData(FList(TReal, [s, s*cospi(rot*a), 0, s*sinpi(rot*a)]), 2)), 2))
    ),

    ASPF_rDFT_Base_VecN_Drop := rec(
	requiredFirstTag := [AVecReg, AVecRegCx],
	forTransposition := true,
	applicable := t -> IsEvenInt(t.params[1].n) and t.hashAs() = ASP.rDFT(t.params[1].n, 1/16).withTags(t.tags), 

	freedoms := t -> [],
	child := (t, fr) -> [ t.withoutFirstTag() ],
	apply := (t, C, Nonterms) -> C[1]
    ),

    ASPF_rDFT_Base_VecN := rec(
	requiredFirstTag := [AVecReg, AVecRegCx],
	forTransposition := false,
	applicable := t -> IsEvenInt(t.params[1].n) and t.hashAs() = ASP.rDFT(t.params[1].n, 1/16).withTags(t.tags), 
	freedoms := t -> [],
	child := (t, fr) -> let(
            n := t.params[1].n, 
	    a := t.params[1].a,
            rot := t.params[1].rot,
            s := Double(t.params[3].scale2d), # NOTE: if Double() is not used, vector code uses _mm_set1_epi32(..) intead of _ps(..)
            j := Ind(n/2-1), 
            exp := rot*a*(fdiv(4,n)), 
            twid := Lambda(j, s*omegapi(exp*(j+1))), 
	    [ TConj(
                  TRC(DFT(n/2, rot) * 
		      TDiag(fPrecompute(diagDirsum(fConst(TReal, 1, s), twid)))),  # NOTE: fPrecompute inside diagDirsum causes twiddles to be vpacked on the fly in the code
	          fId(n), L(n, n/2)
	      ).withTags(t.tags) ]
        ),
	apply := (t, C, Nonterms) -> C[1]
    ),

    ASPF_rDFT_Base_VecN_tr := rec(
	requiredFirstTag := [AVecReg, AVecRegCx],
	forTransposition := false,
	transposed := true,
	applicable := t -> IsEvenInt(t.params[1].n) and t.hashAs() = ASP.rDFT(t.params[1].n, 1/16).withTags(t.tags).transpose(), 
	freedoms := t -> [],
	child := (t, fr) -> let(
            n := t.params[1].n, 
	    a := t.params[1].a,
            rot := t.params[1].rot,
            s := Double(t.params[3].scale2d), # NOTE: if Double() is not used, vector code uses _mm_set1_epi32(..) intead of _ps(..)
            j := Ind(n/2-1), 
            exp := rot*a*(fdiv(4,n)), 
            twid := Lambda(j, s*omegapi(exp*(j+1))), 
	    [ TConj(
                  TRC( TDiag(fPrecompute(diagDirsum(fConst(TReal, 1, s), twid))).conjTranspose() *
                       DFT(n/2, rot).conjTranspose()
                       ),  # NOTE: fPrecompute inside diagDirsum causes twiddles to be vpacked on the fly in the code
	          L(n, 2), fId(n)
	      ).withTags(t.tags) ]
        ),
	apply := (t, C, Nonterms) -> C[1]
    ),

    ASPF_Cpx_rDFT_Base_VecN := rec(
	requiredFirstTag := [AVecReg, AVecRegCx],
	forTransposition := true,
	applicable := t -> IsEvenInt(t.params[1].n) and 
	    ObjId(t.params[1])=XN_skew and t.params[2]=Time_EX and 
	    ObjId(t.params[3])=Freq_1,

	freedoms := t -> [],
	child := (t, fr) -> let(
            n := t.params[1].n, 
	    a := t.params[1].a,
            rot := t.params[1].rot,
            s := Double(t.params[3].scale1d), # NOTE: if Double() is not used, vector code uses _mm_set1_epi32(..) intead of _ps(..)
            j := Ind(n/2-1), 
            exp := rot*a*(fdiv(4,n)), 
            twid := Lambda(j, s*omegapi(exp*(j+1))), 
	    mat := Mat([[1, E(4)], [1, -E(4)]]),
	    [ GT(mat, GTVec, GTVec, [n/2]).withTags(t.tags) *
	      TConj(
                  TRC(DFT(n/2, rot) * 
		      TDiag(fPrecompute(diagDirsum(fConst(TReal, 1, s), twid)))),  # NOTE: fPrecompute inside diagDirsum causes twiddles to be vpacked on the fly in the code
	          L(n, 2), L(n, n/2)
	      ).withTags(t.tags) ]
        ),
	apply := (t, C, Nonterms) -> C[1]
    ),
  
#     ASPF_rDFT_Base4_Vec2b := rec(
#        requiredFirstTag := [AVecReg, AVecRegCx],
#        forTransposition := true,
#        applicable := t -> t.hashAs() = ASP.rDFT(4,1/16).withTags(t.tags) and t.firstTag().v = 2, 
#        apply := (t, C, Nonterms) -> let(
# 	   a := t.params[1].a, 
#            rot := t.params[1].rot,
#            s := t.params[3].scale2d,
#            VTensor(F(2), 2) *
#            DirectSum(VBase(I(2), 2), VBase(J(2), 2)*VDiag(FList(TReal, [1,-1]), 2)) * 
# 	   VTensor(Mat([[1,0],[0,1]]),2) * 
#            VRCDiag(fPrecompute(VData(FList(TReal, [s, s*sinpi(rot*a), 0, -s*cospi(rot*a)]), 2)), 2))
#     ),
    
    ASPF_RDFT3_Base4 := rec(
       requiredFirstTag := ANoTag,
       forTransposition := true,
       applicable := t -> t.hashAs() = ASP.RDFT3(4), 
       apply := (t, C, Nonterms) -> let(
	   D := Dat1d(TReal, 2), d := Diag(1,-1), s := t.params[3].scale2d,
           rot := t.params[1].rot, pm := s*sinpi(rot*1/2), # pm = +/- s 
           NoPull(
	   Data(D, fPrecompute(FList(TReal, [s*cospi(rot*1/4), s*sinpi(rot*1/4) ])), 
               L(4,2) * 
               DirectSum(F(2)*Diag(1,nth(D,0)), d*F(2)*Diag(1, nth(D,1))) * 
               Mat([[s,  0,  0,   0 ],
                    [0,  1,  0,  -1 ],
                    [0,  0,  pm,  0 ],
                    [0,  1,  0,   1 ]]))))
    )
));
