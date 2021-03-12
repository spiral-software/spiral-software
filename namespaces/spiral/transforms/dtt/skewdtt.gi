
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


normCos := normalizeCosine;
dnormCos := x -> When(IsDelay(x) and Type(x)=T_FUNCCALL and x[1] = D(normCos), 
	              x, PEV(normCos(x)));

dnormCos := x -> When(IsRat(x), normCos(x), x);

#F SkewDTT(<non-terminal for DCT/DST type 3/4>, <rat>)
#F Parameters:       [ <transform for DCT or dst type 3 or 4>, <rat> ]
#F Definition:       Let t be dct or dst type 3 or 4 as non-terminal spl.
#F                   Transform( "SkewDTT", [t, r] ) represents the (n x n)-matrix
#F                     [ cos(r_k*l*pi/n) | k,l = 0...n-1 ]
#F                     [ sin(r_k*(l+1)*pi/n) | k,l = 0...n-1 ]
#F                     [ cos(r_k*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F                     [ sin(r_k*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F                   for dct3, dst3, dct4, dst4, respectively
#F                   where the list L of the r_k arises from the list
#F                     [ (r + 2k)/n | 0 <= k < n ]
#F                   by
#F                     1. normalizing the entries to lie in [0, 1] such
#F 		       that their cosine does not change
#F                     2. ordering the values increasing by size
#F Note:             for r = 1/2, the skew DCT3 reduces to the ordinary DCT3.
#F                   The transpose is provided by the transform TSkewDTT
#F Example:          SkewDTT( DCT3(8), 1/3 )
#F
Class(SkewDTT, NonTerminal, rec(
    _short_print := true,
    abbrevs := [ (nt, r) -> Checked(IsNonTerminal(nt), IsRatSym(r), 
	                            ObjId(nt) in [DCT3, DST3, DCT4, DST4],
				    [Copy(nt), dnormCos(r)]) ],

    dims := self >> self.params[1].dimensions,

    terminate := self >> let(
	S := ObjId(self.params[1]),
	N := self.params[1].dimensions[1],
	r := EvalScalar(self.params[2]), 
	Cond(S = DCT3, Mat(SkewDCT_IIIunscaled(N, r)),
	     S = DST3, Mat(SkewDST_IIIunscaled(N, r)),
	     S = DCT4, Mat(SkewDCT_IVunscaled(N, r)),
	     S = DST4, Mat(SkewDST_IVunscaled(N, r)),
	     Error("Unrecognized <self.params[1].symbol>"))),

    isReal := True

#    HashId := self >> self.params[1]
));

RulesFor(SkewDTT, rec(
    SkewDTT_Base2 := rec(
        info             := "DTT_2(r) -> F_2",
        isApplicable     := P -> P[1].dimensions[1] = 2,
        allChildren      := P -> [ [ ] ],
        rule := (P,C) -> 
	    let(S := ObjId(P[1]),
		r := P[2],
		Cond(
		    S = DCT3,  F(2) * Diag([1, CosPi(r/2)]),

		    S = DST3,  F(2) * Diag([SinPi(r/2), SinPi(r)]),

		    S = DCT4,  Diag([CosPi(r/4), SinPi(r/4)]) * F(2) *
				 Mat([[1, -1], [0, 2*CosPi(r/2)]]),

		    S = DST4,  Diag([SinPi(r/4), CosPi(r/4)]) * F(2) *
				 Mat([[1, 1], [0, 2*CosPi(r/2)]])))
    ),

    SkewDCT3_VarSteidl := rec(
        info             := "SkewDCT3_n -> IterDirectSum(SkewDCT3_n/2)",
	switch           := true,
	forTransposition := false,
        isApplicable     := P -> let(S := ObjId(P[1]), n := Rows(P[1]), 
	    S = DCT3 and IsPrimePowerInt(n) and n mod 2 = 0 and n > 2),

	# create one parametrized SkewDTT, use SampledVar() to make timing possible
        allChildren := P -> let(n := Rows(P[1]), r := P[2],
	    [[ SkewDTT(DCT3(n/2), r/2), 
	       SkewDTT(DCT3(n/2), 1 - r/2) ]]), 

        rule := (P,C) -> let(n := Rows(P[1]), r := P[2], 
	    K(n, n/2) *
	    DirectSum(C[1], C[2]) * 
	    Tensor(F(2), I(n/2)) *
	    sparse1(n, r/2))
    ),

    # NOTE: We need to be able to time SkewDTT(n, r) where r is free variable
    #        otherwise DP does not work
    SkewDCT3_VarSteidlIterative := rec(
        info             := "SkewDCT3_n -> IterDirectSum(SkewDCT3_n/2)",
	switch           := false,
	forTransposition := false, 
        isApplicable     := P -> let(S := ObjId(P[1]), n := Rows(P[1]), 
	    S = DCT3 and IsPrimePowerInt(n) and n mod 2 = 0 and n > 2),

	# create one parametrized SkewDTT, use SampledVar() to make timing possible
        allChildren := P -> let(n := Rows(P[1]), r := P[2], i:=Ind(2),
	    [[ SkewDTT(DCT3(n/2), cond(i,1,0) + cond(i,-1,1)*r/2) ]]),

        rule := (P,C, Nonterms) -> let(
	    n := Rows(P[1]), 
	    r := P[2], 
	    ivar := Nonterms[1].params[2].args[1].args[1],

	    K(n, n/2) *
	    IterDirectSum(ivar, 2, C[1]) * 
	    Tensor(F(2), I(n/2)) *
	    sparse1(n, r/2))
    )
));

