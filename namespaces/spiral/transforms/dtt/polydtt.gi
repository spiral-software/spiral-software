
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F PolyDTT( <dtt-nonterminal> )
#F Parameters:       <nonterminal for a (optionally skew) DCT or DST of type 1--8>
#F Definition:       Let <t> be an (otionally skew) DCT or DST of type 1--8. Then
#F                   Transform( "PolyDTT", <t> ) represents the (n x n)-matrix
#F                   obtained from dividing every row in <t> by its first
#F                   entry.
#F                   This transform is called the polynomial version of
#F                   the DCT or DST.
#F Note:             The transpose of PolyDTT(<t>) is
#F                   TPolyDTT(<t>^T).
#F Example:          PolyDTT(DCT2(8)).
#F
Class(PolyDTT, NonTerminal, rec(
    _short_print := true,
    abbrevs := [ nt -> Checked(IsNonTerminal(nt), 
	    ObjId(nt) in 
	    [ DCT1, DCT3, DCT5, DCT7, # PolyDTT(T)=T for these
	      DCT2, DCT4, DCT6, DCT8, 
	      DST1, DST2, DST3, DST4,
	      DST5, DST6, DST7, DST8,
	      SkewDTT],
	    [ Copy(nt) ]
	)],
    dims := self >> self.params[1].dimensions,
    terminate := self >> Mat(PolynomialDTT(MatSPL(TerminateSPL(self.params[1])))),
    isReal := True
));

Class(fr, Exp, rec(
    ev := self >> let(
	m := self.args[1].ev(),
	i := self.args[2].ev(),
	r := self.args[3].ev(),
	Cond(i mod 2 = 0, (r + 2 * Int(i/2)) / m,
	     i mod 2 = 1, (2 - r + 2 * Int(i/2)) / m))));

#F Rules
#F -----
RulesFor(PolyDTT, rec(
    PolyDTT_ToNormal := rec(
	isApplicable := P -> ObjId(P[1]) in [DCT1, DCT3, DCT5, DCT7],
	allChildren := P -> [P[1]],
	rule := P -> P[1]
    ),

    #F PolyDTT_Base2: (base case for non-skew DTTs of size 2)
    #F
    #F   pDCT2_2 = F_2
    #F   pDCT4_2 = F_2 * [[1, -1], [0, sqrt(2)]]
    #F   pDCT6_2 = [ [ 1, 1 ], [ 1, -2 ] ]
    #F   pDCT8_2 = [ [ 1, cos(2*pi/5) ], [ 1, cos(4*pi/5) ] ]
    #F   pDST4_2 = F_2 * [[1, 1], [0, sqrt(2)]]
    #F
    #F   Pueschel/Moura: Discrete Cosine and Sine Transforms, in preparation
    #F
    PolyDTT_Base2 := rec (
	info           := "pDTT_2 -> F_2",
	isApplicable   := P -> Rows(P[1]) = 2 and ObjId(P[1]) <> SkewDTT,
	allChildren    := P -> [[ ]],
	rule := (P, C) -> let(nt := ObjId(P[1]), Cond(
		nt = DCT2, F(2),
		nt = DCT4, F(2) * Mat([[1, -1], [0,sqrt(2)]]),
		nt = DCT6, Mat([[1,1], [1,-2]]),
		nt = DCT8, Mat([[1,2*CosPi(2/5)], [1,2*CosPi(4/5)]]),
		nt = DST2, F(2),
		nt = DST3, F(2) * Diag([1, sqrt(2)]),
		nt = DST4, F(2) * Mat([[1, 1], [0, sqrt(2)]]),
		nt = DST5, Mat([[1,2*CosPi(2/5)], [1,2*CosPi(4/5)]]),
		nt = DST6, Mat([[1,2*CosPi(1/5)], [1,2*CosPi(3/5)]]),
		nt = DST7, Mat([[1,2*CosPi(1/5)], [1,2*CosPi(3/5)]]),
		nt = DST8, Mat([[1,2], [1,-1]]), 
		Error("unrecognized <L.symbol>"))
	)
    ),

   PolyDTT_SkewBase2 := rec (
	info           := "pDTT_2 -> F_2",
	isApplicable   := P -> Rows(P[1]) = 2 and ObjId(P[1]) = SkewDTT and
	                       ObjId(P[1].params[1]) = DST3, 
	allChildren    := P -> [[ ]],
	rule := (P, C) -> let(skewnt := ObjId(P[1].params[1]), r := P[1].params[2], 
	    Cond(
		skewnt = DST3, F(2) * Diag(1, 2*CosPi(r/2)),
		Error("unrecognized <L.symbol>"))
	)
   ),

   PolyDTT_SkewDST3_CT := rec(
       isApplicable := P -> ObjId(P[1]) = SkewDTT and
                            ObjId(P[1].params[1]) = DST3 and
                            Rows(P[1]) > 2 and not IsPrime(Rows(P[1])),

       allChildren := P -> let(
	   N := Rows(P[1].params[1]), r := P[1].params[2],
	   List(DivisorPairs(N), d->
	       let(i := Ind(d[2]),
		   ri := fr(d[2], i, r),
		   [ PolyDTT(SkewDTT(DST3(d[1]), ri)), 
		     PolyDTT(SkewDTT(DST3(d[2]), r)) ]))), 
		
       rule := (P,C) -> let(
	   MN := Rows(P[1]),  N := Rows(C[1]),  M := Rows(C[2]), 
	   r := P[1].params[2],
	   i := C[1].root.params.params[2].args[2],

	    K(MN, N) * 
	    IterDirectSum(i, i.range, C[1]) *
	    Tensor(C[2], I(N)) *
	    B_DST3_U(MN, M)
       )
   )
));
