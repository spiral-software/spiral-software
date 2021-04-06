
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(IMDCT_Odd, IMDST_Odd);

Class(MDCTBase, DTTBase, rec(
    dims := self >> [ self.params, 2*self.params ],
    SmallRandom := () -> Random([2,4,6,8,12,16,18,24])
));

Class(IMDCTBase, DTTBase, rec(
    dims := self >> [ 2*self.params, self.params ],
    SmallRandom := () -> Random([2,4,6,8,12,16,18,24])
));

#F MDCT_Odd(<n>) - Modified Discrete Cosine Transform non-terminal, (oddly-stacked system)
#F Definition: (n x 2n)-matrix [ cos((2i+1)(2j+1+n)/(4n)) | i = 0..n-1, j = 0..2n-1 ]
#F Note:       MDCT_Odd is the transpose of IMDCT_Odd
#F Example:    MDCT_Odd(8)
#F 
Class(MDCT_Odd, MDCTBase, rec(
    terminate := self >> let(n := self.params, Mat(
	List([0 .. n-1], i ->
	    List([0 .. 2*n - 1], j -> 
		CosPi((2*j + 1 + n)*(2*i + 1)/(4*n)))))),

    transpose := self >> IMDCT_Odd(self.params), 
));

MDCT := MDCT_Odd;

#F MDST_Odd(<n>) - Modified Discrete Sine Transform non-terminal, (oddly-stacked system)
#F Definition: (n x 2n)-matrix [ sin((2i+1)(2j+1+n)/(4n)) | i = 0..n-1, j = 0..2n-1 ]
#F Note:       MDST_Odd is the transpose of IMDST_Odd
#F Example:    MDST_Odd(8)
#F 
Class(MDST_Odd, MDCTBase, rec(
    terminate := self >> let(n := self.params, Mat(
	List([0 .. n-1], i ->
	    List([0 .. 2*n - 1], j -> 
		SinPi((2*j + 1 + n)*(2*i + 1)/(4*n)))))),

    transpose := self >> IMDST_Odd(self.params), 
));


#F MDCT_Even(<n>) - Modified Discrete Cosine Transform non-terminal, (evenly-stacked system)
#F Definition: (n   x 2n)-matrix [ cos((2i)(2j+1+n)/(4n)) | i = 0..n-1, j = 0..2n-1 ], n even
#F             (n+1 x 2n)-matrix [ cos((2i)(2j+1+n)/(4n)) | i = 0..n,   j = 0..2n-1 ], n odd
#F
#F Example:    MDCT_Even(8)
#F 
Class(MDCT_Even, MDCTBase, rec(
    dims := self >> [ self.params + When(IsOddInt(self.params), 1, 0), 2*self.params ],

    terminate := self >> let(n := self.params, isodd := When(IsOddInt(n), 1, 0), Mat(
	List([0 .. n-1 + isodd], i ->
	    List([0 .. 2*n - 1], j -> 
		CosPi((2*j + 1 + n)*(2*i)/(4*n)))))),

#    transpose := self >> IMDCT_Even(self.params), 
));

#F MDST_Even(<n>) - Modified Discrete Sine Transform non-terminal, (evenly-stacked system)
#F Definition: (n   x 2n)-matrix [ sin((2i)(2j+1+n)/(4n)) | i = 1..n,   j = 0..2n-1 ], n even
#F             (n-1 x 2n)-matrix [ sin((2i)(2j+1+n)/(4n)) | i = 1..n-1, j = 0..2n-1 ], n odd
#F
#F Example:    MDST_Even(8)
#F 
Class(MDST_Even, MDCTBase, rec(
    dims := self >> [ self.params - When(IsOddInt(self.params), 1, 0), 2*self.params ],

    terminate := self >> let(n := self.params, isodd := When(IsOddInt(n), 1, 0), Mat(
	List([1 .. n-isodd], i ->
	    List([0 .. 2*n - 1], j -> 
		SinPi((2*j + 1 + n)*(2*i)/(4*n)))))),

#    transpose := self >> IMDST_Even(self.params), 
));


#F IMDCT_Odd(<n>) - Inverse Modified Discrete Cosine Transform non-terminal (oddly-stacked system)
#F Definition: (2n x n)-matrix [ cos((2j+1)(2i+1+n)/(4n)) | i=0..2n-1, j=0..n-1 ]
#F Note:       IMDCT_Odd is the transpose of MDCT_Odd
#F Example:    IMDCT_Odd(8)
#F 
Class(IMDCT_Odd, IMDCTBase, rec(
    terminate := self >> let(n:=self.params, Mat(
	List([0 .. 2*n-1], i ->
	    List([0 .. n-1], j -> 
		CosPi((2*i + 1 + n)*(2*j + 1)/(4*n)))))),

    transpose := self >> MDCT_Odd(self.params), 
));

IMDCT := IMDCT_Odd;

#F IMDST_Odd(<n>) - Inverse Modified Discrete Sine Transform non-terminal (oddly-stacked system)
#F Definition: (2n x n)-matrix [ cos((2j+1)(2i+1+n)/(4n)) | i=0..2n-1, j=0..n-1 ]
#F Note:       IMDST_Odd is the transpose of MDST_Odd
#F Example:    IMDST_Odd(8)
#F 
Class(IMDST_Odd, IMDCTBase, rec(
    terminate := self >> let(n:=self.params, Mat(
	List([0 .. 2*n-1], i ->
	    List([0 .. n-1], j -> 
		SinPi((2*i + 1 + n)*(2*j + 1)/(4*n)))))),

    transpose := self >> MDST_Odd(self.params), 
));


_shiftcut := (transform, n, start, cut, shift, boundary) -> Checked(cut in ["re", "im"], 
    let(ofs := When(cut="re", 0, 1),
	N   := Cols(transform),
	i   := Ind(N),

	Gath(H(Rows(transform), n, ofs+start, 2)) * 
	transform * 
	When(boundary=1, I(N), Diag(Lambda(i, cond(leq(i,shift-1), boundary, 1)))) *
	Z(N, -shift))
);

RulesFor(MDCT_Odd, rec(
    #F MDCT_toDCT4:  MDCT_n = DCT4_n * sums
    #F
    MDCT_Odd_toDCT4 := rec (
	info         := "MDCT_n -> DCT4_n",
	isApplicable := P -> P mod 2 = 0,
	allChildren  := P -> [[ DCT4(P) ]], 
	rule := (P, C) -> 
	    C[1] * 
	    J(P) *
	    DirectSum(Tensor(Mat([[1, -1]]), I(P/2)),
		      Tensor(Mat([[-1, -1]]), I(P/2))) *
	    DirectSum(J(P/2), I(P/2), I(P/2), J(P/2))
    ),

    MDCT_Odd_toPRDFT34 := rec(
	info         := "MDCT_n -> PRDFT4_2n or PRDFT3_2n",
	isApplicable := P -> true,
	allChildren  := P -> When(IsEvenInt(P), [[ PRDFT4(2*P) ]], [[ PRDFT3(2*P) ]]), 
	rule := (P, C) -> let(n:=P, 
	    _shiftcut(C[1], n, 0, "re", Int((n+1)/2), -1))
    )

));

RulesFor(MDST_Odd, rec(
    #F MDST_toDST4:  MDST_n = DST4_n * sums
    #F
    MDST_Odd_toDST4 := rec (
	info         := "MDST_n -> DST4_n",
	isApplicable := P -> P mod 2 = 0, 
        # NOTE: implement rule for n odd, converts to DST3
	allChildren  := P -> [[ DST4(P) ]], 
	rule := (P, C) -> let(n:=P,
	    C[1] * 
	    J(n) *
	    DirectSum(Tensor(Mat([[1, 1]]), I(n/2)),
		      Tensor(Mat([[1, -1]]), I(n/2))) *
	    DirectSum(J(n/2), I(n/2), I(n/2), J(n/2)))
    ),

    MDST_Odd_toPRDFT34 := rec(
	info         := "MDST_n -> PRDFT4_2n or PRDFT3_2n",
	isApplicable := P -> true,
	allChildren  := P -> When(IsEvenInt(P), [[ PRDFT4(2*P) ]], [[ PRDFT3(2*P) ]]), 
	rule := (P, C) -> let(n:=P, 
	    _shiftcut(C[1], n, 0, "im", Int((n+1)/2), -1))
    )

));


RulesFor(MDCT_Even, rec(
    MDCT_Even_toPRDFT12 := rec(
	info         := "MDCT_n -> PRDFT1_2n or PRDFT2_2n",
	isApplicable := P -> true,
	allChildren  := P -> When(IsEvenInt(P), [[ PRDFT2(2*P) ]], [[ PRDFT1(2*P) ]]), 
	rule := (P, C) -> let(n:=P, 
	    _shiftcut(C[1], n+(n mod 2), 0, "re", Int((n+1)/2), 1))
    )
));


RulesFor(MDST_Even, rec(
    MDST_Even_toPRDFT12 := rec(
	info         := "MDST_n -> PRDFT1_2n or PRDFT2_2n",
	isApplicable := P -> true,
	allChildren  := P -> When(IsEvenInt(P), [[ PRDFT2(2*P) ]], [[ PRDFT1(2*P) ]]), 
	rule := (P, C) -> let(n:=P, 
	    _shiftcut(C[1], n-(n mod 2), 2, "im", Int((n+1)/2), 1))
    )
));
