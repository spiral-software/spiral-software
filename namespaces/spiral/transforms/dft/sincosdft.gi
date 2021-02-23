
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F SinDFT( <size> )
#F
Class(SinDFT, NonTerminal, rec(
    abbrevs   := [ n -> Checked(IsInt(n), n > 0, n) ],
    dims      := self >> [ self.params, self.params ],
    terminate := self >> Mat(Global.SinDFT(self.params)),
    transpose := self >> self,
    isReal    := True
));

#F CosDFT( <size> )
#F
Class(CosDFT, NonTerminal, rec(
    abbrevs   := [ n -> Checked(IsInt(n), n > 0, n) ],
    dims      := self >> [ self.params, self.params ],
    terminate := self >> Mat(Global.CosDFT(self.params)),
    transpose := self >> self,
    isReal    := True
));

RulesFor(CosDFT, rec(
    #F CosDFT_Base: (base case)
    #F
    #F   CosDFT_2 = F_2
    #F   CosDFT_4 = ... can be done in 5 instad of 7 adds
    #F
    CosDFT_Base := rec (
	info             := "CosDFT_2 -> F_2",
	forTransposition := false,
	isApplicable     := P -> P in [2,4], 
	rule := (P, C) -> When(P=2, F(2), 
	    Mat(MatSPL(copy1(2) * sums3(2))) *
	    DirectSum(F(2), Mat([[1, 1]])) *
	    Perm((2,3), 4))
    ),

    #F CosDFT_Trigonometric: 1984
    #F
    #F   CosDFT_n = copy * sums * (CosDFT_n/2 dirsum DCT2_n/4) * sums * perm
    #F
    #F   Vetterli/Nussbaumer: 
    #F     Simple FFT and DCT Algorithms with Reduced Number
    #F     of Operations, Signal Processing, 1984, pp. 267--278
    #F
    CosDFT_Trigonometric := rec (
	info             := "CosDFT_n --> CosDFT_n/2, DCT2_n/4",
	isApplicable     := P -> P mod 4 = 0 and P > 4,

	allChildren := P -> [[ CosDFT(P/2), DCT2(P/4) ]],

	rule := (P, C) -> 
	    copy1(P/2) * sums3(P/2) *
	    DirectSum(C) *
	    DirectSum(I(P/2), sums2(P/4)) *
	    L(P, 2)
    )
));

RulesFor(SinDFT, rec(
    #F SinDFT_Base: (base case)
    #F
    #F   SinDFT_2 = NullMat_2
    #F   SinDFT_4 = (NullMat_2 dirsum block) ^ perm
    #F
    SinDFT_Base := rec (
	info             := "SinDFT_2 -> 0_2",
	forTransposition := false,
	isApplicable := P -> P in [2,4], 
	rule := (self, P, C) >> When(P = 2, 
		Mat([[0,0],[0,0]]),
		DirectSum(Mat([[0, 0], [0, 0]]),
		          Mat([[1], [-1]]) * Mat([[1, -1]])) ^ Perm((2,3), 4))
    ),

    #F SinDFT_Trigonometric: 1984
    #F
    #F   SinDFT_n = copy * sums * (SinDFT_n/2 dirsum DCT2_n/4) * sums * perm
    #F
    #F   Vetterli/Nussbaumer: 
    #F     Simple FFT and DCT Algorithms with Reduced Number
    #F     of Operations, Signal Processing, 1984, pp. 267--278
    #F
    SinDFT_Trigonometric := rec(
	info         := "SinDFT_n --> SinDFT_n/2, DCT2_n/4",
	isApplicable := P -> P mod 4 = 0 and P > 4, 

	allChildren := P -> [[ SinDFT(P/2), DCT2(P/4) ]],

	rule := (P, C) -> 
	    copy2(P/2) * sums5(P/2) *
	    DirectSum(C) *
	    DirectSum(I(P/2), sums4(P/4)) *
	    L(P, 2)
    )
));
