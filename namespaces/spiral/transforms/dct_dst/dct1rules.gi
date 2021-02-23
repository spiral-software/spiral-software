
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_B_dct1 := n -> let(j:=Ind(n), VStack(
    RowVec(diagDirsum(FList(TReal, [2,1]), fConst(TReal, n-2, 0.0))), 
    HStack(ColVec(fConst(TReal, n-3, 0.0)), 
           RowTensor(n-3, 1, Mat([[1,1]])), 
           ColVec(fConst(TReal, n-3, 0.0))), 
    RowVec(diagDirsum(fConst(TReal, n-2, 0.0), FList(TReal, [1, 2]))), 
    RowVec(Lambda(j, cond(neq(imod(j, 2),0), -1, 1)))
));

RulesFor(DCT1, rec(
    #F DCT1_Base2: (base case for size 2)
    #F
    #F   DCT1_2 = F_2        (scaled)
    #F
    DCT1_Base2 := rec(
	info             := "DCT_1 -> F_2",
	forTransposition := false,
	isApplicable     := P -> P[1]=2, 
	allChildren      := P -> [[]],
	rule             := (P, C) -> F(2)
    ),

    #F DCT1_Base4: DCT1_4 = (M tensor F_2) ^ P
    #F
    DCT1_Base4 := rec(
	info             := "DCT_1 -> (M tensor F_2)^P",
	forTransposition := false,
	isApplicable     := P -> P[1]=4, 
	allChildren      := P -> [[]],
	rule := (P, C) -> 
	    Tensor(Mat([[1, 1], [1, -1/2]]), F(2)) ^ Perm((2,4), 4)
    ),

    #F DCT1_DCT1and3: 1984
    #F
    #F   DCT1_(n+1) = perm * (DCT1_(n/2+1) dirsum DCT3_n/2 ^ perm) * 
    #F             (1 tensor DFT_2)^perm
    #F
    #F   Wang: 
    #F     Fast Algorithms for the Discrete W Transform and the
    #F     Discrete Fourier Transform.
    #F     IEEE Trans. on ASSP, 1984, pp. 803--814
    #F
    DCT1_DCT1and3 := rec (
	info             := "DCT1_n --> DCT1_n/2, DCT3_n/2",
	isApplicable     := P -> (P[1] - 1) mod 2 = 0,
	allChildren := P -> 
	    When(P[1]=3, [[ DCT1(2) ]],
		      [[ DCT1((P[1]+1)/2), DCT3((P[1]-1)/2) ]]),

	rule := (P, C) ->
	    When(P[1]=3, 
		perm2(3) *
		DirectSum(C[1], I(1)) *
		DirectSum(I(1), F(2)) ^ perm4(3),

		perm2(P[1]) *
		DirectSum(C[1], C[2] ^ J((P[1]-1)/2)) *
		DirectSum(I(1), Tensor(I((P[1]-1)/2), F(2))) ^ perm4(P[1]))
    ),

    DCT1_toDCT2 := rec(
        isApplicable := P -> P[1] > 5, 
        allChildren := P -> [[DCT2(P[1]-1)]], 
        rule := (P, C) ->  let(
            n := P[1], j := Ind(n-1), nn := n-1, 
            DirectSum(Diag(Lambda(j, 1/(2*cospi(fdiv(j, 2*nn))))) * 
                      C[1], I(1)) * 
            _B_dct1(n)
        )
    )
));
