
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RulesFor(DCT4, rec(

    #F DCT4_Base2: DCT4_2 = (1,2) * R_13/8
    #F
    DCT4_Base2 := rec(
	info             := "DCT4_2 -> R_13/8",
	forTransposition := false,
	isApplicable     := P -> P[1] = 2, 
	rule := (P, C) -> J(2) * Rot(13/8)
    ),

    #F DCT4_Base3: (base case for size 3)
    #F
    #F   DCT4_2 = (2,3) * (F_2 dirsum 1) * (a dirsum M) * (F_2 dirsum 1) * (2,3)
    #F
    #F derived by hand starting with AREP's decomposition by mon-mon symmetry.
    #F
    DCT4_Base3 := rec (
	info             := "DCT4_3 -> F_2, M, F_2",
	forTransposition := false,
	isApplicable     := P -> P[1] = 3, 
	rule := (P, C) -> 
	    Perm((2,3), 3) * 
	    DirectSum(F(2), I(1)) *
	    DirectSum(Diag(Sqrt(3/8)), Sqrt(1/2)*Mat([[1/2, 1], [1, -1]])) *
	    DirectSum(F(2), I(1)) *
	    Perm((2,3), 3)
    ),

    #F DCT4_DCT2: DCT4_n = sums * DCT2_n * diag
    #F
    #F   inverse - transpose of RuleDCT4_2
    #F
    DCT4_DCT2 := rec (
	info             := "DCT4_n --> DCT2'_n",
	isApplicable     := P -> P[1] > 2,
	allChildren := P -> [[ DCT2(P[1]) ]],
	rule := (P, C) -> 
	    sums1(P[1]) *
	    C[1] *
	    Diag(List([0..P[1]-1], i -> 1/(2*CosPi((2*i+1)/(4*P[1])))))
    ),

    #F DCT4_DCT2t: 1985, DCT4_n = sums * DCT2_n * diag
    #F
    #F   Chan: 
    #F     Direct Methods for computing discrete sinusoidal transforms
    #F     IEE Proceedings, Vol. 137, 1990, pp. 433--442
    #F
    # switched off since critical path too long (sums6)
    # also there seems to be one mult too much
    DCT4_DCT2t := rec (
	info             := "DCT4_n --> DCT2'_n",
	switch           := false,
	isApplicable     := P -> P[1] > 2,
	allChildren      := P -> [[ DCT2(P[1]) ]],
	rule := ( P, C ) -> 
	    sums6(P[1]) *
	    C[1] *
	    Diag(List([0..P[1] - 1], i -> 2 * CosPi((2*i + 1)/(4*P[1]))))
    ),

    #F DCT4_DCT2andDST2: 1985
    #F
    #F   DCT4_n = blocks * perm * (DCT2_n/2 dirsum DST2_n/2) * rotations
    #F
    #F   Wang: (transposed)
    #F     On Computing the Discrete Fourier ans Cosine Transforms,
    #F     IEEE Trans. on Signal Proc., 1985, pp. 1341--1344
    #F
    DCT4_DCT2andDST2 := rec (
	info             := "DCT4_n --> DCT2'_n/2, DST2'_n/2",
	isApplicable     := P -> IsInt(P[1]) and P[1] > 2 and P[1] mod 2 = 0, 
	allChildren := P -> [[ DCT2(P[1]/2), DST2(P[1]/2) ]],
	rule := (P, C) -> 
	    DirectSum(I(1), 
		      Tensor(I(P[1]/2-1), F(2)*J(2)),
		      I(1)) *
	    DirectSum(C[1], C[2]) ^ L(P[1], 2) *
	    DirectSum(List([0..P[1]/2-1], i -> J(2)*Rot(2-(2*P[1]-2*i-1)/(4*P[1])))) * 
	    LIJ(P[1])

	  # LIJ(P[1]) * 
	  # DirectSum(C[1], C[2] ^ J(P[1]/2)) *
	  # DirectSum(List([0..P[1]/2-1], i -> J(2)*Rot(2-(2*P[1]-2*i-1)/(4*P[1])))) 
	  #    ^ LIJ(P[1])
    ),

    #F DCT4_DST4andDST2: 1988
    #F
    #F   DCT4_n = diag * sums * perm * (DST4_n/2 dirsum DST2_n/2) * sign *
    #F            (DFT_2 tensor I_n/2) * diag * perm
    #F
    #F   Rao/Yip:
    #F     The Decimation-in-Frequency Algorithms for a family of 
    #F     Discrete Sine and Cosine Algorithms.
    #F     Circuits, Systems, and Signal Processing, 1988, pp. 3--19
    #F
    DCT4_DST4andDST2 := rec (
	info             := "DCT4_n --> DST4_n/2, DST2_n/2",
	switch           := false,
	isApplicable     := P -> IsInt(P[1]) and P[1] > 2 and P[1] mod 2 = 0,

	allChildren := P -> [[ DST4(P[1]/2), DST2(P[1]/2) ]],

	rule := (P, C) -> 
	    Diag(List([0..P[1] - 1], i -> (-1)^i)) *
	    sums1(P[1]).transpose() *
	    L(P[1], P[1]/2) *
	    DirectSum(C[1], C[2] * (-1)) *
	    Tensor(F(2), I(P[1]/2)) *
	    Diag(Concat(List([1..P[1]/2], i -> 1/(2*SinPi((2*i - 1)/(4*P[1])))),
		        List([1..P[1]/2], i -> 1/(2*CosPi((2*i - 1)/(4*P[1])))))) *
	    IJ(P[1], P[1]/2)
    ),

    #F DCT4_Iterative: 1985, DCT4_n = iterative
    #F
    #F   Chen/Smith/Fralick: 
    #F     A Fast Computational Algorithm for the Discrete Cosine Transform,
    #F     IEEE Trans. on Comm., 1977, pp. 1004--1009
    #F   corrected in:
    #F   Wang: 
    #F     Reconsideration of --above--, IEEE Trans. on Comm., 1983, pp. 121--123
    #F   Wang: 
    #F      Fast Algorithms for the Discrete W Transform and the
    #F      Discrete Fourier Transform.
    #F      IEEE Trans. on ASSP, 1984, pp. 803--814
    #F
    DCT4_Iterative := rec (
	info             := "DCT4_n iterative",
	isApplicable     := P -> P[1] > 2 and Is2Power(P[1]),
	rule := (P, C) -> 
	    LIJ(P[1]) * L(P[1], 2) *
	    DirectSum(List([1..P[1]/2], i -> Rot(1/2-(4*i-3)/(4*P[1])) * J(2))) *

	    Compose(List([1..Log2Int(P[1])-1], j -> let(jj := 2^j,
		    Tensor(I(jj/2), F(2), I(P[1]/jj)) * 
		    Tensor(I(jj/2), 
			DirectSum(I(P[1]/jj), List([1..P[1]/jj/2], 
			     i -> Rot(1/2-(4*i-3)/(2*P[1]/jj)) * J(2))))))) *
	    perm6(P[1])
    )
));

