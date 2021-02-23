
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RulesFor(DCT3, rec(
    #F DCT3_Base2: DCT2_2 = F_2 * diag(1, 1/sqrt(2))
    #F
    DCT3_Base2 := rec (
	info             := "DCT3_2 -> F_2",
	isApplicable     := P -> P[1] = 2,
	rule := (P, C) -> F(2) * Diag(1, Sqrt(1/2))
    ),

    #F DCT3_Base3: DCT3_3 = (I_1 dirsum F_2)^P * (M dirsum c) * Q
    #F well-known, e.g.,
    #F   Pueschel/Moura: Discrete Cosine and Sine Transforms (in preparation)
    DCT3_Base3 := rec (
	info             := "DCT3_3 = (I_1 dirsum F_2)^P * (M dirsum c) * Q",
	isApplicable     := P -> P[1]=3, 
	rule := (P, C) -> 
            (DirectSum(I(1), F(2)) ^ Perm((1,2), 3)) * 
	    DirectSum(Mat([[1, 1/2], [1, -1]]),
		      Diag(CosPi(1/6))) *
	    Perm((2,3), 3)
    ),

    #F DCT3_Base5: (possible base case for size 3)
    #F
    #F   DCT3_5 = P * (I_1 dirsum (SkewDCT3_2's * (pDCT8_2 tensor I_2) * B 
    #F
    #F   Pueschel/Moura: Discrete Cosine and Sine Transforms, in preparation
    #F
    DCT3_Base5 := rec (
	info             := "DCT3_5 -> SkewDCT3_2's, pDCT8_2",
	isApplicable     := P -> P[1]=5,
	allChildren := P -> [[ SkewDTT(DCT3(2), 1/5), 
		               SkewDTT(DCT3(2), 3/5), 
			       PolyDTT(DCT8(2)) ]],
	rule := (P, C) -> 
	    Perm((1,2,4,5,3), 5) *
	    DirectSum(I(1),  
		      DirectSum(C[1], C[2]) * Tensor(C[3], I(2))) *
	    Mat([ [  1,   0,   -1,   0,    1 ],
		  [  1,   0,  1/2,   0,    0 ],
		  [  0,   1,    0,   0,    0 ],
		  [  0,   0,  1/2,   0,  1/2 ],
		  [  0,   0,    0,   1,    0 ] ])
    ),

    #F DCT3_Orth9: DCT3_9 = hand-derived radix-3 Cooley-Tukey for RDFT
    #F
    DCT3_Orth9 := rec (
	info             := "DCT3_9 -> TRDFT_3's",
	isApplicable     := P -> P[1] = 9,
	allChildren := P -> [ [ RDFT(3).transpose() ] ],
	rule := (P, C) ->
	   Perm( (1,5)(2,7,8,4,9,6), 9 ) *
	   Tensor(C[1],	I(3)) *
	   DirectSum(
	       I(3),
	       (DirectSum(I(2), Rot(2/9), Rot(4/9))) ^ Perm((2,4,5,3), 6)) *
	   Tensor(I(3), C[1]) *
	   DirectSum(
	       I(3),
	       Mat([ [ 1,  0,  0, 0, 0, 0 ], 
		     [ 0,  1,  0, 0, 0, 1 ], 
		     [ 0,  0,  1, 0, 1, 0 ],
		     [ 0,  0,  0, 1, 0, 0 ], 
		     [ 0,  0, -1, 0, 1, 0 ], 
		     [ 0, -1,  0, 0, 0, 1 ] ])) *
	   Diag(1, -1, 1, -1, 1, -1, 1, -1, 1) *
	   Perm( (2,7,8)(3,4)(5,9), 9 )
    ),

    #F DCT3_toSkewDCT3:  DCT3_n = DCT3_n(1/2)
    #F
    DCT3_toSkewDCT3 := rec (
	info             := "DCT3_n -> SkewDCT3_n(1/2)",
	forTransposition := false,
	isApplicable     := P -> IsPrimePowerInt(P[1]) and P[1] mod 2 = 0 and P[1] > 2,
	allChildren := P -> [ [ SkewDTT(DCT3(P[1]), 1/2) ] ],
	rule := (P, C) -> C[1]
    )
));
