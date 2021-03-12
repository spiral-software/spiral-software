
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F --------------------------------------------------------------------------------
RulesFor(DST1, rec(
  #F DST1_Base1: DST1_1 = I_1
  #F
  DST1_Base1 := rec (
      info             := "DST_1 -> I_1",
      forTransposition := false,
      isApplicable     := P -> P[1] = 1, 
      rule := (P, C) -> I(1)
  ),

  #F DST1_Base2: DST1_2 = cos(pi/6) * F_2
  #F
  DST1_Base2 := rec (
      info             := "DST_2 -> c * F_2",
      forTransposition := false,
      isApplicable     := P -> P[1] = 2, 
      rule := (P, C) -> CosPi(1/6) * F(2)
  ),

  #F DST1_DST3and1: 1984
  #F
  #F   DST1_(n-1) = perm * (DST3_n/2 dirsum DST1_(n/2-1) ^ perm) * (1 tensor DFT_2)^perm
  #F
  #F   Wang: 
  #F     Fast Algorithms for the Discrete W Transform and the
  #F     Discrete Fourier Transform.
  #F     IEEE Trans. on ASSP, 1984, pp. 803--814
  #F
  DST1_DST3and1 := rec (
      info          := "DST1_(n-1) --> DST3_n/2, DST1_(n/2-1)",
      isApplicable  := P -> (P[1] + 1) mod 2 = 0 and P[1] >= 2,
      allChildren   := P -> [[ DST3((P[1] + 1)/2), DST1((P[1] - 1)/2) ]],
      rule := (P, C) -> # P is odd
	 DirectSum(C[1], C[2]) ^ OS(P[1], 2) * 
	 DirectSum(I(1), Tensor(I((P[1] - 1)/2), F(2))) ^ Z(P[1],P[1]-1) *
	 LIJ(P[1])
         #LIJ(P) *
	 #DirectSum(C[1], C[2] ^ J((P - 1)/2)) *
	 #DirectSum(I(1), Tensor(I((P - 1)/2), F(2))) ^ (Z(P,P-1) * LIJ(P))
  ),

  DST1_toDCT3 := rec( # this rule is probably numerically unstable
      isApplicable  := P -> P[1] >= 3, 
      allChildren   := P -> [[ DCT3(P[1]+1) ]], 
      rule := (P, C) -> let(n:=P[1], N := n+1,
          RowTensor(n, 1, Mat([[1,-1]])) * C[1] * 
          VStack(RowVec(fConst(TReal, n, 0)), 
                 Diag(FList(TReal, List([1..n], i -> CosPi(i/(2*N)) / SinPi(i/N))))))
  )
));

#F --------------------------------------------------------------------------------
RulesFor(DST2, rec(
    #F DST2_Base2: DST2_2 = diag(1/Sqrt(2), 1) * F_2
    #F
    DST2_Base2 := rec(
	info := "DST_2 -> F_2",
	isApplicable := P -> P[1] = 2, 
	rule := (P, C) -> Diag([Sqrt(1/2), 1]) * F(2)
    ),

    #F DST2_toDCT2: DST2_n =  perm * DCT2_n * diag
    #F
    #F   Wang: 
    #F     A Fast Algorithm for the Discrete Sine Transform by the
    #F     Fast Cosine Transform, IEEE Trans. on ASSP, 1982, pp. 814--815
    #F
    DST2_toDCT2 := rec (
	info             := "DST2_n -> DCT2_n",
	isApplicable     := P -> P[1] > 2, 
	allChildren := P -> [[ DCT2(P[1]) ]], 
	rule := (P, C) -> 
	    J(P[1]) * C[1] * Diag(List([0..P[1] - 1], i -> (-1)^i))
    ),

    #F DST2_DST2and4: 1982
    #F
    #F   DST2'_n = perm * (DST4_n/2 dirsum DST2'_n/2^perm) * (1 tensor DFT_2)^perm
    #F
    #F   Wang: 
    #F      Fast Algorithms for the Discrete W Transform and the
    #F      Discrete Fourier Transform.
    #F      IEEE Trans. on ASSP, 1984, pp. 803--814
    #F
    DST2_DST2and4 := rec(
	info             := "DST2_n --> DST4_n/2, DST2_n/2",
	isApplicable     := P -> P[1] > 2 and P[1] mod 2 = 0, 
	allChildren      := P -> [[ DST4(P[1]/2), DST2(P[1]/2) ]],
	rule := (P, C) -> 
	    DirectSum(C[1], C[2]) ^ L(P[1],2) *
	    Tensor(I(Int((P[1] / 2))), F(2)) *
 	    LIJ(P[1])
    )
));

#F --------------------------------------------------------------------------------
RulesFor(DST4, rec(
    #F DST4_Base2: DST4_2 = (1,2) * R_15/8
    #F
    DST4_Base := rec (
	info             := "DST_4 -> R_15/8",
	forTransposition := false,
	isApplicable     := P -> P[1] = 2, 
	rule := (P, C) -> J(2) * Rot(15/8)
    ),

    #F DST4_1: 1984
    #F
    #F   DST4_n = perm * DCT4_n * diag
    #F
    #F   Wang: 
    #F     Fast Algorithms for the Discrete W Transform and the
    #F     Discrete Fourier Transform.
    #F     IEEE Trans. on ASSP, 1984, pp. 803--814
    #F   Chan: 
    #F     Direct Methods for computing discrete sinusoidal transforms
    #F     IEE Proceedings, Vol. 137, 1990, pp. 433--442
    #F
    DST4_toDCT4 := rec (
	info             := "DST4_n --> DCT4_n",
	isApplicable     := P -> P[1] > 2, 
	allChildren := P -> [[ DCT4(P[1]) ]],
	rule := (P, C) -> 
  	    J(P[1]) * C[1] * Diag(List([0..P[1]-1], i -> (-1)^i))
    )
));
