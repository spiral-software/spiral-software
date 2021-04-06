
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F RDFT(<size>) -  Discrete Fourier Transform of a real sequence, non-terminal
#F This is an obsolete real DFT, please use PRDFT
#F
#F Definition: (n x n)-matrix
#F              [ cos(2*pi*k*l/n) | k = 0...Int(n/2), l = 0...n-1 ]
#F              [-sin(2*pi*k*l/n) | k = Int(n/2)+1...n, l = 0...n-1 ]
#F              k and l are row and column indices, respectively.
#F Note:      RDFT is obtained using symmetry of  DFT on a real-valued input
#F Example:   RDFT(8)
Class(RDFT, NonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N > 1, [N]) ],
    terminate := self >> When(self.params[1]=2, F(2), 
	Mat(ApplyFunc(Global.RDFT, self.params))),
    isReal := True,
    dims := self >> [self.params[1], self.params[1]],
    SmallRandom := () -> Random([2, 3, 4, 6, 8, 12, 16, 24, 32])
));

RulesFor(RDFT, rec(
    #F RuleRDFT_Base: RDFT_2 = F_2
    #F
    RDFT_Base := rec (
	info             := "RDFT_2 -> F_2",
	isApplicable     := L -> L[1] = 2, 
	allChildren      := L -> [ [ ] ], 
	rule            := (L, C) -> F(2)
    ), 

    #F RDFT_Trigonometric: 1984 
    #F
    #F   RDFT_n = (DCT1_(n/2+1) dirsum DST1_(n/2-1)) * blocks
    #F
    #F   Wang: 
    #F     Fast Algorithms for the Discrete W Transform and the
    #F     Discrete Fourier Transform.
    #F     IEEE Trans. on ASSP, 1984, pp. 803--814
    #F   Britanak/Rao:
    #F     The fast generalized discrete Fourier transforms: A unified
    #F     approach to the discrete sinusoidal transforms computation.
    #F     Signal Processing 79, 1999, pp. 135--150
    #F
    RDFT_Trigonometric := rec (
	info             := "RDFT_n -> DCT1_(n/2+1), DST1_(n/2-1)",
        # 3 can occur only once
	isApplicable     := L -> L[1] >= 4 and 
	    (Is2Power(L[1]) or (L[1] mod 3 = 0 and Is2Power(L[1]/3))),
	       
        allChildren := L -> [[ DCT1(L[1]/2 + 1), DST1(L[1]/2 - 1) ]], 

	rule := (L, C) -> let(N := L[1], exp := 1, 
	   DirectSum(C[1], exp * C[2] ^ J(N/2 - 1)) *
	   DirectSum(I(1), blocks1(N - 1)))
    ), 


    #F RDFT_CT_Radix2  : Cooley-Tukey Radix-2
    #F
    #F   RDFT_n = M * (F_2 tensor)' * (I_n/2 dirsum rots^(L^n/2_n/4)) * 
    #F            (I_2 tensor RDFT_n/2) * L^n_2
    #F
    #F   derived by hand; the prime denotes that one F_2 is missing in the
    #F   tensor product.
    #F
    RDFT_CT_Radix2 := rec (
	info             := "RDFT_n -> RDFT_n/2",
	isApplicable     := L -> let(N := L[1], exp := 1,
	    N mod 4 = 0 and exp = 1 and
	    (IsPrimePowerInt(N) or       # 3 or 5 can occur only once
		(N mod 3 = 0 and IsPrimePowerInt(N/3)) or
		(N mod 5 = 0 and IsPrimePowerInt(N/5)))),
      
	allChildren := L -> [[ RDFT(L[1]/2) ]], 

	rule := (P, C) -> let(N := P[1], i := Ind(N/4-1),
	    When(N=4, 
		Conjugate(DirectSum(F(2),I(2)), L(4, 2)) *
		Tensor(I(2), C[1]) * L(4, 2), 

		monomial1(N) *
		DirectSum(Tensor(I(N/4), F(2)), I(2), Tensor(I(N/4 - 1), F(2))) ^ L(N, N/2) *
		DirectSum(
		    I(N/2),
		    DirectSum(I(1), IterDirectSum(i, Rot(fdiv(i+1, N/2))), I(1)*(-1)^(N/2)) 
		       ^ M(N/2,N/4)) *
		Tensor(I(2), C[1]) * L(N, 2)))
    ),

    #F RDFT_toDCT2: 1989 
    #F
    #F   RDFT_n = M * DCT2_n * P,   n odd
    #F
    #F   Chan/Ho: Efficient Index-Mapping for Computing the Discrete Cosine Transform
    #F     Electronics Letters, 1989
    #F   see also:
    #F   Heideman: Computation of an Odd-Length DCT from a Real-Valued DFT of
    #F     the same Length, IEEE Trans. Sig. Proc. 40(1), 1992
    #F
    RDFT_toDCT2 := rec(
	info             := "RDFT_n -> DCT2_n",
	switch           := false,
	isApplicable     := L -> 
            # we block odd sizes larger than 3 for now
	    L[1] mod 2 = 1 and L[1] >= 3, 

        allChildren := L ->  [[ DCT2(L[1]) ]], 

	rule := (L, C) -> let(N := L[1], 
	    Diag(LambdaList([0..N-1], i -> (-1)^i)) *
	    OS(N, 2) *
	    C[1] *
	    perm7(N))
    )
));
