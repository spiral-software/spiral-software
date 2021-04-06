
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RulesFor(DFT, rec(
    DFT_CT_Inplace := rec(
	info          := "DFT(mn,k) -> DFT(m, k%m), DFT(n, k%n)",
	maxCodeletSize := 16,

	isApplicable := (self, P) >> P[1] > 2 and not IsPrime(P[1]),

	allChildren  := (self, P) >> Map2( 
	    Filtered(DivisorPairs(P[1]), d -> d[1] <= self.maxCodeletSize), 
	    (m,n) -> [ 
		DFT(m, P[2] mod m), 
		DFT(n, P[2] mod n) ]),
       
	rule := (self,P,C) >> let(mn := P[1], m := Rows(C[1]), n := Rows(C[2]), 
            di := fPrecompute(Tw1(mn, n, P[2])),

	    When(mn > self.maxCodeletSize, # why BB?
		Inplace(Tensor(C[1], I(n)) * Diag(di)), 
		        Tensor(C[1], I(n)) * Diag(di)) *
	    Tensor(I(m), C[2]) *
	    L(mn, m))
    ),

    DFT_SplitRadix_Inplace := rec(
	info             := "DFT_n -> DFT_n/2, DFT_n/4, DFT_n/4",
	forTransposition := true,
	maxSize := 64, 

	isApplicable := (self, P) >> let(N := P[1],
	    N >= 8 and N mod 4 = 0 and N <= self.maxSize),

	allChildren := P -> let(N := P[1], w := P[2], 
	    [[ DFT(N/2, w), DFT(N/4, w) ]]),

	rule := (P, C) -> let(N := P[1], w := P[2], 
	    Inplace(Tensor(F(2), I(N/2))) *
	    DirectSum( 
		C[1], 
		Inplace(BB(
			Tensor(Diag(1, E(4)^w) * F(2), I(N/4)) *
			Diag(Concat(List([0..N/4-1], i->E(N)^(w*i))), 
			            List([0..N/4-1], i->E(N)^(3*w*i))))) *
	        Tensor(I(2), C[2])*L(N/2,2)
	    ) *
	    L(N, 2))
    ),

));
