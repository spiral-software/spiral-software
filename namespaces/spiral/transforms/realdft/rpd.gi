
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# input: imaginary, output: co1 (?)
_j := N -> Tensor(I(approx.CeilingRat(N/2)),J(2)*Diag(-1,1));

RealRR_Out := (N, root) -> let(
    rrind := RR(N, 1, root).tolist(),
    outind := List(rrind{[1..(N+1)/2]}, x->When(x.v >= (N+1)/2, N-x.v, x.v)),
    #same as Refl((N+1)/2, N, (N+1)/2, RR(N,1,root)).tolist(),

    # Reflective RR
    Scat(fTensor(FList((N+1)/2, outind), fId(2))) *
    # Conjugate complex elements that will be reflected
    Diag(ConcatList([0..(N-1)/2], x->When(rrind[x+1].v >= (N+1)/2, [1,-1], [1,1])))
);

RulesFor(PRDFT, rec(

   # PRDFT_PD : Projection of complex DFT partial diagonalization rule to real DFT 
   # See also : PRDFT_Rader
   #
   PRDFT_PD := rec(
	forTransposition := true,
	maxSize          := 13,
	isApplicable     := (self, P) >> P[1] > 2 and P[1] <= self.maxSize and IsPrime(P[1]),
	
	rule := (self,P,C) >> let(N:=P[1], n:=N-1, k:=P[2], root:=PrimitiveRootMod(N),
	    BB(RealRR_Out(N, root) *
	    DirectSum(Mat([[1],[0]]), L(n, n/2)) * 
	    Mat(MatSPL(DFT_PD.core(N, k, root, true) * DFT_PD.A(N))) *
	    DirectSum(I(1), Tensor(F(2), I(n/2))) * 
	    DirectSum(I(1), OS(n, -1)) *
	    Gath(RR(N, 1, root)))
	)
   ),

   # PRDFT_Rader : Projection of complex DFT Rader rule to real DFT 
   #
   #   Note: PRDFT of types 2--4 of odd size (which includes all primes > 2)
   #         can be converted without arithmetic cost to PRDFT1,
   #         which means that this rule enables implementation of a prime 
   #         size PRDFT of any type.
   PRDFT_Rader := rec(
	isApplicable := (self, P) >> P[1] > 3 and IsPrime(P[1]),
	
	diag := (N, k, root) -> let(fulldiag := DFT_Rader.raderDiag(N,k,root),
	    last := Int((N-1)/2),
	    Concatenation(2*fulldiag{[1..last-1]}, [fulldiag[last]])),

	# 3rd col with 0's is for padding used by PRDFT
	raderMid := (self, N, k, root) >> let(n:=N-1,
	    DirectSum(Mat([[1, 1, 0], [1, -1/n, 0], [0,0,0]]),
		      RCDiag(RCData(FData(self.diag(N, k, root)))))),

	allChildren := P -> let(n:=P[1]-1,
	    [[ PRDFT(n/2).transpose(), PRDFT3(n/2).transpose(), PRDFT(n, -1) ]]),

	rule := (self,P,C) >> let(N:=P[1], n:=N-1, k:=P[2], root:=PrimitiveRootMod(N),
	    RealRR_Out(N, root) *
	    DirectSum(Mat([[1],[0]]), ##
		      L(n, n/2) *
		      DirectSum(C[1], C[2] * _j(n/2)) *
		      RC(L_or_OS(n/2+1, 2))) *
	    self.raderMid(N, k, root) *
	    DirectSum(I(1), C[3]) *
	    Gath(RR(N, 1, root))
	)
   )
));

RulesFor(IPRDFT, rec(

   # IPRDFT_PD : Projection of complex DFT partial diagonalization rule to inverse real DFT 
   # See also : IPRDFT_Rader
   #
   IPRDFT_PD := rec(
	forTransposition := true,
	maxSize          := 13,
	isApplicable     := (self, P) >> P[1] > 2 and P[1] <= self.maxSize and IsPrime(P[1]),
	rule := (self,P,C) >> let(N:=P[1], n:=N-1, k:=P[2], root:=PrimitiveRootMod(N),
	    BB(Scat(RR(N, 1, root)) *
	    DirectSum(I(1), OS(n, -1)) *
	    DirectSum(I(1), Tensor(F(2), I(n/2))) * 
	    Mat(MatSPL(TransposedSPL(DFT_PD.core(N, -k, root, true) * DFT_PD.A(N)) * DirectSum(I(1), 2*I(n)))) *
	    DirectSum(Mat([[1,0]]), L(n, 2)) * 
	    RealRR_Out(N, root).transpose())
	)
   ),

   # IPRDFT_Rader : Projection of complex DFT Rader rule to inverse real DFT 
   #
   #   Note: IPRDFT of types 2--4 of odd size (which includes all primes > 2)
   #         can be converted without arithmetic cost to IPRDFT1,
   #         which means that this rule enables implementation of a prime 
   #         size IPRDFT of any type.
   IPRDFT_Rader := CopyFields(PRDFT_Rader, rec(
	# 3rd col with 0's is for padding, we also scale everything but 1st elt by 2,
	# as IPRDFT requires. 
	raderMid := (self, N, k, root) >> let(n:=N-1,
	    DirectSum(Mat([[1, 2, 0], [1, -2/n, 0], [0,0,0]]),
		      RCDiag(RCData(FConj(FData(2*self.diag(N, -k, root))))))),

	allChildren := P -> let(n:=P[1]-1,
	    [[ PRDFT(n/2), PRDFT3(n/2), PRDFT(n,-1).transpose() ]]),

	rule := (self,P,C) >> let(N:=P[1], n:=N-1, k:=P[2], root:=PrimitiveRootMod(N),
	    Scat(RR(N, 1, root)) *
	    DirectSum(I(1), C[3]) *
	    self.raderMid(N, k, root) *
	    DirectSum(Mat([[1,0]]), ##
		      RC(L_or_OS(n/2+1, 2).transpose()) *
		      DirectSum(C[1], _j(n/2).transpose() * C[2]) *
		      L(n, 2)) *
	    RealRR_Out(N, root).transpose()
	)
   ))
));