
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F DHT <size>)
#F Definition: (n x n)-matrix [ cas(2*pi*i*k*l/n) | k,l = 0...n-1 ] or
#F                            [ cos(2*pi*i*k*l/n) + sin(2*pi*i*k*l/n) | k,l = 0...n-1 ]
#F Note:     DHT_n = Re(DFT_n) + Im(DFT_n)
#F Example:  DHT(8)
#F
Class(DHT, NonTerminal, rec(
    abbrevs := [ n -> Checked(IsPosInt(n), [n]) ],
    dims := self >> [self.params[1], self.params[1]],
    terminate := self >> Mat(Sqrt(self.params[1]) * Global.DHT(self.params[1])),
    transpose := self >> Copy(self),
    isReal := self >> true,
    SmallRandom := ( ) -> Random([2, 4, 6, 8, 12, 16, 24, 32]),
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(2.5 * n * d_log(n) / d_log(2)))
));

#F -----------------------------------------------------------------------------
#F XH(size) : represents a matrix used in DHT radix-2 rule (temporary solution)
#F -----------------------------------------------------------------------------
Class(XH, Sym, rec(
    notSupported := true,
    abbrevs := [ n -> Checked(IsInt(n), n >= 4, n mod 4 = 0, n) ],  

    def := N -> When(N = 4, I(4),
	DirectSum(
	    I(N/2 + 1),
	    DirectSum(
		I(1), 
		List([1..N/4-1], i -> Rot(1/2-2*i/N) * J(2))) ^ 
	              perm4(N/2-1))),

    transpose := self >> self,

    isReal := True
));

RulesFor(DHT, rec(
    #F DHT_Base: DHT_2 = F_2
    #F
    DHT_Base := rec (
	info             := "DHT_2 -> F_2",
	forTransposition := false,
	isApplicable     := P -> P[1] = 2,
	allChildren := P -> [[ ]],
	rule := (P, C) -> F(2)
    ),

    DHT_toPRDFT := rec(
        isApplicable := P -> P[1] > 2,
        allChildren := P -> [[ PRDFT(P[1]) ]],
        rule := (P, C) -> let(n := P[1], cut := Mat([[1,0]]),      
            OS(n,-1) * # or set the rotation to -1
            LIJ(n).transpose() * 
            Cond(IsEvenInt(n), DirectSum(cut, Tensor(I(n/2-1), F(2)), cut) * C[1],
                 IsOddInt(n),  DirectSum(cut, Tensor(I((n-1)/2), F(2)))    * C[1]))
    ),

    #F DHT_DCT1andDST1:
    #F
    #F   DHT_n = blocks * (DCT1_(n/2+1) dirsum DST1_(n/2-1)) * blocks
    #F
    #F   Britanak/Rao:
    #F     The fast generalized discrete Fourier transforms: A unified
    #F     approach to the discrete sinusoidal transforms computation.
    #F     Signal Processing 79, 1999, pp. 135--150
    #F
    DHT_DCT1andDST1 := rec (
	info             := "DHT_n -> DCT1_(n/2+1), DST1_(n/2-1)",
	forTransposition := false,
	isApplicable     := P -> P[1] mod 2 = 0 and P[1] > 2, 
	allChildren := P -> [[ DCT1(P[1]/2+1), DST1(P[1]/2-1) ]],
	rule := (P, C) -> let(n:=P[1], 
	    DirectSum(I(1), blocks1(n - 1)) *
	    DirectSum(C[1], C[2] ^ J(n/2 - 1)) *
	    DirectSum(I(1), blocks1(n - 1)))
    ),

    #F DHT_2:
    #F Radix-2 Cooley-Tukey
    #F
    DHT_Radix2 := rec (
	info             := "DHT_n -> DHT_n/2",
	forTransposition := false,
	isApplicable     := P -> P[1] mod 2 = 0 and IsPrimePowerInt(P[1]) and P[1] > 2,
	allChildren := P -> [[ DHT(P[1]/2) ]],
	rule := (P, C) -> let(n:=P[1],
	    Tensor(F(2), I(n/2)) *
	    XH(n) *
	    Tensor(I(2), C[1]) *
	    L(n, 2))
    )
));
