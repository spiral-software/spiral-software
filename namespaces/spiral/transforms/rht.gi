
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F RHT( <log2(size)> ) - Rationalized Haar Transform non-terminal
#F
#F Definition: RHT(n) represents the (2^n x 2^n)-matrix defined recursively as:
#F             RHT(1) := DFT_2
#F             RHT(n) := [[ RHT(n-1) tensor [1, 1] ], 
#F                        [ I_(n-1)  tensor [1 -1] ]]
#F Note:       I_(n-1) denotes the (2^(n-1) x 2^(n-1)) identity matrix
#F             DFT_2 denotes the matrix [[1, 1], [1, -1]]
#F Example:    RHT(6) 
#F             RHT(6).transpose()
Class(RHT, NonTerminal, rec (
    abbrevs := [ n -> Checked(IsInt(n), n > 0, [n]) ],
    dims      := self >> [ 2^self.params[1], 2^self.params[1] ],
    terminate := self >> Cond(
	self.params[1] = 1,  F(2), 
	self.transposed, Mat(RationalizedHaarTransform(2^self.params[1])).transpose(),
	                 Mat(RationalizedHaarTransform(2^self.params[1]))),
    isReal := True,
    SmallRandom := () -> [ Random([1..6]) ]
));

RulesFor(RHT, rec(
    #F RHT_Base: (base case) RHT_(2^1) = F_2
    #F
    RHT_Base := rec (
	info             := "RHT_(2^1) -> F_2",
	forTransposition := false,
	isApplicable     := P -> P[1] = 1, 
	rule := (P, C) -> F(2) 
    ),

    #F RHT_CooleyTukey: RHT_2^(n+1) = (RHT_2^(n) dirsum I_2^n) * (DFT_2 tensor I_2^n) * L
    #F
    RHT_CooleyTukey := rec (
	info             := "RHT_2^(n+1) -> (RHT_2^n dirsum I_n) * (F_2 tensor I_n)",
	isApplicable     := P -> P[1] > 1,
	allChildren      := P -> [[ RHT(P[1] - 1) ]], 
	rule := (P, C) -> 
	    Inplace(DirectSum(C[1], I(2 ^ (P[1]-1)))) *
	    Tensor(F(2), I(2 ^ (P[1]-1))) *
	    L(2^P[1], 2)
    )
));
