
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F IRDFT(<size>, [<normalized?>]) - Inverse RDFT, non-terminal
#F This is an obsolete inverse real DFT, plese use IPRDFT
#F
Class(IRDFT, NonTerminal, rec(
    abbrevs := [ 
	N -> Checked(IsInt(N), N > 1, [N, false]),
	(N, norm) -> Checked(IsInt(N), N > 1, IsBool(norm), [N, norm])
    ],
    ndiag := meth(self)
        local Ls, n;
	n := self.params[1];
	if self.params[2] then 
	    Ls:=List([1..n], i->2/n);
	    Ls[1]:=1/n; 
	    if (n mod 2 = 0) then Ls[n/2+1]:=1/n; fi;
	    return Diag(Ls);
	else return I(self.params[1]);
	fi;
    end,
    transpose := self >> let(exp := 1, Compose(self.ndiag(), RDFT(self.params[1]))),
    terminate := self >> TransposedSPL(TerminateSPL(self.transpose())),
    isReal := True,
    dims := self >> [self.params[1], self.params[1]],
    SmallRandom := () -> Random([2, 3, 4, 6, 8, 12, 16, 24, 32])
));

RulesFor(IRDFT, rec(
    #F IRDFT_Trigonometric: 1984 
    #F
    #F   invRDFT_n = blocks * (DCT1_(n/2+1) dirsum DST1_(n/2-1)) 
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
    IRDFT_Trigonometric := rec(
	info             := "IRDFT_n -> DCT1_(n/2+1), DST1_(n/2-1)",
	forTransposition := false,
	isApplicable     := P -> P[1] mod 2 = 0 and P[1] > 2, 
	allChildren      := P -> [[ DCT1(P[1]/2 + 1), DST1(P[1]/2 - 1) ]],
	rule := (P, C) -> let(
	    n := P[1],
	    exp := 1, # should be an extra IRDFT parameter
	    normalized := P[2],

	    DirectSum(I(1), blocks1(n - 1)) *
	    DirectSum(C[1], exp * C[2] ^ J(n/2 - 1)) * 
	    When(normalized, 
		Diag(List([0..n-1], c -> When(c mod (n/2) = 0, 1/n, 2/n))), 
		I(n)))
    )
));
