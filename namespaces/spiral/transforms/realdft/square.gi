
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(ISRDFT, ISRDFT2, ISRDFT3, ISRDFT4);

Class(SRDFT, PRDFT, rec(
    dims := self >> [self.params[1], self.params[1]],
    inverse := self >> ISRDFT(self.params[1], -self.params[2]).setTransposed(self.transposed),
    inverseViaDiag := self >> self.transpose() * ISRDFT.diag(self.params[1]),
    terminate := self >> let(N := self.params[1], k := self.params[2],
	m  := MatSPL(PRDFT(N,k)),
	sm := Mat(Cond(IsOddInt(N), 
	         Concatenation([m[1]], m{[3..Length(m)]}),
	         Concatenation([m[1]], m{[3..Length(m)-2]}, [m[Length(m)-1]]))),
	Cond(self.transposed, sm.transpose(), sm)),
));
SRDFT1 := SRDFT;
Class(SRDFT2, PRDFT2, rec(
    dims := self >> [self.params[1], self.params[1]],
    inverse := self >> ISRDFT3(self.params[1], -self.params[2]).setTransposed(self.transposed),
    terminate := self >> let(N := self.params[1], k := self.params[2],
	m  := MatSPL(PRDFT2(N,k)),
	sm := Mat(Cond(IsOddInt(N), 
	         Concatenation([m[1]], m{[3..Length(m)]}),
	         Concatenation([m[1]], m{[3..Length(m)-2]}, [m[Length(m)]]))),
	Cond(self.transposed, sm.transpose(), sm)),
));
Class(SRDFT3, PRDFT3, rec(
    dims := self >> [self.params[1], self.params[1]],
    inverse := self >> ISRDFT2(self.params[1], -self.params[2]).setTransposed(self.transposed),
    terminate := self >> let(N := self.params[1], k := self.params[2],
	m  := MatSPL(PRDFT3(N,k)),
	sm := Mat(Cond(IsOddInt(N), 
	         m{[1..Length(m)-1]},
	         m)),
	Cond(self.transposed, sm.transpose(), sm)),
));
Class(SRDFT4, PRDFT4, rec(
    dims := self >> [self.params[1], self.params[1]],
    inverse := self >> ISRDFT4(self.params[1], -self.params[2]).setTransposed(self.transposed),
    terminate := self >> let(N := self.params[1], k := self.params[2],
	m  := MatSPL(PRDFT4(N,k)),
	sm := Mat(Cond(IsOddInt(N), 
	         Concatenation(m{[1..Length(m)-2]}, [m[Length(m)]]),
	         m)),
	Cond(self.transposed, sm.transpose(), sm)),
));

Class(ISRDFT, IPRDFT, rec(
    dims := self >> [self.params[1], self.params[1]],
    diag := n -> When(n=2, I(2), 
        Diag(diagDirsum(fConst(TReal, 1, 1), fConst(TReal, 2*Int((n-1)/2), 2), 
                When(IsEvenInt(n), fConst(TReal, 1, 1), [])))), 

    terminate := self >> let(N := self.params[1], k := self.params[2],
	m  := MatSPL(IPRDFT(N,k)),
	sm := Mat(Cond(IsOddInt(N), 
	         List(m, r->Concatenation([r[1]], r{[3..Length(r)]})),
	         List(m, r->Concatenation([r[1]], r{[3..Length(r)-2]}, [r[Length(r)-1]])))),
	Cond(self.transposed, sm.transpose(), sm)),
));
Class(ISRDFT2, IPRDFT2, rec(
    dims := self >> [self.params[1], self.params[1]],
    terminate := self >> let(N := self.params[1], k := self.params[2],
	m  := MatSPL(IPRDFT2(N,k)),
	sm := Mat(Cond(IsOddInt(N), 
	         List(m, r->r{[1..Length(r)-1]}),
	         m)),
	Cond(self.transposed, sm.transpose(), sm)),
));
Class(ISRDFT3, IPRDFT3, rec(
    dims := self >> [self.params[1], self.params[1]],
    terminate := self >> let(N := self.params[1], k := self.params[2],
	m  := MatSPL(IPRDFT3(N,k)),
	sm := Mat(Cond(IsOddInt(N), 
	         List(m, r->Concatenation([r[1]], r{[3..Length(r)]})),
	         List(m, r->Concatenation([r[1]], r{[3..Length(r)-2]}, [r[Length(r)]])))),
	Cond(self.transposed, sm.transpose(), sm)),
));
Class(ISRDFT4, IPRDFT4, rec(
    dims := self >> [self.params[1], self.params[1]],
    terminate := self >> let(N := self.params[1], k := self.params[2],
	m  := MatSPL(IPRDFT4(N,k)),
	sm := Mat(Cond(IsOddInt(N), 
	         List(m, r->Concatenation(r{[1..Length(r)-2]}, [r[Length(r)]])),
	         m)),
	Cond(self.transposed, sm.transpose(), sm)),
));

RulesFor(SRDFT1, rec(
    SRDFT1_toPRDFT1 := rec(
	isApplicable := P -> true,
	allChildren := P -> [[ PRDFT1(P[1], P[2]) ]],
	rule := (P, C) -> Pack_CCS(P[1]) * C[1]
    )
));

RulesFor(SRDFT3, rec(
    SRDFT3_toPRDFT3 := rec(
	isApplicable := P -> true,
	allChildren := P -> [[ PRDFT3(P[1], P[2]) ]],
	rule := (P, C) -> Pack_CCS3(P[1]) * C[1]
    )
));
