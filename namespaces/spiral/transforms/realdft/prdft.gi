
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(RealDFT_Base, TaggedNonTerminal, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k]) ],
    
    terminate := self >> let(N := self.params[1], k := self.params[2], 
	rr := When(self.transposed, Cols(self), Rows(self)),
	mat := Mat(List([0..rr-1], r -> When(r mod 2 = 0, 
                    List([0..N-1], c -> self.projRe(self.omega(N,k,Int(r/2),c))),
                    List([0..N-1], c -> self.projIm(self.omega(N,k,Int(r/2),c)))))),
        When(self.transposed, mat.transpose(), mat)),

    toAMat := self >> self.terminate().toAMat(), 

    isReal := True,
    SmallRandom := () -> Random([2..16]), 
    LargeRandom := () -> 2 ^ Random([6..15]),

    normalizedArithCost := self >> let(n := self.params[1], 
       floor(2.5 * n * log(n) / log(2.0))),

    hashAs := self >> let(t:=ObjId(self)(self.params[1], 1).withTags(self.getTags()),
        When(self.transposed, t.transpose(), t))

));


Class(PDHT13_Base, RealDFT_Base, rec(
    projRe := w -> Re(w)+Im(w),
    projIm := w -> Re(w)-Im(w)
));

Class(PDHT24_Base, RealDFT_Base, rec(
    projRe := w -> Re(w)+Im(w),
    projIm := w -> -Re(w)+Im(w)
));

Class(PRDFT_Base, RealDFT_Base, rec(
    projRe := Re,
    projIm := Im
));

Class(IPRDFT_Base, TaggedNonTerminal, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
	         (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k]) ],
    terminate := self >> let(n:=self.params[1], k:=self.params[2],
	self.rdft(n, -k).terminate().transpose() * self.diag(n)),
    isReal := True,
    SmallRandom := () -> Random([2..16]), 
    LargeRandom := () -> 2 ^ Random([6..15]),

    diag0 := n -> When(n=2, I(4), 
        Diag(diagDirsum(fConst(TReal, 2, 1), fConst(TReal, 2*Int((n-1)/2), 2), 
                When(IsEvenInt(n), fConst(TReal, 2, 1), [])))),

    diag1 := n -> Diag(diagDirsum(fConst(TReal, 2*Int(n/2), 2), 
                          When(IsOddInt(n), fConst(TReal, 2, 1), []))),

    normalizedArithCost := self >> let(n := self.params[1], 
       floor(2.5 * n * log(n) / log(2.0))),

    hashAs := self >> let(t:=ObjId(self)(self.params[1], 1).withTags(self.getTags()),
        When(self.transposed, t.transpose(), t))
));

Declare(IPRDFT, IPRDFT1, IPRDFT2, IPRDFT3, IPRDFT4);

# PRDFT(<N>, <k>) - Packed Real DFT Nonterminal
#   Is a N+2 x N real matrix. N+2 outputs correspond
#   to Floor(N/2)+1 complex outputs of complex DFT
#   the other half of complex DFT outputs are complex
#
Class(PRDFT, PRDFT_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod n]) ],
    dims := self >> let(n:=self.params[1], d := [2*(Int(n/2)+1), n], When(self.transposed, Reversed(d), d)),
    omega := (N,k,r,c) -> E(N)^(k*r*c),
    inverse := self >> IPRDFT1(self.params[1], -self.params[2]),
    inverseViaDiag := self >> self.transpose() * IPRDFT.diag(self.params[1])
));
PRDFT1 := PRDFT;

Class(PRDFT2, PRDFT_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (2*n)]) ],
    dims := self >> let(n:=self.params[1], d := [2*(Int(n/2)+1), n], When(self.transposed, Reversed(d), d)),
    omega := (N,k,r,c) -> E(2*N)^(k*r*(2*c+1)),
    inverse := self >> IPRDFT3(self.params[1], -self.params[2])
));

Class(PRDFT3, PRDFT_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (2*n)]) ],
    dims := self >> let(n:=self.params[1], d := [2*(Int((n+1)/2)), n], When(self.transposed, Reversed(d), d)),
    omega := (N,k,r,c) -> E(2*N)^(k*(2*r+1)*c),
    inverse := self >> IPRDFT2(self.params[1], -self.params[2])
));

Class(PRDFT4, PRDFT_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (4*n)]) ],
    dims := self >> let(n:=self.params[1], d := [2*(Int((n+1)/2)), n], When(self.transposed, Reversed(d), d)),
    omega := (N,k,r,c) -> E(4*N)^(k*(2*r+1)*(2*c+1)),
    inverse := self >> IPRDFT4(self.params[1], -self.params[2])
));

# Class(SkewRDFT3, PRDFT_Base, rec(
#     abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1, 0, 0]),
#       (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k, 0, 0]), 
#       (n,k,lr,rr) -> Checked(IsPosInt(n), IsInt(k), IsRat(lr), IsRat(rr), 
# 	  Gcd(n,k)=1, [n, k, lr, rr])
#     ],
#     dims := self >> let(n:=self.params[1], [ 2*Int((n+1)/2), n ]),
#     omega := (self,N,k,r,c) >> let(
# 	lnum:=Numerator(self.params[3]), lden:=Denominator(self.params[3]),
# 	rnum:=Numerator(self.params[4]), rden:=Denominator(self.params[4]),
# 	E(2*N*lden*rden) ^ (k*(lden*(2*r+1)+lnum)*(rden*c+rnum)))
# ));
# Class(SkewRDFT4, SkewRDFT3, rec(
#     dims := self >> let(n:=self.params[1], [ 2*Int((n+1)/2), n ]),
#     omega := (self,N,k,r,c) >> let(
# 	lnum:=Numerator(self.params[3]), lden:=Denominator(self.params[3]),
# 	rnum:=Numerator(self.params[4]), rden:=Denominator(self.params[4]),
# 	E(4*N*lden*rden) ^ (k*(lden*(2*r+1)+lnum)*(rden*(2*c+1)+rnum)))
# ));

#
# Hartley
#
Class(PDHT, PDHT13_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (n)]) ],
    dims := self >> let(n:=self.params[1], d := [2*(Int(n/2)+1), n], When(self.transposed, Reversed(d), d)),
    omega := PRDFT.omega
));
PDHT1 := PDHT;

Class(PDHT2, PDHT24_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (2*n)]) ],
    dims := self >> let(n:=self.params[1], d := [2*(Int(n/2)+1), n], When(self.transposed, Reversed(d), d)),
    omega := PRDFT2.omega
));

Class(PDHT3, PDHT13_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (2*n)]) ],
    dims := self >> let(n:=self.params[1], d := [2*(Int((n+1)/2)), n], When(self.transposed, Reversed(d), d)),
    omega := PRDFT3.omega
));

Class(PDHT4, PDHT24_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (4*n)]) ],
    dims := self >> let(n:=self.params[1], d := [2*(Int((n+1)/2)), n], When(self.transposed, Reversed(d), d)),
    omega := PRDFT4.omega
));


# IPRDFT(<N>, <k>) - Inverse Packed Real DFT Nonterminal
#   Is a N x N+2 real matrix. When applied to N+2 outputs
#   of PRDFT yields the original real input.
Class(IPRDFT, IPRDFT_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (n)]) ],
    dims := self >> let(n:=self.params[1], d := [n, 2*(Int(n/2)+1)], When(self.transposed, Reversed(d), d)),
    rdft := PRDFT,
    diag := IPRDFT_Base.diag0
));
IPRDFT1 := IPRDFT;

# IPRDFT2 is inverse if PRDFT3
Class(IPRDFT2, IPRDFT_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (2*n)]) ],
    dims := self >> let(n:=self.params[1], d := [n, 2*(Int((n+1)/2))], When(self.transposed, Reversed(d), d)),
    rdft := PRDFT3,
    diag := IPRDFT_Base.diag1
)); 

# IPRDFT3 is inverse if PRDFT2
Class(IPRDFT3, IPRDFT_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (2*n)]) ],
    dims := self >> let(n:=self.params[1], d := [n, 2*(Int(n/2)+1)], When(self.transposed, Reversed(d), d)),
    rdft := PRDFT2,
    diag := IPRDFT_Base.diag0
));

Class(IPRDFT4, IPRDFT_Base, rec(
    abbrevs := [ (n) -> Checked(IsPosInt(n),  [n, 1]),
                 (n,k) -> Checked(IsPosInt(n), IsInt(k), Gcd(n,k)=1, [n, k mod (4*n)]) ],
    dims := self >> let(n:=self.params[1], d := [n, 2*(Int((n+1)/2))], When(self.transposed, Reversed(d), d)),
    rdft := PRDFT4,
    diag := IPRDFT_Base.diag1
));

