
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(UTwid, DiagFunc, rec(
    def := (N, n, k, a, b, i) -> rec(size:=n),
    lambda := self >> let(N:=self.params[1], n:=self.size, k:=self.params[3],
	a := self.params[4], b := self.params[5], i := self.params[6],
	na := Numerator(a),   nb := Numerator(b),
	da := Denominator(a), db := Denominator(b),
	w := E(N*da*db)^k, j := Ind(n),
	Lambda(j, cond(eq(j,n/2), omega(da*db,k), omega(N*da*db, k*(i*da+na)*(j*db+nb)))))
));
Class(UTwid1, DiagFunc, rec(
    def := (N, n, k, a, b, i) -> rec(size:=n),
    lambda := self >> let(N:=self.params[1], n:=self.size, k:=self.params[3],
	a := self.params[4], b := self.params[5], i := self.params[6],
	na := Numerator(a),   nb := Numerator(b),
	da := Denominator(a), db := Denominator(b),
	w := E(N*da*db)^k, j := Ind(n),
	Lambda(j, cond(eq(j,n/2), 1, omega(N*da*db, k*(i*da+na)*(j*db+nb)))))
));

Class(UDFT_NonTerm, DFT_NonTerm, rec(
    isReal := False,
));

Class(UDFT, UDFT_NonTerm, rec(
   abbrevs := [ 
    (n)    -> Checked(IsPosIntSym(n), AnySyms(n) or (n mod 2=0), [n, 1]),
    (n,k)  -> Checked(IsPosIntSym(n), AnySyms(n) or (n mod 2=0), 
	              IsIntSym(k), AnySyms(n,k) or Gcd(n,k)=1,               
		      [n, When(AnySyms(n,k), k, k mod n)]) ],
   omega4pow := (r,c)->4*r*c,
   terminate := self >> let(
	n:=self.params[1], k:=self.params[2], m:=MatSPL(DFT(2))^-1,
	DirectSum(Mat(m), I(n-2))^L(n,n/2) *
	DFT(n,k).terminate()),

   conjTranspose := self >> ObjId(self)(self.params[1], -self.params[2]).transpose()
));
UDFT1 := UDFT;

Class(UDFT2, UDFT_NonTerm, rec(
   abbrevs := [ 
    (n)    -> Checked(IsPosIntSym(n), AnySyms(n) or (n mod 2=0), [n, 1]),
    (n,k)  -> Checked(IsPosIntSym(n), AnySyms(n) or (n mod 2=0), 
	              IsIntSym(k), AnySyms(n,k) or Gcd(n,k)=1,               
		      [n, When(AnySyms(n,k), k, k mod (2*n))]) ],
    omega4pow := (r,c)->2*r*(2*c+1),
    terminate := self >> let(
	n:=self.params[1], k:=self.params[2], m:=MatSPL(DFT2(2,k))^-1,
	DirectSum(Mat(m), I(n-2))^L(n,n/2) *
	DFT2(n,k).terminate()),
));

Class(UUDFT, UDFT_NonTerm, rec(
   abbrevs := [ 
    (n)    -> Checked(IsPosIntSym(n), AnySyms(n) or (n mod 4=0), [n, 1]),
    (n,k)  -> Checked(IsPosIntSym(n), AnySyms(n) or (n mod 4=0), 
	              IsIntSym(k), AnySyms(n,k) or Gcd(n,k)=1,               
		      [n, When(AnySyms(n,k), k, k mod n)]) ],
    omega4pow := (r,c)->4*r*c,
    terminate := self >> let(
	n:=self.params[1], k:=self.params[2], m:=MatSPL(DFT(2))^-1,
	DirectSum(Mat(m), I((n-4)/2), Mat(m), I((n-4)/2))^L(n,n/2) *
	DFT(n,k).terminate()),
));
UUDFT1 := UUDFT;

Class(UUDFT2, UDFT_NonTerm, rec(
   abbrevs := [ 
    (n)    -> Checked(IsPosIntSym(n), AnySyms(n) or (n mod 4=0), [n, 1]),
    (n,k)  -> Checked(IsPosIntSym(n), AnySyms(n) or (n mod 4=0), 
	              IsIntSym(k), AnySyms(n,k) or Gcd(n,k)=1,               
		      [n, When(AnySyms(n,k), k, k mod (2*n))])  ],
    omega4pow := (r,c)->2*r*(2*c+1),
    terminate := self >> let(
	n:=self.params[1], k:=self.params[2], m:=MatSPL(DFT2(2,k))^-1,
	DirectSum(Mat(m), I((n-2)/2), Mat(m), I((n-2)/2))^L(n,n/2) *
	DFT2(n,k).terminate()),
));
