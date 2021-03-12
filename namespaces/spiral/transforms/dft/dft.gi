
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F DFT(<size>, [<exp>]) - Discrete Fourier Transform non-terminal
#F                         gcd(size, exp) = 1
#F Definition: (n x n)-matrix [ e^((2*pi*exp*i)*k*l/n) | k,l = 0...n-1 ],
#F             observe that canonical root e^(2*pi*i/n) is exponentiated to <exp>
#F             default exp=1,
#              i = sqrt(-1)
#F Example, direct  : DFT(8) ,
#F          inverse : DFT(8, -1)
#F          rotated : DFT(8, 3)
#F
Class(DFT_NonTerm, TaggedNonTerminal, rec(
    abbrevs := [
    (n)       -> Checked(IsPosIntSym(n),
        [_unwrap(n), 1]),
    (n,k)     -> Checked(IsPosIntSym(n), IsIntSym(k), AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
        [_unwrap(n), When(AnySyms(n,k), k, k mod _unwrap(n))])
    ],

    hashAs := self >> ObjId(self)(self.params[1], 1).withTags(self.getTags()),

    dims := self >> When(self.isReal(), 2* [ self.params[1], self.params[1] ], [ self.params[1], self.params[1] ]),

    terminate := self >> let(N := self.params[1], K := self.params[2],
        t := List([0..N-1], r -> List([0..N-1], c -> E(4*N)^(K*self.omega4pow(r,c)))),
            When(self.isReal(), MatAMat(RC(Mat(t)).toAMat()), Mat(t))),

    isReal := self >> false,
    SmallRandom := () -> Random([2..16]),
    LargeRandom := () -> 2 ^ Random([6..15]),

    normalizedArithCost := self >> let(n := self.params[1],
        floor(5.0 * n * log(n) / log(2.0))),

    TType := T_Complex(TUnknown)
));

Declare(DFT, DFT2, DFT3, DFT4);

Class(DFT, DFT_NonTerm, rec(
    transpose     := self >> DFT(self.params[1], self.params[2]).withTags(self.getTags()),
    conjTranspose := self >> DFT(self.params[1], -self.params[2]).withTags(self.getTags()),
    inverse := self >> self.conjTranspose(),
    omega4pow := (r,c) -> 4*r*c,
    printlatex := (self) >> Print(" \\DFT_{", self.params[1], "} ")
));

DFT1 := DFT;
IDFT := n -> DFT(n,-1);


Class(DFT2, DFT_NonTerm, rec(
    abbrevs := [
    (n)       -> Checked(IsPosIntSym(n),
        [_unwrap(n), 1]),
    (n,k)     -> Checked(IsPosIntSym(n), IsIntSym(k), AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
        [_unwrap(n), When(AnySyms(n,k), k, k mod (2*_unwrap(n)))]),
    ],

    transpose     := self >> DFT3(self.params[1], self.params[2]).withTags(self.getTags()),
    conjTranspose := self >> DFT3(self.params[1], -self.params[2]).withTags(self.getTags()),
    inverse := self >> self.conjTranspose(),
    omega4pow := (r,c) -> 2*r*(2*c+1),
));

Class(DFT3, DFT_NonTerm, rec(
    abbrevs := [
    (n)       -> Checked(IsPosIntSym(n),
        [_unwrap(n), 1]),
    (n,k)     -> Checked(IsPosIntSym(n), IsIntSym(k), AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
        [_unwrap(n), When(AnySyms(n,k), k, k mod (2*_unwrap(n)))]),
    ],

    transpose     := self >> DFT2(self.params[1], self.params[2]).withTags(self.getTags()),
    conjTranspose := self >> DFT2(self.params[1], -self.params[2]).withTags(self.getTags()),
    inverse := self >> self.conjTranspose(),
    omega4pow := (r,c) -> (2*r+1)*2*c,
));

Class(DFT4, DFT_NonTerm, rec(
    abbrevs := [
    (n)       -> Checked(IsPosIntSym(n),
        [_unwrap(n), 1]),
    (n,k)     -> Checked(IsPosIntSym(n), IsIntSym(k), AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
        [_unwrap(n), When(AnySyms(n,k), k, k mod (4*_unwrap(n)))]),
    ],

    transpose     := self >> DFT4(self.params[1], self.params[2]).withTags(self.getTags()),
    conjTranspose := self >> DFT4(self.params[1], -self.params[2]).withTags(self.getTags()),
    inverse := self >> self.conjTranspose(),
    omega4pow := (r,c) -> (2*r+1)*(2*c+1),
));


paramFunc := function(i, m, f, n)
    local res;

#    res := List([0..m-1], e -> SubstBottomUp(Copy(f), i, g -> e));
    res := List([0..m-1], e -> f(e,m));
    res := diagDirsum(res);
    res := fCompose(res, fTensor(fBase(m, i), fId(n)));

    return res;
end;

#F -----------------------------------------------------------------------------
#F Tw1(n, d)
#F Tw1(n, d, k) : diagonal function for Cooley-Tukey FFT, namely
#F   <d> | <n> and
#F   Tw1(n, d, k) = direct_sum_(i = 1)^n/d diag(w_n^0, ..., w_n^(d-1)^i
#F   If the second version is used, w_n is replaced by w_n^k, gcd(<n>, <k>) = 1.
#F
#F -----------------------------------------------------------------------------
Tw1 := (n,d,k) -> Checked(
    IsPosIntSym(n), IsPosIntSym(d), IsIntSym(k),
    fCompose(dOmega(n,k),diagTensor(dLin(div(n,d), 1, 0, TInt), dLin(d, 1, 0, TInt))));

Tw2 := (n,d,k) -> Checked(
    IsPosIntSym(n), IsPosIntSym(d), IsIntSym(k),
    fCompose(dOmega(2*n,k),
         diagTensor(dLin(div(n,d), 2, 1, TInt), dLin(d, 1, 0, TInt))));

Tw3 := (n,d,k) -> Checked(
    IsPosIntSym(n), IsPosIntSym(d), IsIntSym(k),
    fCompose(dOmega(2*n,k),
         diagTensor(dLin(div(n,d), 1, 0, TInt), dLin(d, 2, 1, TInt))));

Tw4 := (n,d,k) -> Checked(
    IsPosIntSym(n), IsPosIntSym(d), IsIntSym(k),
    fCompose(dOmega(4*n,k),
         diagTensor(dLin(div(n,d), 2, 1, TInt), dLin(d, 2, 1, TInt))));

# j-row, i-col
Class(Twid, DiagFunc, rec(
    def := (N, n, rot, a, b, i) -> rec(size:=n),
    lambda := self >> let(
        N := self.params[1], n := self.size, rot := self.params[3],
        a := self.params[4], b := self.params[5], i := self.params[6],
        na := Numerator(a),   nb := Numerator(b),
        da := Denominator(a), db := Denominator(b),
        j := Ind(n),
        Lambda(j, omega(N*da*db, rot*(i*da+na)*(j*db+nb)))),

    range := self >> TComplex,
    domain := self >> self.params[2]
));

Class(ScaledTwid, DiagFunc, rec(
    def := (N, n, k, a, b, i, scaling) -> rec(size:=n),
    lambda := self >> let(N:=self.params[1], n:=self.size, k:=self.params[3],
        a := self.params[4], b := self.params[5], i := self.params[6], scaling := self.params[7],
        na := Numerator(a),   nb := Numerator(b),
        da := Denominator(a), db := Denominator(b),
        j := Ind(n),
        Lambda(j, scaling * omega(N*da*db, k*(i*da+na)*(j*db+nb)))),
    range := self >> TComplex
));

#F TC6(radix, iteration, exp) - diagonal generating function for Stockham algorithm
#F					generates the omega-matrix as described in Van Loin - radix 4 Stockham framework
#F					first parameter of dLin stands for the total dimension of returned function
TC6 := (r, iter, exp) -> Checked(
    IsPosIntSym(r), IsInt(exp),
	fCompose(dOmegaPow(r^(iter+1), 1, exp), dLin(r^iter, 1, 0, TInt)));

#F TC5(radix, iteration) - diagonal generating function for Stockham algorithm - generates a diagonal function
#DP: IsInt(iter) can't be checked, because iter is calculated based on a free variable and not yet evaluated here
TC5 := (r, iter) -> Checked(
    IsPosIntSym(r),
    diagDirsum(fConst(r^iter,1), fCompose(dOmega(r^(iter+1), 1), dLin(r^iter, 1, 0, TInt))));

#F TC4(radix, iteration) - diagonal generating function for Stockham algorithm
#F					generates the omega-matrix as described in Van Loin - Stockham Autosort Network
Class(TC4, DiagFunc, rec(
    abbrevs := [ (rdx, iter, exp) -> [rdx, iter, exp],
	         (rdx, iter, exp) -> [rdx, iter, exp]],
    def := (rdx, iter, exp) -> rec(size := rdx^iter),

    lambda := self >> let(rdx:=self.params[1],
	iter:=self.params[2], n:=rdx^(iter+1), i:=Ind(rdx^iter), exp:=self.params[3],	#n = L (van Loin), i = L/rdx (van Loin)
	Lambda(i, omega(n,i)^exp)),

    range := self >> TInt,
));


Class(Stockham_radix, DiagFunc, rec(
    gen := meth(self, rdx, j)
		local diag;

        if IsInt(rdx) and rdx=2 then
			diag := diagDirsum(TC6(rdx, j, 0), TC6(rdx, j, 1));
		elif rdx = 4 then
			diag := diagDirsum(TC6(rdx, j, 0), TC6(rdx, j, 1), TC6(rdx, j, 2), TC6(rdx, j, 3));
		elif rdx = 8 then
			diag := diagDirsum(TC6(rdx, j, 0), TC6(rdx, j, 1), TC6(rdx, j, 2), TC6(rdx, j, 3), TC6(rdx, j, 4), TC6(rdx, j, 5), TC6(rdx, j, 6), TC6(rdx, j, 7));
		elif rdx = 16 then
			diag := diagDirsum(TC6(rdx, j, 0), TC6(rdx, j, 1), TC6(rdx, j, 2), TC6(rdx, j, 3), TC6(rdx, j, 4), TC6(rdx, j, 5), TC6(rdx, j, 6), TC6(rdx, j, 7),
								TC6(rdx, j, 8), TC6(rdx, j, 9), TC6(rdx, j, 10), TC6(rdx, j, 11), TC6(rdx, j, 12), TC6(rdx, j, 13), TC6(rdx, j, 14), TC6(rdx, j, 15));
		elif rdx > 16 then
            Error("Radix > 16 not supported");
        fi;

        return diag;
    end,

    extract := meth(self, list, elems)
		local i, lc, stride;

		lc := [];
		stride := Length(list)/elems;

		for i in [0..elems-1] do
			lc[i+1] := list[1 + i*stride];
		od;

		return lc;
    end,
));


#F TC2(size, radix, c, k) - diagonal generating function for Pease algorithm
#F
#F TC2 is (5) in Jeremy's Pease paper.
#F
#
# Note: due to current limitations, Spiral is unable to evaluate the
# dimensions of this function properly.
#
# example: TDiag(fPrecompute(TC(N, rdx, j, m)))
#
TC2 := (n, r, c, k) -> Checked(
    IsPosIntSym(n), IsPosIntSym(r), IsPosIntSym(k),
    fCompose(dOmega(n / r^c, k),
	diagTensor(diagTensor(dLin(n/(r^(c+1)), 1, 0, TInt), fConst(r^c, 1)),
		dLin(r, 1, 0, TInt))));
	
#F TC(size, radix, c, k) - diagonal generating function for Pease algorithm
#F
#F TC is (5) in Jeremy's Pease paper.
#F
Class(TC, DiagFunc, rec(
    abbrevs := [ (n, r, c) -> [n, r, c, 1],
	         (n, r, c, k) -> [n, r, c, k]],
    def := (n, r, c, k) -> rec(size := n),
    lambda := self >> let(n:=self.params[1], r:=self.params[2],
	c:=self.params[3], k:=self.params[4], i:=Ind(n),
	Lambda(i, omega(n/(r^c), k*idiv(i, r^(c+1)) * imod(i, r)))),
    
    domain := (self) >> self.params[1],
    range := self >> TComplex,
));

#F TCBase(size, radix, k) - diagonal base function for Pease algorithm.
#F This is equivalent to:
#F      L(size,radix) * TC(size, radix, 0, k) * L(size, size/radix).
#F
#F TC is (5) in Jeremy's Pease paper.
#F
Class(TCBase, DiagFunc, rec(
    abbrevs := [ (n, r) -> [n, r, 0, 1],
	         (n, r, k) -> [n, r, 0, k]],
    def := (n, r, c, k) -> rec(size := n-(n/r)),
    lambda := self >> let(n:=self.params[1], r:=self.params[2],
	c:=self.params[3], k:=self.params[4], i:=Ind(n - n/r),
	Lambda(i, omega(n/(r^c), k*idiv(
	    idiv((i+n/r), (r^c)), (n/r^(c+1))) *
		imod(idiv((i+n/r), (r^c)), n/(r^(c+1)))))
		),

    range := self >> TComplex,

));



# TI(n, r, c) - Diagonal generating function for Iterative FFT algorithm
#    where r is the radix, n = size, and c is the iteration
Class(TI, DiagFunc, rec(

    abbrevs := [ (n, r, c) -> [n, r, c, 1],
                 (n, r, c, k) -> [n, r, c, k] ],

    def := (n, r, c, k) -> rec(size := n/(r^c)),
    lambda := self >> let(n:=self.params[1], r:=self.params[2],
	c := self.params[3], k:=self.params[4],
	TC(n/(r^c), r, 0, k).lambda()),

    range := self >> TComplex,
    domain := self >> self.params[1]/(self.params[2]^self.params[3]),
));

# TIFold(n, r, c) - Diagonal generating function for Iterative FFT algorithm
#    where r is the radix, n = size, and c is the iteration.  This is
#    the same as TI, except we will implement it differently because the
#    surrounding structure is not wide enough to perform all the optimizations.
Class(TIFold, DiagFunc, rec(

    abbrevs := [ (n,r,c) -> [n,r, c, 1],
           [ (n, r, c, k) -> [n, r, c, k]]],

    def := (n, r, c, k) -> rec(size := n/(r^c)),
    lambda := self >> let(n:=self.params[1], r:=self.params[2],
	c := self.params[3], k:=self.params[4],
	TC(n/(r^c), r, 0).lambda()),

    range := self >> TComplex,
	
));



#F ZDFT( <size>, <rshift>, <lshift> )
#F    Discrete Fourier Transform with a cyclic shift on
#F    one or both sides
Class(ZDFT, NonTerminal, rec(
    abbrevs := [ (n,r,l) -> Checked(IsInt(n), n > 0, [n, r, l]) ],
    dims := self >> [ self.params[1], self.params[1] ],

    terminate := self >> let(N:=self.params[1], K:=self.params[2],
    Mat(List([0..N-1],
         r -> List([0..N-1], c -> E(N)^(K*r*c))))),

    transpose := self >> self.__bases__[1](self.params[1], self.params[3], self.params[2]),
    isReal := False,
    SmallRandom := () -> Random([2..16]),
    LargeRandom := () -> 2 ^ Random([6..15])
));


#
# BRDFT(k) = BR(k)*DFT(k)
# This is the same as DRDFT(k,2) = DR(k,2)*DRDFT(k,2)
Class(BRDFT, NonTerminal, rec
(
  abbrevs   := [ (n) -> Checked(IsPosInt(n), [n, []]) ],
#D                 (n, pv) -> Checked(IsPosInt(n), IsList(pv), [n, pv]) ],
  dims      := self >> let(size := self.params[1], [size, size]),
  terminate := self >> Compose(DR(self.params[1], 2), DFT(self.params[1])),
  transpose := self >> Copy(self),
  isReal    := self >> true,
  SmallRandom := () -> Random([2..5]),
  LargeRandom := () -> Random([6..15]),
#D  setpv := meth(self, pv)
#D    local s;
#D    s:= Copy(self);
#D    s.params[2] := pv;
#D    return s;
#D  end,
#D  tagpos := 2

));

#
# DFTBR(k, r) = DFT(k)*BR(k, r)
# This is the same as DFTDR(k,2) = DRDFT(k,2)*BR(k,2)
Class(DFTBR, NonTerminal, rec(
  abbrevs   := [ (n) -> Checked(IsPosInt(n), [n, []]) ],
#D                 (n, pv) -> Checked(IsPosInt(n), IsList(pv), [n, pv]) ],
  dims      := self >> let(size := self.params[1], [size, size]),
  terminate := self >> Compose(DFT(self.params[1]), DR(self.params[1], 2)),
  transpose := self >> Copy(self),
  isReal    := self >> false,
  SmallRandom := () -> Random([2..5]),
  LargeRandom := () -> Random([6..15]),
#D  setpv := meth(self, pv)
#D    local s;
#D    s:= Copy(self);
#D    s.params[2] := pv;
#D    return s;
#D  end,
#D  tagpos := 2

));


Class(DRDFT_Base, TaggedNonTerminal, rec(
  abbrevs   := [ (n,k, r)      -> Checked(IsPosIntSym(n), IsPosIntSym(r), IsInt(k), [n, k, r]) ],
#D                 (n, r, pv) -> Checked(IsPosIntSym(n), IsPosIntSym(r), IsList(pv), [n, r, pv]) ],
  dims      := self >> let(size := self.params[1], [size, size]),
  isReal    := self >> false,
  SmallRandom := () -> Random([2..5]),
  LargeRandom := () -> Random([6..15]),
#D  setpv := meth(self, pv)
#D    local s;
#D    s:= Copy(self);
#D    s.params[3] := pv;
#D    return s;
#D  end,
#D  tagpos := 3
));

Declare(DFTDR);

# DRDFT(n, k, r) = DR(n, r)*DFT(n, k)
#
Class(DRDFT, DRDFT_Base, rec(
  terminate := self >> Compose(DR(self.params[1], self.params[3]), DFT(self.params[1], self.params[2])),
  transpose := self >> ApplyFunc(DFTDR, self.params).withTags(self.getTags()),
));

#
# DFTDR(n, k, r) = DFT(n,k)*DR(n, r)
Class(DFTDR, DRDFT_Base, rec(
  terminate := self >> Compose(DFT(self.params[1], self.params[2]), DR(self.params[1], self.params[3])),
  transpose := self >> ApplyFunc(DRDFT, self.params).withTags(self.getTags()),
));
