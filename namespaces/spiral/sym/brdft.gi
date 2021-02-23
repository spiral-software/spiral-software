
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# assuming the angles 2pi*a
alpha2pi := (n,i,a) -> Cond(i mod 2 = 0, (a + Int(i/2))/n, (1-a + Int(i/2))/n);
 
Mat_rDFT := (N,a,k) -> let(n:=N/2,
     ConcatList([0..n-1], i -> let(aa := alpha2pi(n,i,a), [ 
        Concatenation(     List([0..n-1], j -> cospi(k*2*j*aa)), 
                    (-1)^i*List([0..n-1], j -> -sinpi(k*2*j*aa))),
        Concatenation(     List([0..n-1], j -> sinpi(k*2*j*aa)), 
                    (-1)^i*List([0..n-1], j -> cospi(k*2*j*aa))) ])));

Mat_nrDFT := (N,a,k) -> let(n:=N/2,
     ConcatList([0..n-1], i -> let(
        raw := (i+a)/n,
        sign := When(raw > 1/2, -1, 1) * When(a > 1/2, -1, 1),
        aa := When(raw > 1/2, 1-raw, raw), [ 
        Concatenation(     List([0..n-1], j -> cospi(k*2*j*aa)), 
                      sign*List([0..n-1], j -> -sinpi(k*2*j*aa))),
        Concatenation(     List([0..n-1], j -> sinpi(k*2*j*aa)), 
                      sign*List([0..n-1], j -> cospi(k*2*j*aa))) ])));

_row_BRDFT_13 := (n,u,k) -> let(s := 1/sinpi(k*2*u),
   [ List([0..n-1], j -> -s*sinpi(k*2*u*(j-1))),
     List([0..n-1], j ->  s*sinpi(k*2*u*j)) ]);

# scaled = 2*cospi(u) * unscaled
# since 2*cospi(u) / sinpi(2*u) = 1/sinpi(u)
#
_row_BRDFT_24 := (n,u, k,scaled) -> let(s := When(scaled, 1/sinpi(k*u), 1/sinpi(k*2*u)),
   [ List([0..n-1], j -> -s*sinpi(k*2*u*(j-1/2))),
     List([0..n-1], j ->  s*sinpi(k*2*u*(j+1/2))) ]);

# F.T. for the polynomial algebra below, basis in time = (1, x, ..., x^(n-1))
#   R[x] / x^n - 1
#
Mat_BRDFT1 := (n,k) -> When(IsEvenInt(n),
   Concatenation(
       [List([0..n-1], i-> 1)],
       [List([0..n-1], i-> (-1)^i)],
       ConcatList([1..n/2-1], i -> _row_BRDFT_13(n, i/n, k))),
   Concatenation(
       [List([0..n-1], i-> 1)],
       ConcatList([1..(n-1)/2], i -> _row_BRDFT_13(n, i/n, k)))
);

Mat_BRDFT1U := (n,k) -> When(IsOddInt(n), Mat_BRDFT1(n), 
   Concatenation(
       [List([0..n-1], i-> When(IsEvenInt(i), 1, 0))],
       [List([0..n-1], i-> When(IsEvenInt(i), 0, 1))],
       ConcatList([1..n/2-1], i -> _row_BRDFT_13(n, i/n, k))));

# -- i1 dst1 / dst1 i1
# -- i1 dst5 / dst5


# F.T. for the polynomial algebra below, basis in time = (x^(1/2),x^(3/2),..., x^(n-1/2))
#   R[x] / x^n - 1
#
Mat_BRDFT2 := (n,k,scaled) -> When(IsEvenInt(n),
   Concatenation(
       [List([0..n-1], i-> 1)],
       [List([0..n-1], i-> (-1)^i)],
       ConcatList([1..n/2-1], i -> _row_BRDFT_24(n, i/n, k, scaled))),
   Concatenation(
       [List([0..n-1], i-> 1)],
       ConcatList([1..(n-1)/2], i -> _row_BRDFT_24(n, i/n, k, scaled))));
# - dct2/dst2 dst2/dst2 - breaks?
# - i1 dst6 dst6 -- ok
# - i1 dst5 dst5 -- ok

# F.T. for the polynomial algebras below, basis in time = (1,x,..., x^(n-1))
# u = 1/4, n - even or odd
#   R[x] / x^n + 1
#
# u <> 1/4, n - even ("skew" BRDFT3)
#   R[x] / x^n - 2cos(2*pi*u) x^(n/2) + 1
#
Mat_BRDFT3 := (n, u, k) -> When(IsEvenInt(n),
   ConcatList([0..n/2-1], i -> _row_BRDFT_13(n, 2*alpha2pi(n, i, u), k)),
   Concatenation(
       ConcatList([0..(n-3)/2], i -> _row_BRDFT_13(n, 2*alpha2pi(n, i, u), k)),
       [List([0..n-1], i-> (-1)^i)])
);
# dst3/dst3

# F.T. for the polynomial algebras below, basis in time = (x^(1/2),x^(3/2),..., x^(n-1/2))
# u = 1/4, n - even or odd
#   R[x] / x^n + 1
#
# u <> 1/4, n - even ("skew" BRDFT4)
#   R[x] / x^n - 2cos(2*pi*u) x^(n/2) + 1
#
Mat_BRDFT4 := (n, u, k, scaled) -> When(IsEvenInt(n),
   ConcatList([0..n/2-1], i -> _row_BRDFT_24(n, 2*alpha2pi(n, i, u), k, scaled)),
   Concatenation(
       ConcatList([0..(n-3)/2], i -> _row_BRDFT_24(n, 2*alpha2pi(n, i, u), k, scaled)),
       [List([0..n-1], i-> (-1)^i)])
);
# dst4/dst4 -- scale(brdft4)

# spiral> $ * scale(Mat_BRDFT4(16,1/4));
# [ [ 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
#   [ 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 ],
#   [ 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0 ],
#   [ 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0 ],
#   [ 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0 ],
#   [ 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0 ],
#   [ 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0 ],
#   [ 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0 ],
#   [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
#   [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ],
#   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
#   [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
#   [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ],
#   [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
#   [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
#   [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0 ] ]

Class(Base_BRDFT, NonTerminal, TaggedObjectMixin, rec(
   isReal := True,
   dims := self >> let(n:=self.params[1], [n, n]),
   print := (self, i, is) >> let(base_print := NonTerminal.print, 
       Print(base_print(self, i, is), 
           When(IsBound(self.tags), Print(".withTags(", self.tags, ")"))))
));

Class(rDFT, Base_BRDFT, rec(
   abbrevs := [ (n)       -> Checked(IsPosIntSym(n), [n, 1/4, 1]),
                (n,a,rot) -> Checked(IsPosIntSym(n), IsRatSym(a), IsIntSym(rot), 
                                     IsSymbolic(rot) or Gcd(n,rot)=1,
                    let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa, rot])) ],

   terminate := self >> Mat(Mat_rDFT(self.params[1], EvalScalar(self.params[2]), self.params[3])),
   hashAs := self >> ObjId(self)(self.params[1]).withTags(self.getTags()),
   HashId := self >> Concatenation([self.params[1]], self.getTags())
));

# BRDFT1(<n>) - real DFT of type 1 non-terminal
#  Freq. basis :
#      (1), (1,x)*, (1)
#  Time basis :
#      (1, x, ..., x^(n-1))
#  Polynomial algebra :
#      R[x] / x^n - 1
#
Class(BRDFT1, Base_BRDFT, rec(
   abbrevs := [ (n)   -> Checked(IsPosIntSym(n), [n, 1]),
                (n,rot) -> Checked(IsPosIntSym(n), IsIntSym(rot), [n, rot]) ],
   terminate := self >> let(mat := Mat(Mat_BRDFT1(self.params[1], EvalScalar(self.params[2]))),
       When(self.transposed, mat.transpose(), mat))
));

# Same as BRDFT1 but does not decompose polynomial algebra completely,
# keeping 1D spectral components fused (ie first spectral component is 
# R[x]/x^2-1), with basis (1,x) as in all other components.
#
Class(UBRDFT1, Base_BRDFT, rec(
   abbrevs := [ (n)   -> Checked(IsPosIntSym(n), [n, 1]),
                (n,rot) -> Checked(IsPosIntSym(n), IsIntSym(rot), [n, rot]) ],
   terminate := self >> Mat(Mat_BRDFT1U(self.params[1], EvalScalar(self.params[2])))
));


# BRDFT3(<n>) - real DFT of type 3 non-terminal
#  Freq. basis :
#      (1), (1,x)*, (1)
#  Time basis :
#      (1, x, ..., x^(n-1))
#  Polynomial algebra :
#      R[x] / x^n + 1
#
Class(BRDFT3, Base_BRDFT, rec(
   abbrevs := [ (n)   -> Checked(IsPosIntSym(n),           [n, 1/4, 1]),
                (n,a) -> Checked(IsPosIntSym(n), IsRatSym(a),
                    let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa, 1])),
                (n,a, rot) -> Checked(IsPosIntSym(n), IsRatSym(a), IsIntSym(rot), 
                    let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa, rot])) ],

   terminate := self >> Mat(Mat_BRDFT3(self.params[1], EvalScalar(self.params[2]), EvalScalar(self.params[3]))),
   hashAs := self >> ObjId(self)(self.params[1]),
));

# -----------------------------------
# Base changes and older misc stuff
# -----------------------------------

bblock := a -> let(aa := -2*cospi(2*a), [[-1,  aa], [-aa, aa^2-1]]);

bruun2 := (m,a) -> Checked((m mod 4)=0, let(
     a12 := When(a <= 1/2,
              [a/2, (a+1)/2],
              [(1-a)/2, (1-a+1)/2]),
         BlockMat([[I(m/2), Tensor(Mat(bblock(a12[1])), I(m/4))],
                   [I(m/2), Tensor(Mat(bblock(a12[2])), I(m/4))]])));

_omega := a -> E(Denominator(a))^Numerator(a);
_shift := lst -> Concatenation(lst{[2..Length(lst)]}, [lst[1]]);
_shift2 := lst -> Concatenation(lst{[2..Length(lst)]}, [-lst[1]]);

# Polynomial transform for p(x)
# P_{b,a} = [p_l(a_k)]_{k,l}
# where b = {p_0, ...} is basis, typically {1,x,x^2,...}
# and a = {a_0, ...} are roots of p(x)
#
# 2nd col is column of roots
#
# roots of x^n - omega{2pi*a}  (==omega( (a+i)/n ))
CycRoots := (n, a) -> List([0..n-1], i -> _omega((a+i)/n));

# roots of x^2n - 2 x^n cos(2*pi*f) + 1
# roots are ordered CCW on the unit circle
PhiRoots := (N, f) -> let(n:=Numerator(f), d:=Denominator(f),
    ConcatList([0..N-1], i -> [E(N*d)^(n+d*i), E(N*d)^(-n-d*(N-1-i))]));

PhiRoots2 := (N, f) -> let(n:=Numerator(f), d:=Denominator(f),
    Concatenation(
    List([0..N-1], i -> [E(2*N*d)^(n+d*i), E(2*N*d)^(-n-d*(2*N-1-i))])));

phiRootsNewOrder := (n, a) -> # first omega will have negative angle, we have to make it last => _shift
    _shift(ConcatList([0..n-1], i -> [_omega((-a+i)/n), _omega((a+i)/n)]));

phiRoots2NewOrder := (n, a) -> # first omega will have negative angle, we have to make it last and negate => _shift2
    _shift2(ConcatList([0..n-1], i -> [_omega((-a+i)/(2*n)), _omega((a+i)/(2*n))]));

Mat_BDFT := (n, a, rot) -> let(
    roots := phiRootsNewOrder(n/2, a),
    List([1..n], r -> List([0..n-1], c -> roots[r]^(rot*c))));

Mat_BDFT2 := (n, a, rot) -> let(
    roots := phiRoots2NewOrder(n/2, a),
    List([1..n], r -> List([0..n-1], c -> roots[r]^(rot*c))));

alphai := (n,i,a)  -> Cond(i mod 2 = 0, (a + Int(i/2))/n, (1-a + Int(i/2))/n);

time_r := a -> [[1, -cospi(2*a)/sinpi(2*a)],
                [0, 1/sinpi(2*a)]];

r_time := a -> [[1, cospi(2*a)],
                [0, sinpi(2*a)]];

B1_time_r := n -> DirectSum(I(1), List([1..Int((n-1)/2)], i-> Mat(time_r(i/n))), I(_even(n)));
B3_time_r := n -> DirectSum(List([0..Int(n/2)-1], i -> Mat(time_r((i+1/2)/n))), I(_odd(n)));

B3_r_time := (n,a) -> DirectSum(List([0..Int(n/2)-1], i -> Mat(r_time(alphai(n,i,a)))), I(_odd(n)));

#ls := (n,func) -> List([0..n-1], func);

#MatBRDFT3 := (N,a) -> MatSPL(BlockMat(let(n:=N/2,
#     ls(n, i->let(phi := [[0,-1],[1,2*cospi(2*alphai(n,i,a))]],
#    ls(n, j->Mat(phi^(2*j))))))));

#MatBRDFT33 := (N,a) -> MatSPL(BlockMat(let(n:=N/2,
#     ls(n, i->let(phi := [[0,1],[1,-2*E(4)*sinpi(2*alphai(n,i,a))]],
#    ls(n, j->Mat(phi^(2*j))))))));


#r:=MatSPL(B3_time_r(7)*SRDFT3(7));

#MatBRDFT4 := (N,a) -> MatSPL(BlockMat(let(n:=N/2,
#     ls(n, i->let(start:=MatSPL(BRDFT4(2,alphai(n,i,a))), phi := [[0,-1],[1,2*cospi(2*alphai(n,i,a))]],
#    ls(n, j->Mat(start*phi^(2*j))))))));

# roots := List(PhiRoots(100, 1/(4*23)), ComplexCyc);
# DoForAll(last, x->AppendTo("roots", ReComplex(x), " ", ImComplex(x), "\n"));
# PrintTo("tmp.gnuplot", "plot 'roots' with lp");
# SYS_EXEC("gnuplot tmp.gnuplot -");

# m88:=MatSPL(M(16,2))*MatSPL(DFT3(16))*MatSPL(bruun2(16,1/4))^-1;
scale := mat -> List(mat, x->x/x[1]);
# Note:
#   GathExtend(n, Even1).transpose() * BSkewDFT3(n, a) * VStack(I(n/2), O(n/2)) == SkewDTT(DCT3(n/2), 2*a)
#   GathExtend(n, Odd1).transpose()  * BSkewDFT4(n, a) * VStack(I(n/2), O(n/2)) == SkewDTT(DCT4(n/2), 2*a)
#
# BRDFT3 supports Odd0 input symmetry
# BRDFT4 supports Odd1 input symmetry (incorrectly defined in Spiral)
#
#   x3  := n ->       MatSPL(Gath(H(n, n/2, 0, 2)) * BRDFT3(n) * GathExtend(n, Odd0));   == PolyDTT(DCT3(n/2)) == DCT3(n/2), other half=zero
#   x33 := n ->       MatSPL(Gath(H(n, n/2, 1, 2)) * BRDFT3(n) * GathExtend(n, Even00)); == PolyDTT(DST3(n/2)), other half=first half * cospi( (2*i+1) / m), eg 1/8, 3/8, 5/8, 7/8 for n=8
#   y3  := n ->     2*MatSPL(Gath(H(n, n/2, 0, 2)) * BRDFT3(n) * DirectSum(I(1), -J(n-1))
#                  1/2*                                               * GathExtend(n, Odd1)));  == PolyDTT(DCT4(n/2)), other half=same
#   y33 := n ->     2*MatSPL(Gath(H(n, n/2, 0, 2)) * BRDFT3(n) * DirectSum(I(1), -J(n-1)) *
#                                                                   GathExtend(n, Even1));  == PolyDTT(DST4(n/2)), other half=neg
#
#   z3  := n ->       MatSPL(Gath(H(n, n/2, 1, 2)) * BRDFT4(n) * GathExtend(n, Odd1));  == PolyDTT(DCT4(n/2)), !new def
#   z33 := n -> scale(MatSPL(Gath(H(n, n/2, 0, 2)) * BRDFT4(n) * GathExtend(n, Even1)));  == PolyDTT(DST4(n/2)), !new def
#
#   x44 := n -> scale((-E(4)) * MatSPL(Gath(H(n, n/2, 1, 2)) * BRDFT4(n) * GathExtend(n, Odd1)));  == PolyDTT(DCT4(n/2))
#   x4 :=  n ->       (-E(4)) * MatSPL(Gath(H(n, n/2, 0, 2)) * BRDFT4(n) * GathExtend(n, Even1));  == DST4(n/2)
#   x4 :=  n ->  scale(-E(4)) * MatSPL(Gath(H(n, n/2, 0, 2)) * BRDFT4(n) * GathExtend(n, Even1));  == PolyDTT(DST4(n/2))

#   spiral> pm(Gath(H(16,8,0,2))*BRDFT3(16)*GathExtend(16, Odd0));
#
#   y := (n,a) -> MatSPL(Gath(H(n, n/2, 0, 2)) * BRDFT3(n,a) * DirectSum(I(1), -VStack(O(2,n/2-1), RI(n/2,n/2-1), O(n/2-3, n/2-1))));
#  == PolyDTT(SkewDTT(DST3(n/2), 2*a))
#


# # Decomposes C[x]/phi(.) or R[x]/phi(.) into 2-dim spectral
# # components with basis 1,x
# #
# Class(BRDFT3, NonTerminal, rec(
#     abbrevs := [ (n)   -> Checked(IsPosInt(n),           [n, 1/4]),
#                  (n,a) -> Checked(IsPosInt(n), IsRatSym(a),
#            let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa])) ],

#     terminate := self >> let(n:=self.params[1], a:=self.params[2],
#   M(n,n/2) * self.xmat() * BSkewDFT3(n, a).terminate()),

#     xmat := self >> let(n:=self.params[1], a:=self.params[2],
#   roots := Take(PhiRoots(n/2,a),n/2), cc := Global.Conjugate,
#   DirectSum(List(roots, r->Mat([[1,r],[1,cc(r)]]^-1)))^LIJ(n)),

#     isReal := True,
#     dims := self >> let(n:=self.params[1], [n, n]),
#     HashId := self >> self.params[1]
# ));

# Class(BRDFT4, BRDFT3, rec(
#     terminate := self >> let(n:=self.params[1], a:=self.params[2],
#   M(n,n/2) * self.xmat() * BSkewDFT4(n, a).terminate()),
# #    xmat := self >> let(n:=self.params[1], a:=self.params[2],
# # roots := Take(PhiRoots2(n/2,a),n/2), cc := Global.Conjugate,
# # DirectSum(List(roots, r->Mat([[r,r^3],[-cc(r),-cc(r)^3]]^-1)))^LIJ(n)),
# ));

#Class(BRDFT4, BRDFT3, rec(
#    terminate := self >> let(n:=self.params[1], a:=self.params[2],
#   Mat(scale(MatSPL(M(n,n/2) * self.xmat() * BSkewDFT4(n, a))))),
##    xmat := self >> let(n:=self.params[1], a:=self.params[2],
##  roots := Take(PhiRoots2(n/2,a),n/2), cc := Global.Conjugate,
##  DirectSum(List(roots, r->Mat([[r,r^3],[-cc(r),-cc(r)^3]]^-1)))^LIJ(n)),
#));

# -------------------------------------------------------------
# Other transforms
# -------------------------------------------------------------

# roots of BSkewDFT3(2*n, u)
#   { E(n)^(2u+i) } union { E(n)^(-2u-i) }
#
Class(BSkewDFT3, NonTerminal, TaggedObjectMixin, rec(
    abbrevs := [ (n)   -> Checked(IsPosIntSym(n),           [n, 1/4, 1]),
                 (n,a) -> Checked(IsPosIntSym(n), IsRatSym(a),
                    let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa, 1])),
                 (n,a,rot) -> Checked(IsPosIntSym(n), IsRatSym(a), IsIntSym(rot), 
                    let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa, rot])) ],

    terminate := self >> let(N := self.params[1], a := self.params[2], rot := self.params[3],
        roots := PhiRoots(N/2, EvalScalar(a)),
        Mat(List([1..N], r ->
                List([0..N-1], c -> roots[r]^(rot*c))))),

    isReal := False,
    dims := self >> let(n:=self.params[1], [n, n]),
    hashAs := self >> ObjId(self)(self.params[1]),
));

Class(PSkewDFT3, NonTerminal, TaggedObjectMixin, rec(
    abbrevs := [ (n)   -> Checked(IsPosInt(n),                  [n, 1/4, 1]),
                 (n,a) -> Checked(IsPosInt(n), IsRatSym(a),  
                     let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa, 1])),
                 (n,a,rot) -> Checked(IsPosInt(n), IsRatSym(a), IsIntSym(rot), 
                     let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa, rot])) ],

    terminate := self >> let(N := self.params[1], a := EvalScalar(self.params[2]),
        rot := EvalScalar(self.params[3]), 
        roots := PhiRoots(N/2, a),
        Mat(List([1..N], r ->
            Concatenation(
                List([0..N/2-1], c -> roots[r]^(rot*c)), 
                List([N/2..N-1], c -> (roots[r]^(rot*c) - roots[r]^(rot*(c-N/2))*cospi(2*a*rot))/sinpi(2*a*rot)))))),

    isReal := False,
    dims := self >> let(n:=self.params[1], [n, n]),
    hashAs := self >> ObjId(self)(self.params[1]),
));

Class(BSkewDFT4, BSkewDFT3, rec(
    terminate := self >> let(N := self.params[1], a := EvalScalar(self.params[2]), rot := EvalScalar(self.params[3]), 
	roots := PhiRoots2(N/2, a),
        Mat(List([1..N], r -> 
	     List([0..N-1], c -> roots[r]^(rot*(2*c+1)))))),
    hashAs := self >> ObjId(self)(self.params[1]),
));

# NOTE: rot is missing
Class(BSkewPRDFT, PRDFT3, TaggedObjectMixin, rec(
    abbrevs := [ (n)   -> Checked(IsPosInt(n),           [n, 1/4]),
                 (n,a) -> Checked(IsPosInt(n), IsRatSym(a),
             let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa])) ],

    terminate := self >> let(N := self.params[1], a := EvalScalar(self.params[2]),
	roots := PhiRoots(N/2, a),
        Mat(List([0..N-1], r -> 
	     When(r mod 2 = 0, 
		 List([0..N-1], c -> self.projRe(roots[1+r/2]^c)),
		 List([0..N-1], c -> self.projIm(roots[1+(r-1)/2]^c)))))),
    hashAs := self >> ObjId(self)(self.params[1]),
));
# NOTE: rot is missing
Class(BSkewRDFT, PRDFT3, TaggedObjectMixin, rec(
    abbrevs := [ (n)   -> Checked(IsPosInt(n),           [n, 1/4]),
                 (n,a) -> Checked(IsPosInt(n), IsRatSym(a),
             let(aa := When(IsRat(a) and a>1/2, 1-a,a), [n, aa])) ],

    terminate := self >> let(N := self.params[1], a := EvalScalar(self.params[2]),
	K(N,2) * BSkewPRDFT(N,a).terminate()),

    hashAs := self >> ObjId(self)(self.params[1]),
));
