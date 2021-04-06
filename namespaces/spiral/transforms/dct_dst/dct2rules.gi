
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RulesFor(DCT2, rec(
    #F DCT2_DCT2and4: 1977
    #F
    #F   DCT2_n = perm * (DCT2_n/2 dirsum DCT4_n/2^perm) * (1 tensor DFT_2)^perm
    #F
    #F   Chen/Fralick/Smith: 
    #F     A Fast Computational Algorithm for the
    #F     Discrete Cosine Transform, IEEE Trans. on Comm., 1977, pp. 1004--1009.
    #F   Wang: 
    #F     Reconsideration of --above--
    #F     Circ., Systems, and Signal Proc., 1983, 121--123.
    #F   Rao/Yip: 
    #F     Discrete Cosine Transform, Academic Press, 1990, pp. 53
    #F
    DCT2_DCT2and4 := rec(
	info         := "DCT2'_n --> DCT2'_n/2, DCT4_n/2",
	isApplicable := P -> P[1] > 2 and P[1] mod 2 = 0,
	allChildren  := P -> [[ DCT2(P[1]/2), DCT4(P[1]/2) ]],
	rule         := (P, C) -> 
	    LIJ(P[1]) *
	    DirectSum(C[1], C[2] ^ J(P[1]/2)) *
	    Tensor(I(Int((P[1] / 2))), F(2)) ^ LIJ(P[1]) # this line was blocks1(P[1])
    ),

    #F DCT2_toRDFT:
    #F
    #F   DCT2_n = blocks * RDFT_n * perm
    #F 
    #F for n even
    #F
    DCT2_toRDFT := rec (
	info         := "DCT2_n -> RDFT_n",
	isApplicable := P -> IsEvenInt(P[1]), 
	allChildren  := P -> [[ SRDFT(P[1]) ]], 

	rule := (P, C) -> let(i := Ind(P[1]/2-1),
            # first an X shaped matrix where opposite diagonal is lowered by 1
            LIJ(P[1]).transpose() *
	    DirectSum(I(1), 
		      When(i.range > 0, IterDirectSum(i, i.range, Rot(fdiv(1+i, 2*P[1]))*J(2)), []),
		      Diag(Sqrt(1/2))) *
	    C[1] * K(P[1], 2)
	)
    ),
#rdft = d os dct p7
#dct = os^-1 d^-1 rdft * p7^-1
    DCT2_toRDFT_odd := rec(
	info             := "DCT2_n -> RDFT_n",
	isApplicable     := P -> IsOddInt(P[1]),
        allChildren      := P -> [[ PRDFT1(P[1]) ]], 
	forTransposition := true,
	rule := (P, C) -> let(N := P[1], 
	    OS(N, 2).transpose() *
	    Diag(List([0..N-1], i -> (-1)^i)) *
	    DirectSum(Mat([[1,0]]), LIJ(N-1).transpose()) *
	    C[1] *
	    perm7(N).transpose()
	)
    ),

    #F DCT2_PrimePowerInduction:
    #F
    #F   DCT2'_n = perm * sparse * (1 tensor DCT2'_n/3) * 
    #F             ( dirsum of 3x3 blocks ) ^ perm * perm
    #F
    #F   Pueschel/Moura: Discrete Cosine and Sine Transforms
    #F 
    DCT2_PrimePowerInduction := rec (
	info             := "DCT2_3n -> DCT2_n",
	isApplicable := P -> P[1] mod 3 = 0 and P[1] <> 3,
	allChildren  := P -> [[ DCT2(P[1]/3) ]],

        # the sparse matrix occuring in the rule; it has the form
        # [ [ I_n, Z_n ], [ Z_n, I_n ] ], Z_n has only 1's on the upper 
        # diagonal; n = L/3
	sparsemat := function ( n )
	    local L, i;
            # diagonal 1's
	    L := List([1..2*n], i -> [i, i, 1]);
            # the other 1's
	    for i in [1..n-1] do  Add(L, [i, n+1+i, 1]);   od;
	    for i in [1..n-1] do  Add(L, [n+i, i+1, 1]);   od;
	    return Sparse(L);
	end,

        # the 3x3 blocks occuring; so to say the twiddle factors
        # note: the block for i = (n-1)/2 is a DCT of size 3 and can be done
        # in 4 adds, 2 mults; but here only in 5 adds and 2 mults
	block3 := function ( i, n )
	    local M, H, M1, ii;
	    ii := i + 1/2;
	    M := TransposedMat(
		[ [ 1,                1,                       1                      ],
		  [CosPi(  ii/(3*n)), CosPi(  (2*n-ii)/(3*n)), CosPi(  (2*n+ii)/(3*n))],
		  [CosPi(2*ii/(3*n)), CosPi(2*(2*n-ii)/(3*n)), CosPi(2*(2*n+ii)/(3*n))] ]
	    );
	    M := 3 * DiagonalMat([1, 1/2, 1/2]) * M^-1;
            # now M has 1's in the first row
	    H := [[1, 1, 1], [1, -1, 0], [1, 0, -1]];
	    M1 := M*H^-1;
            # M*H has the structure 1 dirsum 2x2 block
            # get the 2x2 block
	    M1 := Sublist(List(M1, r -> Sublist(r, [2,3])), [2,3]);
	    return DirectSum(I(1), Mat(M1)) * Mat(H);
	end,

	rule := (self, P, C) >> let(n := P[1]/3, 
	    L(3*n, n) *
	    DirectSum(I(n), self.sparsemat(n)) *
	    Tensor(I(3), C[1]) *
	    (DirectSum(List([1..n], i -> self.block3(i, n))) ^ L(3*n, n)) *
	    IJ(3*n, n))
    ),

#F DCT2_PrimeFactor: 1985
#F
#F   DCT2_nm = P1 * (1_(n+m-1) dirsum (1_(nm-n-m+1) tensor F_2)) *
#F             P2 * (DCT2_n tensor DCT2_m) * P3,    gcd(n, m) = 1
#F
#F Yang/Narasimha: Prime Factor Decomposition of the Discrete Cosine Transform,
#F   Proc. ICASSP, pp. 772--775, 1985
#F
#F see also:
#F Feig/Linzer: Scaled DCT's on Input Sizes that Are Composite, IEEE Transactions on 
#F   Signal Processing 43(1), pp. 43--50, 1995
#F
DCT2_PrimeFactor := rec (
  info             := "DCT2_nm -> DCT2_n tensor DCT2_m",
  isApplicable     := P -> not IsPrimePowerInt(P[1]), 
  allChildren      := P -> List(DivisorPairsRP(P[1]), p -> [ DCT2(p[1]), DCT2(p[2]) ]),

  rule := function ( P, C )
    local n, n1, n2, dctperm1, dctperm2;

    n1 := C[1].dimensions[1];
    n2 := C[2].dimensions[1];
    n  := P[1];

    # first permutation
    dctperm1 := function ( n1, n2 )
      local n, g1, g2, L, i, j;

      n  := n1*n2;  
      g1 := QuotientMod(1, n2, n1);
      g2 := n2 - QuotientMod(1, n1, n2);

      # indices
      L := [ ];
      for i in [0..n1 - 1] do
	for j in [0..n2 - 1] do
	  if (i + j) mod 2 = 0 then
	    Add(L, ((2*i + 1)*g1*n2 - (2*j + 1)*g2*n1) mod (4*n));
	  else
	    Add(L, ((2*i + 1)*g1*n2 + (2*j + 1)*g2*n1) mod (4*n));
	  fi;
	od;
      od;

      for i in [1..Length(L)] do
	if L[i] < 2*n then
	  L[i] := (L[i] - 1)/2;
	else
	  L[i] := (4*n - L[i] - 1)/2;
	fi;
      od;

      return PermList(L + 1);
    end;

    # second and third permutation
    dctperm2 := function ( n1, n2 )
      local n, K, L, i, j, q2;

      n  := n1*n2;

      L := [ ];
      K := [ ];

      # start with the border
      # i = 0
      for j in [0..n2 - 1] do
	Add(L, j);
	Add(K, j*n1);
      od;

      # j = 0
      for i in [1..n1 - 1] do
	Add(L, i*n2);
	Add(K, i*n2);
      od;

      # now the interior; two cases
      if n1 mod 2 = 0 then
	for i in [1..n1/2 - 1] do
	  for j in [1..n2 - 1] do
	    q2 := i*n2 + j*n1;
	    if q2 > n then
	      q2 := q2 - 2*n;
	      Add(L, (n1 - i)*n2 + (n2 - j));
	      Add(L, i*n2 + j);
	    else
	      Add(L, i*n2 + j);
	      Add(L, (n1 - i)*n2 + (n2 - j));
	    fi;
	    Add(K, AbsInt(i*n2 - j*n1));
	    Add(K, AbsInt(q2));
	  od;
	od;
	i := n1/2;
	for j in [1..(n2 - 1)/2] do
	  q2 := i*n2 + j*n1;
	  if q2 > n then
	    q2 := q2 - 2*n;
	    Add(L, (n1 - i)*n2 + (n2 - j));
	    Add(L, i*n2 + j);
	  else
	    Add(L, i*n2 + j);
	    Add(L, (n1 - i)*n2 + (n2 - j));
	  fi;
	  Add(K, AbsInt(i*n2 - j*n1));
	  Add(K, AbsInt(q2));
	od;
      else # n1 mod 2 <> 0
	for i in [1..(n1 - 1)/2] do
	  for j in [1..n2 - 1] do
	    q2 := i*n2 + j*n1;
	    if q2 > n then
	      q2 := q2 - 2*n;
	      Add(L, (n1 - i)*n2 + (n2 - j));
	      Add(L, i*n2 + j);
	    else
	      Add(L, i*n2 + j);
	      Add(L, (n1 - i)*n2 + (n2 - j));
	    fi;
	    Add(K, AbsInt(i*n2 - j*n1));
	    Add(K, AbsInt(q2));
	  od;
	od;
      fi;

      return [PermList(L + 1), PermList(K + 1)];
    end;

    # actual rule
    return
      Perm(dctperm2(n1, n2)[2]^-1, n) *
      DirectSum(
	I(n1 + n2 - 1),
	Tensor(I((n - n1 - n2 + 1)/2), F(2))
      ) *
      Perm(dctperm2(n1, n2)[1], n) *
      Tensor(C) *
      Perm(dctperm1(n1, n2), n);
  end
)
));
