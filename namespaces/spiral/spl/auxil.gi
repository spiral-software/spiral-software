
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Auxiliary Functions for Symbols and Non-Terminals
# =================================================
# MP, from 08/17/00


#F Stride Permutations
#F -------------------
#F

# Let K a field and d | n. Then the stride permutation L_d^n is a
# permutation of degree n defined by
#   L_d^n (e_i tensor e_j) = (e_j tensor e_i)
#   for all i, j
# where e_i is the ith canonical base vector in K^d
# and e_j the jth in K^(n/d).
# Hence L_d^n exchanges the tensor components of dimension
# d and n/d resp.
# L_d^n can also be viewed the following way. Let x be 
# a (d x n/d)-matrix stored in row major order. L_d^n permutes
# x such that it is stored in column major order 
# (i.e. transposes x).

# Lemma: L_d^n is given as a permutation of 0..n-1 by
#   i -> i * d mod (n - 1), 0 <= i < n-1.
#
# Lemma: The following computation rules hold:
#   a) L_r^rst * L_s^rst = L_rs^rst.
#   b) L_t^rst = (L_t^rt tensor 1_s)(1_r tensor L_t^st).
#
# Lemma: L_2^(2i) has exactly 2 fixpoints.
#
# Lemma: (has to be proven)
#   p prime: ord(L_(p^i)^(p^n)) = n/gcd(n, i).


#F StridePerm( <n>, <d> )
#F   returns the stride permutation L_d^n (see above)
#F

StridePerm := function ( n, d )
  local M;

  # check arguments
  if not ( IsInt(n) and IsInt(d) and n > 0 and d > 0 ) then
    Error("<n> and <d> must be positive integers");
  fi;
  if not n mod d = 0 then
    Error("<d> must divide <n>");
  fi;

  # create matrix and transpose
  M := List([1..d], i -> [(i-1)*n/d+1..i*n/d]);
  
  return 
    MappingPermListList(
      Concatenation(TransposedMat(M)),
      [1..n]
    );
end;

# Computes the same thing, but slightly slower.

StridePerm1 := function ( n, d )
  local L;

  # check arguments
  if not ( IsInt(n) and IsInt(d) and n > 0 and d > 0 ) then
    Error("<n> and <d> must be positive integers");
  fi;
  if not n mod d = 0 then
    Error("<d> must divide <n>");
  fi;

  # compute images according to lemma above
  L    := List([1..n], i -> (i - 1) * d mod (n - 1) + 1);
  L[n] := n;

  # return permutation
  return MappingPermListList([1..n], L);
end;

# CycleTypeStridePerm( <n>, <d> )
#   computes the cycle type of L_d^n. The cycle type is a
#   list of pairs [ .., [c_i, n_i], .. ], which denotes
#   that the permutation contains exactly n_i cycles of
#   length c_i. 
#   If c_i = 1, then n_i is the number of fixpoints.
#

CycleTypeStridePerm := function ( n, d )
  return Collected(CycleLengths(StridePerm(n, d), [1..n]));
end;


#F Twiddle Matrices
#F ----------------
#F

# The twiddle factors appear in the Cooley-Tukey FFT, namely,
# for d|n we have:
#   F_n = (F_n/d tensor I_d) * T(n, d) * (I_n/d tensor F_d) * L(n, n/d)
# where F_n is a DFT of size n, and L denotes a stride permutation
# defined as above


#F TwiddleDiag( <n>, <d> [, <k> ] )
#F TwiddleDiag( <list-of-n-d[-k]> )
#F   returns the twiddle factors for <d>|<n> as a list:
#F     T(n, d) = direct_sum_(i = 1)^n/d diag(w_n^0, ..., w_n^(d-1)^i
#F   If the optional parameter <k> is given then w_n is replaced by 
#F   w_n^k, gcd(<n>, <k>) = 1.
#F

TwiddleDiag := function ( arg )
  local n, d, k;

  if Length(arg) = 1 and IsList(arg[1]) then
    arg := arg[1];
  fi;

  if Length(arg) = 2 then
    n := arg[1];
    d := arg[2];
    k := 1;
  elif Length(arg) = 3 then
    n := arg[1];
    d := arg[2];
    k := arg[3];
  else
    Error(
      "usage:\n",
      "  TwiddleDiag( <n>, <d> [, <k> ] )\n",
      "  TwiddleDiag( <list-of-n-d[-k]> )"
    );
  fi;

  # check arguments
  if not ( IsInt(n) and IsInt(d) and n > 0 and d > 0 ) then
    Error("<n> and <d> must be positive integers");
  fi;
  if not n mod d = 0 then
    Error("<d> must divide <n>");
  fi;
  if not Gcd(n, k) = 1 then
    Error("gcd(<n>, <k>) must be 1");
  fi;
  
  return
    Concatenation(
      List([1..n/d], i -> List([1..d], j -> E(n)^(k*(j-1)*(i-1))))
    );
end;
