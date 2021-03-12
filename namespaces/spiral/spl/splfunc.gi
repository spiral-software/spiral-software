
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Recursive data structure for SPL programs
# =========================================
# needs AREP, approx.g, aux.g, scalar.g, optrec.g

#F SPLs
#F ====
#F
#F *** WARNING ***
#F *** INFORMATION BELOW IS OBSOLETE ***
#F
#F This file contains a recursive data structure to
#F represent SPL programs in GAP. The data structure is 
#F also named SPL. An spl is a GAP record with certain
#F fields as explained below.
#F There are two main applications of spl:
#F 1. They are used by the rule-based formula generator to
#F    generate SPL programs within GAP.
#F 2. As an interface between amats (cf. amat.g in AREP) and
#F    SPL programs.
#F
#F We define a SPL object (SPL) recursively in BNF as
#F the disjoint union of the following cases
#F
#F <SPL> ::= 
#F ; atomic cases
#F   | <mat>                                 ; "mat"
#F   | <sparse>                              ; "sparse"
#F   | <mon>                                 ; "mon"  (invertible)
#F   | <diag>                                ; "diag" 
#F   | <perm>                                ; "perm" (invertible)
#F 
#F ; parametrized symbols
#F   | Symbol(<name>, <params>)              ; "symbol" 
#F
#F ; non-terminal symbols
#F   | NonTerminal(<name>, <params>)         ; "nonTerminal"
#F
#F ; composed cases
#F   | <SPL> * .. * <SPL>                    ; "compose"
#F   | DirectSum(<SPL>, .., <SPL>)           ; "directSum"
#F   | TensorProduct(<SPL>, .., <SPL>)       ; "tensor"
#F   | <scalar> * <SPL>                      ; "scalarMultiple"
#F   | <SPL> ^ <permSPL>                     ; "conjugate"
#F   | <SPL> ^ <positive-int>                ; "power"
#F
#F The design of SPL is similar to that of AMat with the
#F exception of the types "symbol" and "nonTerminal".
#F The type "symbol" has been introduced to represent parametrized
#F matrices in an efficient way. E.g., the Fourier transform F_n of 
#F size n is presented as
#F   SPLSymbol( "F", [ 3 ] )  and in SPL as  (F 3)
#F without any matrix being stored. The symbol should also be part of SPL
#F (the language). The symbols are contained in the Symboltable (below) 
#F together with symbol-specific functions. The symboltable is meant to be 
#F extended.
#F
#F Note: you can use scalars to build spls (see scalar.g) but currently
#F only for "scalarMultiple" and "diag".
#F
#F An SPL is a GAP-Rec T with the following mandatory 
#F fields common to all types of SPL
#F
#F   isSPL        := true, identifies spls
#F   operations   := SPLOps, GAP operations record for T
#F   type         : a string identifying the type of T
#F   dimensions   : size of the matrix represented (= [rows, columns])
#F
#F The characteristic of the base field is always = 0.
#F
#F The following fields are optional for any type of spl S:
#F
#F   splOptions: a list of strings supplying options for the spl compiler
#F     "unrolled": specifies that S is to be unrolled
#F     when exported to spl. In this case this option is disregarded for 
#F     all subtrees (if an spl is to be unrolled, then all its subtrees have
#F     to be unrolled).
#F     "pcl": a pcl label should be set in the spl program
#F   pclLabel: a label for pcl profiling
#F   root: contains the spl that has been expanded to obtain S.
#F
#F The following fields are mandatory to the type of T (T.type):
#F
#F type = "mat"
#F   T.element: matrix represented
#F type = "sparse"
#F   T.element: list of triples [i, j, a_(i,j)]
#F type = "mon"
#F   T.element: mon represented
#F type = "diag"
#F   T.element: list containing the diagonal
#F type = "perm"
#F   T.element: perm represented
#F type = "symbol"
#F   T.symbol: string representing the symbol
#F   T.params: list of parameters for the symbol
#F type = "nonTerminal"
#F   T.symbol: string representing the symbol
#F   T.isRealTransform: true if the transform is real, false else; 
#F     a transform is real iff the matrix is real for all choices of parameters
#F   T.params: list of parameters for the symbol
#F   T.index: the index of the non-terminal in NonTerminalTableSPL
#F   T.ruleinds: the indices of applicable rules from RuleTable
#F type = "compose"
#F   T.children(): list of spls, the factors
#F type = "directSum"
#F   T.children(): list of spls, the summands
#F type = "tensor"
#F   T.children(): list of spls, the tensor factors
#F type = "scalarMultiple"
#F   T.scalar: the scalar
#F   T.element: the SPL multiplied
#F type = "conjugate"
#F   T.element: the SPL to be conjugated
#F   T.conjugation: the conjugating SPL
#F type = "power"
#F   T.element: the SPL raised to the exponent
#F   T.exponent: the exponent, an integer > 0
#F
#F *** WARNING ***
#F *** INFORMATION ABOVE IS OBSOLETE ***

# Internal Functions for SPLs
# ----------------------------
#

# the operations record
SPLOps := CantCopy(OperationsRecord("SPLOps"));

# SPLOps.IsDegree( <perm>, <degree> )
#   tests whether <degree> is a degree consistent with <perm>

SPLOps.IsDegree := function ( p, n )
  if not IsPerm(p) then
    Error("usage: SPLOps.IsDegree( <perm>, <degree> )");
  fi;
  return 
    IsInt(n) and 
    n >= 0 and 
    ( p = ( ) or n >= LargestMovedPointPerm(p) );
end;

# SPLOps.IsDimension( <dims/deg> )
#   tests whether <dims/deg> is a valid dimension for a 
#   matrix, i.e. a positive integer or a pair of them and
#   returns a pair representing the dimension.

SPLOps.IsDimension := function ( D )
  if IsList(D) then
    if not ( 
      Length(D) = 2 and 
      ForAll(D, n -> IsInt(n) and n > 0) 
    ) then
      Error("<D> must be a list of two positive ints");
    fi;
    return D;
  else
    if not ( IsInt(D) and D > 0 ) then
      Error("<D> must be a positive int");
    fi;
    return [D, D];
  fi;
end;


#F Fundamental Constructors and Tests for SPLs
#F --------------------------------------------
#F

#F IsSPL( <obj> ) 
#F   tests whether <obj> is an SPL. The particular case of SPL
#F   is provided by SPL.type and the additional fields are available
#F   directly from the record.
#F
IsSPL := S -> IsRec(S) and IsBound(S.isSPL) and S.isSPL;

#F IsPermutationSPL( <spl> )
#F   returns true, if <spl> is of type "perm" or a symbol that
#F   represents a permutation.
#F
IsPermutationSPL := function ( S )
  if not IsSPL(S) then
      Error("<S> must be an spl");
  elif IsBound(S.isPermutation) then
      return S.isPermutation();
  else 
      return false;
  fi;
end;

#F Comparison of SPLs
#F ------------------
#F
 
#F SPLOps.\=( <spl1>, <spl2> )
#F   returns true if <spl1> and <spl2> are structurally equal.
#F
SPLOps.\= := (S1, S2) -> Cond(IsRec(S1) and IsBound(S1.equals), S1.equals(S2), false);
# NOTE: this function will not yield a good order is Set(..) is used
SPLOps.\< := (S1, S2) -> Cond(IsRec(S1) and IsBound(S1.lessThan), S1.lessThan(S2), BagAddr(S1) < BagAddr(S2));

#F IsIdenticalSPL ( <spl1>, <spl2> )
#F    returns true if <spl1> and <spl2> are spls with identical
#F    syntax trees.
#F
IsIdenticalSPL := (S1, S2) ->
   Checked(IsSPL(S1), IsSPL(S2), 
       Cond(S1.dimensions <> S2.dimensions, false,
	    ObjId(S1) <> ObjId(S2), false,
	    S1.equals(S2)));

IsEqualObj := (o1,o2) -> Cond(
    Same(o1,o2), true, 
    not IsRec(o1) or not IsBound(o1.equals), o1=o2,
    o1.equals(o2)
);

#F IsHashIdenticalSPL ( <spl1>, <spl2> )
#F    returns true if <spl1> and <spl2> are identical for the
#F    hashing purposes (see spiral/search). 
#F

IsHashIdenticalSPL := function( S1, S2 )
    if not ( IsSPL(S1) and IsSPL(S2) ) then
	Error("usage: IsHashIdenticalSPL ( <spl1>, <spl2> )");
    fi;

    if IsBound(S1.hashId) and IsBound(S2.hashId) then
       return Same(ObjId(S1), ObjId(S2)) and
              S1.dimensions = S2.dimensions and
              S1.transposed = S2.transposed and
              S1.hashId = S2.hashId;
    elif IsBound(S1.HashId) and IsBound(S2.HashId) then #NOTE: IS THAT RIGHT???
       return Same(ObjId(S1), ObjId(S2)) and
              When(ForAll(Concat(S1.dimensions, S2.dimensions), IsInt), S1.dimensions = S2.dimensions, true) and
              S1.transposed = S2.transposed and
              S1.HashId() = S2.HashId();
    else 
       return IsIdenticalSPL( S1, S2 );
    fi;
end;

#F Converting SPLs
#F ---------------

#F MatSparseSPL ( <"sparse"-spl> )
#F   returns the matrix represented by the spl of type "sparse".
#F
MatSparseSPL := function ( S )
  local M, t;
  #Constraint(ObjId(S) = Sparse);
  M := NullMat(S.dimensions[1], S.dimensions[2]);
  for t in S.element do
    M[t[1]][t[2]] := t[3];
  od;

  return M;
end;



#F AMatSPL( <SPL> )
#F   converts <SPL> into an amat.
#F
AMatSPL := S -> S.toAMat();

#F MatSPL( <spl> )
#F   returns the matrix represented by <spl>.
#F
MatSPL := S -> MatAMat(AMatSPL(S));

#F PermSPL( <spl> )
#F   returns the permutation represented by <spl>,
#F   or false if <spl> is no permutation.
#F
PermSPL := S -> PermAMat(AMatSPL(S));


#F Other Functions
#F ---------------
#F

#F TerminateSPL( <spl> )
#F   returns an spl equivalent to <spl> by replacing
#F   all non-terminals with terminal spls.
#F
TerminateSPL := S -> S.terminate();


#F TransposedSPL ( <SPL> )
#F   returns an spl representing the transpose of <SPL>.
#F
TransposedSPL := S -> S.transpose();


# ArithmeticCostScalarComplex ( <scalar> )
#   returns the arithmetic cost of multiplying <scalar>
#   to a complex number in the form
#     [adds, mults, sign-changes]
#   <scalar> can be a scalar (see scalar.g) or a number.
#
ArithmeticCostScalarComplex := function ( s )
  local n;  
  # convert scalar into number (scalar.g)
  s := EvalScalar(s);

  if s = 1 then     return [0, 0, 0];
  elif s = -1 then  return [0, 0, 2];
  # s is real
  elif GaloisCyc(s, -1) = s then return [0, 2, 0];
  else
    n := NofCyc(s);

    # determine exponent of s as root of unity
    if n mod 2 = 1 and s ^ n = -1 then
      n := 2 * n;
    fi;

    # s is no root of unity
    if s^n <> 1 then
      return [2, 4, 0];
    fi;

    # special roots of unity
    if n = 4 then  return [0, 0, 1];
    elif n = 8 then return [2, 2, 0];
    else return [2, 4, 0];
    fi;
  fi;
end;

IsRealNumber := function (num) 
   if IsValue(num) then num := num.v; fi;

   if IsSymbolic(num) then 
       return When(IsVecT(num.t) or IsArrayT(num.t), not IsComplexT(num.t.t), not IsComplexT(num.t));
   elif IsComplex(num) then return ImComplex(num)=0; 
   elif IsDouble(num) then return true;
   elif IsCyc(num) then return GaloisCyc(num, -1) = num;
   elif IsList(num) then return ForAll(num, IsRealNumber);
   else return true;
   fi;
end;

#F IsRealSPL( <spl> )
#F   returns true if <spl> contains no complex subexpressions
#F   and false else. Note that this function does not determine whether
#F   <terminal spl> represents a real matrix! E.g.,
#F     E(4) * Mat([[-E(4)]])
#F   is a real matrix, but contains complex expressions. The result is
#F   stored in .isReal
#F   If the field nonTerminal is set in <spl> then true is returned iff
#F   the nonTerminal is a real transform. This assumes that a real transform
#F   is never expanded using complex expressions!
#F   The function is used to determine whether the compiler option -R
#F   should be used on exporting to spl.
#F   
IsRealSPL     := S -> S.isReal();

IsIdentity    := S -> IsBound(S.isIdentity) and S.isIdentity();
IsIdentitySPL := S -> IsSPL(S) and IsIdentity(S);
IsIdentityObj := S -> IsBound(S.isIdentity) and S.isIdentity = True;
