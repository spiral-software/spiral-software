# -*- Mode: shell-script -*-
# Monomial Matrices, 
# SE, 30.-31.01.97, GAPv3.4

# After 1.2:
# - 18.06.01: bug in MonOps.\* removed affecting 
#   right cosets of monomial groups

#F "Mon" -- Monomial Matrices, Efficiently Represented
#F ===================================================
#F
#F The elements of the class Mon represent monomial matrices
#F over some field efficiently. A monomial matrix is a matrix
#F which contains exactly one non-zero entry in every row and
#F in every column. Hence, these matrices are of the form
#F 
#F   perm * diag([d_1, .., d_n])     for d_k <> 0*d_k.
#F
#F The MonOps class is derived from the GroupElementOps class.
#F The elements in Mon are records of the form
#F   rec(
#F     isMon          := true, 
#F     isGroupElement := true,
#F     domain         := GroupElements,
#F     char           := <the characteristics of the base field>
#F     perm           := <permutation; moves <= Length(diag) points>
#F     diag           := <non-empty list of non-zero field elements>
#F     operations     := MonOps
#F   ).
#F
#F The convention for permutations is chosen to match right 
#F application point^perm; (1,2,3) = [[0,1,0],[0,0,1],[1,0,0]]. 
#F The convention for monomial operations is perm*diag. Hence,
#F diag scales the columns of the perm-matrix.
#F

#if not IsBound(DirectSumPerm) then
#  ReadAREPFile("tools.g");
#fi;

MonOps := OperationsRecord("MonOps", GroupElementOps);

#F IsMon( <obj> )
#F   tests if <obj> is a Mon-object.
#F

IsMon := function ( obj )
  return IsRec(obj) and IsBound(obj.isMon) and obj.isMon;
end;

#F IsPermMon( <mon> )
#F IsDiagMon( <mon> )
#F   tests if a mon is a permutation resp. diagonal

IsPermMon := function ( mon )
  if not IsMon(mon) then
    Error("usage: IsPermMon( <mon> )");
  fi;
  return ForAll(mon.diag, x -> x = x^0);
end;

IsDiagMon := function ( mon )
  if not IsMon(mon) then
    Error("usage: IsPermMon( <mon> )");
  fi;
  return mon.perm = ();  
end;

#F Mon( <mon> )
#F Mon( <perm>, <diag> )
#F Mon( <diag>, <perm> )
#F Mon( <perm>, <deg> [, <char/field>] )
#F Mon( <diag> )
#F   constructs a Mon-object from some defining information.
#F   <char/field> defaults to 0. The non-empty list <diag> must 
#F   contain non-zero field elements. Note that perm*diag is
#F   not the same as diag*perm.
#F

# MonOps.CheckChar( <char/field> )
#   tests whether argument is a characteristic or 
#   a field and returns the characteristic

MonOps.CheckChar := function ( x )
  if x = 0 then
    return 0;
  elif IsInt(x) and x > 0 and IsPrime(x) then
    return x;
  elif IsField(x) then
    return x.char;
  else
    Error("<x> must be 0, positive prime, or field");
  fi;
end;


MonOps.MonNC := function ( char, perm, diag )
  return 
    rec(
      isMon          := true,
      isGroupElement := true,
      domain         := GroupElements,
      char           := char,
      perm           := perm,
      diag           := diag,
      operations     := MonOps
    );
end;

Mon := function ( arg )
  local char, deg, perm, diag, one;

  if Length(arg) = 1 and IsMon(arg[1]) then

    return arg[1];

  elif 
    Length(arg) = 2 and 
    IsPerm(arg[1]) and 
    IsList(arg[2]) and Length(arg[2]) >= 1
  then
    perm := arg[1];
    diag := arg[2];
    char := Characteristic( DefaultField(diag) );

  elif 
    Length(arg) = 2 and 
    IsList(arg[1]) and Length(arg[1]) >= 1 and
    IsPerm(arg[2])
  then
    perm := arg[2];
    diag := Permuted(arg[1], perm);
    char := Characteristic( DefaultField(diag) );

  elif 
    Length(arg) = 2 and 
    IsPerm(arg[1]) and 
    IsInt(arg[2]) and arg[2] >= 1
  then
    perm := arg[1];
    deg  := arg[2];
    diag := List([1..deg], i -> 1);
    char := 0;

  elif
    Length(arg) = 3 and 
    IsPerm(arg[1]) and 
    IsInt(arg[2]) and arg[2] >= 1 
  then
    perm := arg[1];
    deg  := arg[2];
    char := MonOps.CheckChar(arg[3]);
    if char = 0 then
      one := 1;
    else
      one := FiniteField(char).one;
    fi;
    diag := List([1..deg], i -> one);

  elif
    Length(arg) = 1 and
    IsList(arg[1]) and Length(arg[1]) >= 1
  then
    perm := ();
    diag := arg[1];
    char := Characteristic( DefaultField(diag) );

  else
    Error("wrong parameters, perhaps missing <deg>");
  fi;

  if not ( 
    perm = () or
    LargestMovedPointPerm(perm) <= Length(diag) 
  ) then
    Error("<perm> moves more points than in <diag>");
  fi;
  if not ForAll(diag, x -> x <> 0*x) then
    Error("<diag> must not contain zero");
  fi;

  return MonOps.MonNC(char, perm, diag);
end;

MonOps.PrintNC := function ( mon, indent, indentStep )
  local newline;

  newline := function ( )
    local i;

    Print("\n");
    for i in [1..indent] do
      Print(" ");
    od;
  end;

  if ForAll(mon.diag, x -> x = x^0) then
    if mon.char = 0 then

      # use Mon( <perm>, <deg> )
      Print(
        "Mon( ", 
        mon.perm, ", ", 
        Length(mon.diag), " )"
      );

    else

      # use Mon( <perm>, <deg>, <char> )
      Print(
        "Mon( ", 
        mon.perm, ", ", 
        Length(mon.diag), ", GF(", 
        mon.char, ") )"
      );

    fi;
  elif mon.perm = () then

    # use Mon( <diag> )
    Print("Mon( ", mon.diag, " )");

  else

    # use Mon( <perm>, <diag> )
    Print("Mon(");
    indent := indent + indentStep;
    newline();
      Print(mon.perm, ",");
      newline();
      Print(mon.diag);
    indent := indent - indentStep;
    newline();
    Print(")");

  fi;
end;

MonOps.Print := function ( mon )
  if not IsMon(mon) then
    Error("<mon> must be a Mon");
  fi;
  MonOps.PrintNC(mon, 0, 2);
end;


#F MatMon( <mon> )
#F MonMat( <mat> )
#F   convert between Mon-objects and the monomial matrices
#F   represented by the Mon-objects. The <mat> must be a 
#F   square matrix. If it is not monomial false is returned.
#F

MatMon := function ( mon )
  local mat, primeField, i, j;

  if mon = false then
    return false;
  fi;
  if not IsMon(mon) then
    Error("<mon> must be monomial operation");
  fi;
  if mon.char = 0 then
    primeField := Rationals;
  else
    primeField := FiniteField(mon.char);
  fi;
  mat := 
    Permuted(
      IdentityMat(
        Length(mon.diag), 
        primeField
      ),
      mon.perm^-1
    );
  for i in [1..Length(mon.diag)] do
    for j in [1..Length(mon.diag)] do
      mat[i][j] := mat[i][j] * mon.diag[j];
    od;
  od;
  return mat;
end;

MonMat := function ( mat )
  local deg, perm, diag,  i, j, k;

  if mat = false then
    return false;
  fi;
  if not ( 
    IsMat(mat) and Length(mat) = Length(mat[1]) 
  ) then
    Error("<mat> must be a square matrix");
  fi;
  deg := Length(mat);

  perm := [ ];
  diag := [ ];
  for i in [1..deg] do
    for j in [1..deg] do
      if mat[i][j] <> 0*mat[i][j] then
        if not IsBound(diag[j]) then
          perm[i] := j;
          diag[j] := mat[i][j];
        else
          return false;
        fi;
      fi;
    od;
  od;
  for k in [1..deg] do
    if not ( IsBound(diag[k]) and IsBound(perm[k]) ) then
      return false;
    fi;
  od;
  return Mon(PermList(perm), diag);
end;

#F PermMon( <mon> )
#F   convert a mon object to a permutation if possible, 
#F   if not, false is returned
#F

PermMon := function ( mon )
  if mon = false then
    return false;
  fi;
  if not IsMon(mon) then
    Error("usage: PermMon( <mon> )");
  fi;
  if ForAny(mon.diag, x -> x <> x^0) then
    return false;
  fi;
  return mon.perm;
end;

#F DegreeMon( <mon> )
#F CharacteristicMon( <mon> )
#F OrderMon( <mon> )
#F TransposedMon( <mon> )
#F DeterminantMon( <mon> )
#F TraceMon( <mon> )
#F   properties of a monomial matrix.
#F

DegreeMon := function ( mon )
  if not IsMon(mon) then
    Error("<mon> must be a monomial operation");
  fi;
  return Length(mon.diag);
end;

CharacteristicMon := function ( mon )
  if not IsMon(mon) then
    Error("<mon> must be a monomial operation");
  fi;
  return mon.char;
end;

OrderMon := function ( mon )
  local rperm, rdiag, rd, d;

  if not IsMon(mon) then
    Error("<mon> must be a monomial operation");
  fi;
  
  rperm := OrderPerm(mon.perm);
  mon   := mon^rperm;
  if mon.char = 0 then
    rdiag := 1;
    for d in mon.diag do
      rd := OrderCyc(d);
      if rd = "infinity" then
        return rd;
      fi;
      rdiag := LcmInt(rdiag, rd);
    od;
  else
    rdiag := 1;    
    for d in mon.diag do
      rdiag := LcmInt(rdiag, OrderFFE(d));
    od;
  fi;
  return rperm * rdiag;
end;

TransposedMon := function ( mon )
  if not IsMon(mon) then
    Error("<mon> must be a monomial operation");
  fi;
  return Mon(mon.diag, mon.perm^-1);
end;

DeterminantMon := function ( mon )
  if not IsMon(mon) then
    Error("<mon> must be a monomial operation");
  fi;
  return SignPerm(mon.perm) * Product(mon.diag);
end;

TraceMon := function ( mon )
  local tr, k;

  if not IsMon(mon) then
    Error("<mon> must be a monomial operation");
  fi;

  if mon.char = 0 then
    tr := 0;
  else
    tr := FiniteField(mon.char).zero;
  fi;
  for k in [1..Length(mon.diag)] do
    if k^mon.perm = k then
      tr := tr + mon.diag[k];
    fi;
  od;
  return tr;
end;

#F GaloisMon( <mon>, <int/fieldAut> )
#F   a Galois conjugate of <mon>. The second argument
#F   is either a field automorphism or an integer k
#F   specifying the automorphism x -> GaloisCyc(x, k) if
#F   char = 0 or x -> x^(FrobeniusAut^k) if char is prime.
#F

GaloisMon := function ( mon, galoisAut )
  local diag, x;

  if not IsMon(mon) then
    Error("<mon> must be a Mon");
  fi;
  if IsInt(galoisAut) then

    if mon.char = 0 then

      # use GaloisCyc
      return 
        Mon(
          mon.perm, 
          List(mon.diag, x -> GaloisCyc(x, galoisAut))
        );

    else
 
      # use Frobenius
      return
        Mon(
          mon.perm,
          List(mon.diag, x -> x^(mon.char^galoisAut))
        );

    fi;

  elif IsMapping(galoisAut) then

    if not IsField(galoisAut.source) then
      Error("<galoisAut> must be a field automorphism");
    fi;
    diag := [ ];
    for x in mon.diag do
      if not x in galoisAut.source then
        Error("<x> must be in <galoisAut>.source");
      fi;
      Add(diag, x ^ galoisAut);
    od;
    return Mon(mon.perm, diag);

  else
    Error("usage: GaloisMon( <mon>, <int/fieldAut> )");
  fi;
end;

#F DirectSumMon( <mon1>, .., <monN> ) ; N >= 1
#F DirectSumMon( <list-of-mon> )
#F   forms the direct sum of the given Mons. There must be
#F   at least one summand and all summands must have a common
#F   characteristic of the base field. Note that the leftmost
#F   summand is in the upper left corner of the matrix.
#F

DirectSumMon := function( arg )
  local diag, m;

  if IsList(arg[1]) then
    arg := arg[1];
  fi;
  if not ( Length(arg) > 0 and ForAll(arg, IsMon) ) then
    Error(
      "usage:\n",
      "  DirectSumMon( <mon1>, .., <monN> ) ; N >= 1\n",
      "  DirectSum( <list-of-mon> )"
    );
  fi;
  if not ForAll(arg, m -> m.char = arg[1].char) then
    Error("sorry, Mons must have common characteristic");
  fi;

  # construct result
  diag := [ ];
  for m in arg do
    Append(diag, m.diag);
  od;
  return 
    Mon(
      DirectSumPerm(
        List(arg, m -> Length(m.diag)),
        List(arg, m -> m.perm)
      ),
      diag
   );
end;

#F TensorProductMon( <mon1>, .., <monN> ) ; N >= 1
#F TensorProductMon( <list-of-mon> )
#F   forms the tensor product of the given Mons. There must be
#F   at least one factor and all factors must have a common
#F   characteristic of the base field. Note that the leftmost
#F   factor determines the coarsest structure.
#F

TensorProductMon := function( arg )
  local diag, i;

  if IsList(arg[1]) then
    arg := arg[1];
  fi;
  if not( Length(arg) > 0 and ForAll(arg, IsMon) ) then
    Error(
      "usage:\n",
      "  TensorProductMon( <mon1>, .., <monN> ) ; N >= 1\n",
      "  TensorProductMon( <list-of-mon> )"
    );
  fi;
  if not ForAll(arg, m -> m.char = arg[1].char) then
    Error("sorry, Mons must have common characteristic");
  fi;

  # construct result
  diag := arg[1].diag;
  for i in [2..Length(arg)] do

    # diag := diag (x) arg[i].diag
    diag :=
      Concatenation(
        List(
          diag,
          d -> d * arg[i].diag
        )
      );
  od;

  return 
    Mon(
      TensorProductPerm(
        List(arg, m -> Length(m.diag)),
        List(arg, m -> m.perm)
      ),
      diag
   );
end;

#F <mon1> =  <mon2>
#F <mon1> <= <mon2>
#F <mon1> <  <mon2>
#F <mon1> <> <mon2>
#F <mon1> >  <mon2>
#F <mon1> >= <mon2>
#F   compare two objects representing a monomial matrix.
#F   The total order defined on the set of all Mon-objects
#F   is defined as the order on the [perm, diag] pairs.
#F

MonOps.\= := function ( mon1, mon2 )
  return
    IsMon(mon1) and
    IsMon(mon2) and
    mon1.perm = mon2.perm and 
    mon1.diag = mon2.diag;
end;

MonOps.\< := function ( mon1, mon2 )
  return
    [mon1.perm, mon1.diag] <
    [mon2.perm, mon2.diag];
end;


#F <mon1> * <mon2>
#F <mon1> / <mon2>
#F <mon1> ^ <mon2>
#F <mon> ^ <int>
#F [<int>, <fldelm>] ^ <mon>
#F Comm( <mon1>, <mon2> )
#F LeftQuotient( <mon1>, <mon2> )
#F   fundamental arithmetics with the monomial matrices. 
#F   The degree of the result is the maximum of the degrees
#F   of the operands. Note that ^ is the natural action of the
#F   monomial matrices on Cartesian([1..<degree>], <field>).
#F

MonOps.ProductNC := function ( mon1, mon2 )
  local one, deg, diag, k;

  if mon1.char = 0 then
    one := 1;
  else
    one := FiniteField(mon1.char).one;
  fi;
  deg := Maximum(Length(mon1.diag), Length(mon2.diag));

  diag := List([1..deg], k -> one);
  for k in [1..Length(mon1.diag)] do
    diag[k^mon2.perm] := mon1.diag[k];
  od;
  for k in [1..Length(mon2.diag)] do
    diag[k] := diag[k] * mon2.diag[k];
  od;

  return
    MonOps.MonNC(
      mon1.char,
      mon1.perm * mon2.perm,
      diag
    );
end;

MonOps.InverseNC := function ( mon )
  return
    Mon(
      mon.perm^-1,
      List(
        [1..Length(mon.diag)], 
        k -> 1/mon.diag[k^mon.perm]
      )
    );
end;

MonOps.PowerNC := function ( mon, exp )
  local mon1;

  if exp < 0 then
    return MonOps.PowerNC(MonOps.InverseNC(mon), -exp);
  elif exp = 0 then
    return Mon((), Length(mon.diag), mon.char);
  elif exp = 1 then
    return mon;
  else
    mon1 := MonOps.PowerNC(mon, QuoInt(exp, 2));
    if RemInt(exp, 2) = 1 then
      return mon1*mon1 * mon;
    else
      return mon1*mon1;
    fi;
  fi;
end;

MonOps.ScalarMultiple := function ( scalar, mon )

  if not IsMon(mon) then
    Error("usage: MonOps.ScalarMultiple( <scalar>, <mon> )");
  fi;
  if not DefaultField(scalar).char = mon.char then
    Error("<scalar> and <mon> must have the same char");
  fi;

  return Mon(mon.perm, scalar*mon.diag);  
end;

MonOps.\* := function ( mon1, mon2 )
  local one, deg, diag, k;

  if IsList(mon1) and IsMon(mon2) then
    return List(mon1, m -> m*mon2);
  elif IsMon(mon1) and IsList(mon2) then
    return List(mon2, m -> mon1*m);
  elif IsGroup(mon1) and IsMon(mon2) then
    return GroupOps.\*(mon1, mon2);
  elif IsRightCoset(mon1) and IsMon(mon2) then
    return RightCosetGroupOps.\*(mon1, mon2);
  elif IsMon(mon1) and IsMon(mon2) then
    if not mon1.char = mon2.char then
      Error("Sorry, <mon1> and <mon2> are of different char");
    fi;
    return MonOps.ProductNC(mon1, mon2);
  elif IsMon(mon2) then
    return MonOps.ScalarMultiple(mon1, mon2);
  elif IsMon(mon1) then
    return MonOps.ScalarMultiple(mon2, mon1);
  else
    Error("sorry, cannot compute <mon1> * <mon2>");
  fi;
end;

MonOps.\^ := function ( mon, exp )
  if IsMon(mon) and IsMon(exp) then
    return exp^-1 * mon * exp;
  elif IsMon(mon) and IsInt(exp) then
    return MonOps.PowerNC(mon, exp);
  elif IsList(mon) and Length(mon) = 2 and IsMon(exp) then
    if not ( 
      IsInt(mon[1]) and mon[1] in [1..Length(exp.diag)] and
      mon[2] in FieldElements    
    ) then
      Error("point must be [<pos-int>, <fieldelm>]");
    fi;
    return 
      [ mon[1]^exp.perm, 
        mon[2] * exp.diag[mon[1]^exp.perm] 
      ];
  else
    Error("wrong arguments for ^");
  fi;
end;


#F CharPolyCyclesMon( <mon> )
#F   the sorted list of characteristic polynomials of the 
#F   cycles of <mon>. The polynomials are of the form
#F   X(<field>)^<cycle-length> - <constant> where the
#F   <field> is the same for all polynomials. (This makes
#F   it possible to apply Product without pain.)
#F

CharPolyCyclesMon := function ( mon )
  local X, cycles;

  if not IsMon(mon) then
    Error("<mon> must be a monomial operation");
  fi;

  # A cycle of a monomial matrix perm * diag is of the
  # form (1, .., |c|) * Sublist(diag, c) where c is a 
  # cycle of perm of the form 
  #   c = [c[1], c[1]^perm, c[1]^(perm^2), ..].
  #
  # The char. polynomial of a monomial cycle is
  #   f(X; (1, .., n) * diag) = X^n - Product(diag).
  # (This can be seen by expanding the determinant
  # by the first column.)

  X      := Indeterminate(DefaultField(mon.diag));
  cycles :=
    List(
      Cycles(mon.perm, [1..Length(mon.diag)]),
      c -> X^Length(c) - Product(Sublist(mon.diag, c))
    );
  Sort(cycles);
  return cycles;
end;

# ...in Arbeit

MonOps.OrbitMon := function ( group, point )
  local orbit, recent, p, g, pg;

  orbit  := [ point ];
  recent := [ point ];
  while Length(recent) > 0 do
    p := recent[Length(recent)];
    Unbind(recent[Length(recent)]);
    for g in group.generators do
      pg := p^g;
      if not pg in orbit then
        AddSet(orbit, pg);
        Add(recent, pg);
      fi;
    od;
  od;
  return orbit;
end;

# Beispiel:
#
# m1 := Mon( 
#   ( 1,20, 2,18, 4,17, 7,16, 8, 6)( 3,10,12,11, 9,14,19,13)( 5,15), 
#   [ Z(11), Z(11)^8, Z(11)^6, Z(11)^7, Z(11)^5, Z(11)^5, Z(11)^7, Z(11)^9, 
#     Z(11)^4, Z(11)^5, Z(11)^9, Z(11)^8, Z(11)^6, Z(11)^2, Z(11)^4, Z(11)^8, 
#     Z(11)^5, Z(11)^0, Z(11)^7, Z(11)^3 ] );
#
# Group(m1) hat auf [1..20] x Units(GF(11)) genau drei Orbits
# der L"angen 20, 80 und 100.
# 
# Frage1: Wie kann man zu einer Menge von <mon>-Erzeugern ein 
#         m"oglichst kurzes Orbit finden?
#         (-> Gut f"ur Stabilisatorkette)
#
# Frage2: Wie multipliziert man Gruppenelemente die durch ihre
#         Baseimages repr"asentiert werden ohne in die regul"are
#         Darstellung r"uberzugehen?
#

# MonOps.NaturalDomain := function ( deg, field )
#   return
#     Cartesian(
#       [1..deg], 
#       Filtered(Elements(field), x -> x <> 0*x)
#     );
# end;
