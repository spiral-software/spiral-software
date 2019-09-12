# -*- Mode: shell-script -*-
# Higher Functions for Constructive Representation Theory
# based on the class ARep, MP, 26.05.97 - , GAPv3.4

# Nach Submission (AREP 1.1)
# --------------------------
# 21.07.99: In kbsDecompositionARep den hint "hom" eingefuegt
#           beim Aufruf von RestrictionToSubmoduleARep
# 22.07.99: Funktionen FindSparseInvertible und FindRandomInvertible
#           geschrieben. Damit sind die Matrizen aus dem
#           Intertwiningraum von Fall 1 der Switch recursion
#           schoener und haben sogar Struktur in vielen Faellen.
# 23.07.99: Switch recursion nach neuesten Erkenntnissen implementiert.
#           DecompositionMonRep macht jetzt zuerst eine treue
#           Darstellung (einer ag Gruppe mit minimalen Erzeugendensystem)
#           und macht mit dieser weiter.
# 24.07.99: Zerlegung abelscher Darstellungen in ein aeusseres
#           Tensorprodukt (im treuen permrep Fall) geht jetzt ohne 
#           die Normalteiler zu berechnen.
# 27.07.99: Beim Zerlegen wird jetzt auf treue Darstellungen reduziert.
#           DecompositionMonRep ist jetzt neu strukturiert. Der primitive
#           und 2fach transitive Fall ist erstmal rausgeworfen.
# 01.08.99: Abelschen permrep-Fall besser gemacht. Es werden nicht mehr
#           die Charaktere des Normalteilers ausgewalzt.
# 05.08.99: In induction recursion wird Minkwitz Erweiterung jetzt nur noch
#           f"ur nicht monomiale F"alle ben"otigt. Machts aber nicht schneller,
#           aber anscheinend auch nicht langsamer.
  
#F Constructive Representation Theory
#F ==================================
#F
#F Based on the class ARep a number of functions is
#F provided to calculate with representations. 
#F The idea is to calculate with representations
#F up to equality not only up to equivalence.
#F Most functions deal with permutation and monomial
#F representations and are decompositions in a sense.
#F 

# Auxiliary Functions
# -------------------

# RootOfRootOfUnity( <root-of-unity>, <prime> )
#   computes a <prime>th root of a cyclotomic, 
#   which is a root of unity. The root is chosen
#   in the smallest extending field.

RootOfRootOfUnity := function ( c, p )
  local n, k;

  if not IsCyc(c) and IsPrime(p) and p > 0 then
    Error("<c> must be a cyc, <p> a positive prime");
  fi;
  
  # determine the exponent of c,
  # i.e. the smallest n >= 1 with c ^ n = 1
  n := NofCyc(c);
  if n mod 2 = 1 and c ^ n = -1 then
    n := 2 * n;
  fi;
    
  if n mod p <> 0 then

    # (c ^ (1/p mod n)) ^ p = c,
    # the field does not have to be extended
    return c ^ (1/p mod n);
  fi;

  # determine, for which k: c = E(n) ^ k,
  # then use (E(np) ^ k) ^ p = E(n) ^ k = c
  k := 1;
  while c <> E(n) ^ k do
    k := k + 1;
  od;

  return E(n * p) ^ k;
end;


# RootOfUnity( <n> [, <char>] )
#   returns a primitive <n>th root of unity of 
#   the given characteristic <char>. The default
#   characteristic is 0.

RootOfUnity := function( arg )
  local n, char;

  if Length(arg) = 1 then
    n    := arg[1];
    char := 0;
  elif Length(arg) = 2 then
    n    := arg[1];
    char := arg[2];
  else
    Error("usage: RootOfUnity( <n> [, <char>] )");
  fi;

  if not( IsInt(n) and n >= 1 ) then
    Error("<n> must be a positive integer");
  fi;
  char := AMatOps.CheckChar(char);

  if char = 0 then
    return E(n);
  fi;
  
  return "noch nicht implementiert";
end;


#F Functions for Characters
#F ------------------------
#F

#F IsRestrictedCharacter( <chi>, <chisub> )
#F   tests if <chisub> is a restriction of <chi>.
#F

IsRestrictedCharacter := function ( chiG, chiH )
  local ccs;

  if not ( IsCharacter(chiG) and IsCharacter(chiH) ) then
    Error("<chiG>, <chiH> must be characters");
  fi;

  ccs := chiH.source.conjugacyClasses;
  return
    ForAll(
      [1..Length(ccs)],
      i -> ccs[i].representative ^ chiG = chiH.values[i]
    );
end;


#F AllExtendingCharacters( <chi>, <supergrp> )
#F   calculates the list of all irreducible characters 
#F   of <supergrp> whose restriction is the irreducible 
#F   character <chi>.
#F

AllExtendingCharacters := function ( chi, G )
  local ccs, subgrp;

  if not( IsCharacter(chi) and IsGroup(G) ) then
    Error("usage: AllExtendingCharacters( <character>, <supergroup> )");
  fi;
  if not ForAll(chi.source.generators, g -> g in G) then
    Error("<G> must contain <chi>.source");
  fi;
  if not IsIrreducible(chi) then
    Error("<G> must be irreducible");
  fi;

  return 
    Filtered(
      Irr(G),
      chiG -> IsRestrictedCharacter(chiG, chi)
    );
end;


#F OneExtendingCharacter( <character>, <supergroup> )
#F   calculates a character of <supergroup> extending
#F   <character> or returns false.
#F

OneExtendingCharacter := function( chi, G )
  local pos;

  if not( IsCharacter(chi) and IsGroup(G) ) then
    Error("usage: OneExtendingCharacter( <character>, <supergroup> )");
  fi;
  if not ForAll(chi.source.generators, g -> g in G) then
    Error("<G> must contain <chi>.source");
  fi;
  if not IsIrreducible(chi) then
    Error("<G> must be irreducible");
  fi;

  pos :=
    PositionProperty( 
      Irr(G),
      chiG -> IsRestrictedCharacter(chiG, chi)
    );

  if pos = false then
    return false;    
  fi;
  return Irr(G)[pos];
end;


#F Intertwining Space of Representations
#F -------------------------------------
#F

#F IntertwiningSpaceARep( <arep1>, <arep2> )
#F   computes a base of the intertwining space 
#F     Int( <arep1>, <arep2> ) = { M | <arep1> M = M <arep2> } 
#F   represented by a list of amats.
#F   The convention for the intertwining space follows 
#F   Clausen, Baum. It is consistent with Minkwitz. 
#F   Note that Int( <arep1>, <arep2> ) consists of 
#F   deg(rep2) x deg(rep1)-matrices.
#F

# ARepOps.IntertwiningSpacePermRepsNC( <permrep1>, <permrep2> )
#   computes the intertwining space using an orbit algorithm
#   for the permutation operation involved. This function does
#   no argument checking, it is for internal use only.
#

ARepOps.IntertwiningSpacePermRepsNC := function ( R1, R2 )
  local orbits, base;

  # compute the orbits of the action of G on the set
  # [1..d1] x [1..d2] (di = Ri.degree) via the operation
  #   [i1, i2]^g := [i1^(R1(g)^-1), i2^(R2(g)^-1)] 
  # for g in G, i1 in [1..d1], i2 in [1..d2].
  orbits := 
    Orbits(
      R1.source, 
      Cartesian([1..R1.degree], [1..R2.degree]), 
      function ( i, g )
        return [ i[1]/PermAMat(g^R1), i[2]/PermAMat(g^R2) ];
      end
    );

  # construct the base corresponding to the orbits
  base :=
    List(
      orbits, 
      function ( orbit )
        local A, i;

        A := MatAMat(NullAMat([R1.degree, R2.degree], R1.char));
        for i in orbit do
          A[i[1]][i[2]] := AMatOps.OneNC(R1.char);
        od;
        return A;
      end
    );

  return List(base, AMatMat);
end;


# ConjugationMat( <L1>, <L2> )
#   For two lists of the same length of square matrices
#   <L1> and <L2> a base of the vectorspace V characterized
#   through
#     V = < A | M1 * A = A * M2 forall Mi in Li >
#   is calculated as a list of amats. 
#   The method is solving linear equations.


ConjugationMat := function( L1, L2 )
  local d1, d2, char, LGS, i, j, M1, M2, M, base;

  if not (
    IsList(L1) and 
    IsList(L2) and
    Length(L1) = Length(L2) and
    ForAll(L1, IsMat) and
    ForAll(L2, IsMat) and
    ForAll(L1, m -> DimensionsMat(m)[1] = DimensionsMat(m)[2]) and
    ForAll(L2, m -> DimensionsMat(m)[1] = DimensionsMat(m)[2])
  ) then
    Error("<L1> and <L2> must be lists of square matrices of the same length");
  fi;
  if not (
    DefaultField(L1[1][1][1]).char = 
    DefaultField(L2[1][1][1]).char 
  ) then
    Error("matrices in <L1> and <L2> must have the same char");
  fi;

  d1   := DimensionsMat(L1[1])[1];
  d2   := DimensionsMat(L2[1])[1];
  char := DefaultField(L1[1][1][1]).char;
  LGS  := List([1..d1*d2], i -> [ ]);
  for i in [1..Length(L1)] do
    M1 := 
      TensorProductAMat(
        IdentityPermAMat(d1, char),
        AMatMat(L2[i])
      );
    M2 := 
      TensorProductAMat(
        TransposedAMat(AMatMat(L1[i])),
        IdentityPermAMat(d2, char)
      );
    M := MatAMat(M1) - MatAMat(M2);
    for j in [1..d1*d2] do
      Append(LGS[j], M[j]);
    od;
  od;

  # compute a basis B of the nullspace of LGS as a list of matrices A
  base := NullspaceMat(LGS);
  base :=
    List(
      base, 
      function ( Aflat )
        return 
          List([1..d1], i -> 
            Sublist(Aflat, [1 + (i - 1)*d2 .. i*d2])
          );
      end
    );

  if Length(base) = 0 then
    return [ NullAMat( [d1, d2], char ) ];
  fi;

  return List(base, AMatMat);
end;


# ARepOps.IntertwiningSpaceLinearEquationsNC( <R1>, <R2> )
#   computes the intertwining space by solving the linear
#   equations for the matrices directly. 
#   This function is the most general but may be very slow 
#   since there are deg(R1)*deg(R2) many variables. 
#   The function does no argument checking, it is for internal 
#   use only.
#

ARepOps.IntertwiningSpaceLinearEquationsNC := function( R1, R2 )
  
   return
     ConjugationMat(
       List(R1.source.theGenerators, g -> MatAMat(ImageARep(g, R1))),
       List(R2.source.theGenerators, g -> MatAMat(ImageARep(g, R2)))
     );
end;


IntertwiningSpaceARep := function ( R1, R2 )

  # check R1, R2
  if not (
    IsARep(R1) and IsARep(R2) and
    R1.source = R2.source and
    R1.char   = R2.char
  ) then
    Error("<R1>, <R2> must be areps of a common group of same char");
  fi;

  # check for special cases
  if IsPermRep(R1) and IsPermRep(R2) then
    return ARepOps.IntertwiningSpacePermRepsNC(R1, R2);
  fi;
  return ARepOps.IntertwiningSpaceLinearEquationsNC(R1, R2);
end;


#F IntertwiningNumberARep( <arep1>, <arep2> )
#F   calculates the intertwining number of <arep1>, <arep2>
#F   which is the dimension of the intertwining space
#F   or the scalar product of the characters resp.
#F   Since the characters are used for computation, this
#F   function works only if the Maschke condition holds
#F   for both R1 and R2.
#F

IntertwiningNumberARep := function ( R1, R2 )
  local R3;

  # check R1, R2
  if not (
    IsARep(R1) and IsARep(R2) and
    IsIdentical(R1.source, R2.source) and
    R1.char = R2.char
  ) then
    Error("<R1>, <R2> must be areps of a common group of same char");
  fi;

  if not IsIdentical(R1.source, R2.source) then
    R3 := ARepOps.CopyWithNewSource(R2, R1.source);
    return ScalarProduct(CharacterARep(R1), CharacterARep(R3));
  fi;

  return ScalarProduct(CharacterARep(R1), CharacterARep(R2));
end;


#F Functions for Permutation and Monomial Representations
#F ------------------------------------------------------
#F
#F UnderlyingPermARep( <arep> )
#F   constructs the underlying permrep of the monrep <arep>
#F   as a "perm"-ARep, which can be obtained by replacing 
#F   all entries in the monomial matrices by 1 (in the 
#F   suitable field).
#F

UnderlyingPermARep := function ( R )
  local R1;

  if not IsARep(R) then 
    Error("usage: IsTransitivePermRep( <arep> )");
  fi;
  if not IsMonRep(R) then
    Error("<R> must be a monrep");
  fi;

  # check if R is already a permrep
  if IsPermRep(R) then 
    return PermARepARep(R);
  fi;

  R  := MonARepARep(R);
  R1 := 
    ARepByImages(
      R.source,
      List(R.theImages, m -> m.perm),
      R.degree,
      R.char,
      "hom"
    );

  # check if R.induction is bound
  if IsBound(R.induction) then
    R1.induction := 
      ConjugateARep(
        InductionARep(
          TrivialMonARep(R.induction.rep.rep.source, 1), 
          R.source
        ),
        IdentityMonAMat(R.degree, R.char)
      );
  fi;
 
  return R1;
end;


#F IsTransitiveMonRep( <arep> )
#F   decides whether <arep> is a transitive monrep or not.
#F   The result is stored in the field <arep>.isTransitive.
#F

IsTransitiveMonRep := function ( R )
  local M;

  if not IsARep(R) then 
    Error("usage: IsTransitiveMonRep( <arep> )");
  fi;
  if not IsMonRep(R) then
    Error("<R> must be a monrep");
  fi;

  if IsBound(R.isTransitive) then
    return R.isTransitive;
  fi;
  if IsBound(R.transitivity) and R.transitivity >= 1 then
    R.isTransitive := true;
    return R.isTransitive;
  fi;

  M := UnderlyingPermARep(MonARepARep(R));
  R.isTransitive := 
    IsTransitive(Group(M.theImages, ()), [1..R.degree]);
  return R.isTransitive;
end;


#F IsPrimitiveMonRep( <arep> )
#F   decides whether <arep> is a primitive monrep or not.
#F

IsPrimitiveMonRep := function ( R )
  local M;

  if not IsARep(R) then 
    Error("usage: IsPrimitiveMonRep( <arep> )");
  fi;
  if not IsMonRep(R) then
    Error("<R> must be a monrep");
  fi;

  # use, that a monrep is primitive, if it
  # is double transitive
  if IsBound(R.transitivity) and R.transitivity >= 2 then
    return true;
  fi;

  M := UnderlyingPermARep(MonARepARep(R));
  return 
    IsPrimitive(Group(M.theImages, ()), [1..R.degree]);
end;


#F TransitivityDegreeMonRep( <arep> )
#F   returns the transitivity degree of the monrep <arep>
#F

TransitivityDegreeMonRep := function ( R )
  if not IsARep(R) then 
    Error("usage: TransitivityDegreeMonRep( <arep> )");
  fi;
  if not IsMonRep(R) then
    Error("<R> must be a monrep");
  fi;

  if IsBound(R.transitivity) then
    return R.transitivity;
  fi;

  R              := UnderlyingPermARep(MonARepARep(R));
  R.transitivity := 
    Transitivity(Group(R.theImages, ()), [1..R.degree]);
  
  return R.transitivity;
end;


#F OrbitDecompositionMonRep( <arep> )
#F   decomposes the monrep <arep> with respect to the
#F   orbits as
#F     <arep> = DirectSum(R1, .., Rn) ^ P
#F   where R1 .. Rn are "mon"-AReps and P denotes a 
#F   "perm"-AMat. 
#F

OrbitDecompositionMonRep := function ( R )
  local G, orbs, pi, L, i;

  if not IsARep(R) then
    Error("usage: OrbitDecompositionARep( <arep> )");
  fi;
  if not IsMonRep(R) then
    Error("<R> must be a monrep");
  fi;

  R := MonARepARep(R);

  # the group obtained by the underlying perms of the images
  G    := Group(List(R.theImages, m -> m.perm), ());
  orbs := Orbits(G, [1..R.degree]);

  # construct perm to conjugate the orbits in a row
  pi   := MappingPermListList([1..R.degree], Concatenation(orbs));

  # construct transitive constituents
  L    := [ ];
  for i in [1..Length(orbs)] do
    L[i] := 
      ARepByImages( 
        R.source, 
        List(
          R.theImages, 
          m -> Mon(Permutation(m.perm, orbs[i]), Sublist(m.diag, orbs[i]))
        ),
        "hom"
      );
  od;

  return 
    ConjugateARep(
      DirectSumARep(L), 
      AMatPerm(pi, R.degree, R.char),
      "invertible"
    );
end;


#F TransitiveToInductionMonRep( <arep> [, <point> ] )
#F   decomposes a transitive monrep <arep> as 
#F     <arep> = 
#F       ConjugateARep(
#F         InductionARep(L, <arep>.source, T),
#F         D
#F       )
#F   where Stab is the stabilizer of <point>, L a onedimensional
#F   "mon"-arep of Stab and D a diagonal "mon"-AMat. The list
#F   T is a transversal of Stab in <arep>.source.
#F   The default for <point> is the largest point (<arep>.degree).
#F   If <arep> is a permrep then D is the IdentityMonAMat of
#F   suitable size and char. If <point> = <arep>.degree, then
#F   the result is stored in <arep>.induction.
#F

TransitiveToInductionMonRep := function ( arg )
  local R, point, store, P, GP, Stab, T, i, diag, L;

  # dispatch
  if Length(arg) = 1 then
    R     := arg[1];
    point := R.degree;
  elif Length(arg) = 2 then
    R     := arg[1];
    point := arg[2];
  else
    Error("usage: TransitiveToInductionMonRep( <arep> [, <point> ] )");
  fi;

  if not IsARep(R) then 
    Error("<R> must be an arep");
  fi;
  if not IsMonRep(R) then
    Error("<R> must be a monrep");
  fi;
  if not IsTransitiveMonRep(R) then
    Error("<R> must be transitive");
  fi;
  if not (
    IsInt(point) and
    point > 0 and
    point <= R.degree 
  ) then
    Error("must be 1 <= <point> <= R.degree");
  fi;

  R  := MonARepARep(R);

  # check if stored
  if point = R.degree and IsBound(R.induction) then
    return R.induction;
  fi;

  P  := UnderlyingPermARep(R);
  GP := Group(P.theImages, ());

  # store result if point = R.degree
  if point = R.degree then
    store := true;
  else
    store := false;
  fi;

  Stab := 
    GroupWithGenerators(
      PreImages(
        ARepOps.Hom(P), 
        Stabilizer(GP, point)
      )
    );

  # construct transversal T of P.source/Stab with the 
  # property point^(T[i]^P) = i
  T := [ ];
  for i in [1..P.degree] do
    T[i] := 
      PreImagesRepresentative(
        ARepOps.Hom(P),
        RepresentativeOperation(GP, point, i)
      );
  od;

  # catch the case that R is even a permrep
  if IsPermRep(R) then
    if store then
      R.induction := 
	ConjugateARep(
	  InductionARep(
	    TrivialMonARep(Stab, 1, R.char),
	    R.source,
	    T
	  ),
	  IdentityMonAMat(R.degree, R.char),
	  "invertible"
	);
      return R.induction;
    else
      return
	ConjugateARep(
	  InductionARep(
	    TrivialMonARep(Stab, 1, R.char),
	    R.source,
	    T
	  ),
	  IdentityMonAMat(R.degree, R.char),
	  "invertible"
	);
    fi;
  fi;

  # construct 1-dimensional representation from which
  # M is induced
  L := 
    ARepByImages(
      Stab,   
      List(
        Stab.theGenerators, 
        g -> Mon((), [ MonAMat(g^R).diag[point] ])
      ),
      "hom"
    );

  # construct the diagonal matrix which conjugates
  # the induced representation into M. Note that here
  # it makes a difference whether we consider the
  # representation corresponding to a right or a left
  # module. Our representations act from the right.
  diag := 
    List(
      T, 
      t -> 
        TransposedMon(
          MonAMat(ImageARep(t, R))
        ).diag[point]
    );
  if store then
    R.induction := 
      ConjugateARep(
	InductionARep(L, R.source, T), 
	DiagonalAMat(diag),
	"invertible"
      );
    return R.induction;
  else
    return
      ConjugateARep(
	InductionARep(L, R.source, T), 
	DiagonalAMat(diag),
	"invertible"
      );
  fi;
end;


#F Transformation and Decomposition of areps
#F -----------------------------------------
#F
#F InsertedInductionARep( <"induction"-arep>, <group> )
#F   given an "induction"-ARep RUG = InductionARep(RU, G)
#F   and the <group> H such that U <= H <= G, this function
#F   decomposes RUG into
#F     RUG = InductionARep(InductionARep(RU, H), G) ^ M
#F   where M is an AMat with structure similar to the
#F   induced representation RUG.
#F

InsertedInductionARep := function ( RUG, H )
  local 
    RUHG,          # Ind(Ind(RU, H, TUH), G, THG) = Ind(RU, G, TUHG)
    G, H, U,       # G >= H >= U
    TUG,           # transversal of Ind(RU, G)
    TUHG,          # complex product TUH*THG
    perm, diag, C, # monomial correction matrix
    t, i, j;       # temporaries

  if not ( IsARep(RUG) and RUG.type = "induction" ) then
    Error("<RUG> must be an \"induction\"-ARep");
  fi;
  if not ( 
    IsGroup(H) and 
    ForAll(H.generators, h -> h in RUG.source) 
  ) then
    Error("<H> must be a group contained in <RUG>.source");
  fi;
  if not (
    ForAll(RUG.rep.source.generators, u -> u in H)
  ) then
    Error("<H> must contain <RUG>.rep.source");
  fi;
  TUG := RUG.transversal;

  # construct double induction
  RUHG := 
    InductionARep(
      InductionARep(RUG.rep, H),
      RUG.source
    );
  G := RUHG.source;
  H := RUHG.rep.source;
  U := RUHG.rep.rep.source;

  # build complex product transversal
  TUHG := [ ];
  for t in RUHG.transversal do
    Append(TUHG, RUHG.rep.transversal * t);
  od;

  # find perm,diag such that U*TUHG[i] = U*TUG[i^perm]
  perm := [ ];
  diag := [ ];
  for i in [1..Length(TUHG)] do
    j := 1;
    while not TUHG[i]/TUG[j] in U do
      j := j + 1;
    od;
    Add(perm, j);
    Add(diag, ImageARep(TUHG[i]/TUG[j], RUG.rep));
  od;
  perm := PermList(perm);
  diag := Permuted(diag, perm);

  # construct result, in the case RUG.rep.degree = 1
  # the conjugating amat is of type perm resp. mon
  if RUG.rep.degree = 1 then
    if ARepOps.IsTrivialOneRep(RUG.rep) then
      C := 
        AMatPerm(
          perm, 
          Length(diag), 
          RUG.char
        );
    else
      C := 
        AMatMon(
          Mon(
            perm, 
            List(diag, x -> MatAMat(x)[1][1])
          )
        );
    fi;
  else
    C := 
      TensorProductAMat(
        AMatPerm(perm, Length(TUG), RUG.char), 
        IdentityPermAMat(RUG.rep.degree, RUG.char)
      ) *
      DirectSumAMat(diag);
  fi;
  return ConjugateARep(RUHG, C, "invertible");
end;


# ConjugationPermLists ( [ <permgrp>,] <list-of-perm1>, <list-of-perm2> )
#   calculates a permutation p in <permgrp> which conjugates
#   <list-of-perm1> elementwise onto <list-of-perm2>. 
#   The default for <permgrp> is the symmetric group on degree 
#   many points.
#

# The algorithm:
#   conjugate the first permutations onto eachother, calculate
#   the centralizer and go on by conjugating with elements
#   in the suitable coset.

ConjugationPermLists := function ( arg )
  local G, perms1, perms2, max, p, lmp, i, r;

  if Length(arg) = 3 then
    G      := arg[1];
    perms1 := arg[2];
    perms2 := arg[3];
  elif Length(arg) = 2 then
    perms1 := arg[1];
    perms2 := arg[2];
  else
    Error(
      "usage: \n",
      "  ConjugationPermLists ( [ <permgrp>,] <list-of-perm>, <list-of-perm> )"
    );
  fi;

  if not (
    IsList(perms1) and
    Length(perms1) > 0 and
    ForAll(perms1, IsPerm) and
    IsList(perms2) and
    Length(perms2) = Length(perms1) and
    ForAll(perms2, IsPerm) 
  ) then
    Error("<perms1> and <perms2> must be permlists of the same length");
  fi;

  # get largest moved point
  max := 1;
  for p in perms1 do
    if p <> () then
      lmp := LargestMovedPointPerm(p);
    else
      lmp := 0;
    fi;
    if lmp > max then
      max := lmp;
    fi;
  od;
  for p in perms2 do
    if p <> () then
      lmp := LargestMovedPointPerm(p);
    else
      lmp := 0;
    fi;
    if lmp > max then
      max := lmp;
    fi;
  od;

  # set G the symmetric group 
  if Length(arg) = 2 then
    G := SymmetricGroup(max);
  fi;

  if not IsPermGroup(G) then
    Error("<G> must be a permgroup");
  fi;

  # check if cycletypes are equal
  for i in [1..Length(perms1)] do
    if 
      Collected(CycleLengths(perms1[i], [1..max])) <>
      Collected(CycleLengths(perms2[i], [1..max]))
    then
      return false;
    fi;
  od;

  p := ( );
  for i in [1..Length(perms1)] do
    r := RepresentativeOperation(G, perms1[i] ^ p, perms2[i]);
    if r = false then
      return false;
    fi;
    G := Centralizer(G, perms2[i]);
    p := p * r;
  od;  

  return p;
end;


#F ConjugationTransitiveMonReps( <arep1>, <arep2> )
#F   returns a "mon"-amat m on <arep1>.degree many points
#F   such that <arep1> ^ m = <arep2> and false if this is
#F   not possible. The areps must have common source and
#F   characteristic.
#F

# The algorithm:
#   1. Decompose R1 as 
#        R1 = InductionARep(L1, G, T1) ^ D1,
#      with a onedimensional rep L1 of H <= G, a transversal 
#      T1 of H\G and a diagonal matrix D1.
#   2. Test whether H stabilizes a point via R2, if not, 
#      return false, else decompose R2 as
#        R2 = InductionARep(L2, G, T2) ^ D2,
#      with a onedimensional rep L2 of H, a transversal 
#      T2 of H\G and a diagonal matrix D2.
#   3. Test whether L1 and L2 are conjugated, i.e. there is
#      an s in G with
#        L2(x) = L1(s*x*s^-1) for all x in H,
#      if not return false, else we have the identity
#        R2 = InductionARep(L1, G, s*T2) ^ D2.
#   4. Calculate a monomial matrix M, such that
#        InductionARep(L1, G, s*T2) =
#        InductionARep(L1, G, T1) ^ M,
#      by change of transversal T1 -> s*T2.
#   5. Return D1^-1 * M * D2

ConjugationTransitiveMonReps := function ( R1, R2 )
  local M1, M2, H, perms, point, s, t, i, T1, T2, Tpi, pi, D;

  if not ( IsARep(R1) and IsARep(R2) ) then
    Error("usage: ConjugationTransitiveMonReps( <arep1>, <arep2> )");
  fi;
  if not ( IsMonRep(R1) and IsMonRep(R2) ) then
    Error("<R1> and <R2> must be monreps");
  fi;
  if not ( IsTransitiveMonRep(R1) and IsTransitiveMonRep(R1) ) then
    Error("<R1> and <R2> must be transitive");
  fi;
  if not (
    R1.source = R2.source and
    R1.char = R2.char
  ) then
    Error("<R1> and <R2> must be reps of the same source and char");
  fi;

  if not ( R1.degree = R2.degree ) then
    return false;
  fi;
  R1 := MonARepARep(R1);
  R2 := MonARepARep(R2);
  M1 := TransitiveToInductionMonRep(R1);
  H  := M1.rep.rep.source;

  # check if Stabilizer of R1 stabilizes 
  # one point via R2 and if so, decompose
  # R2 w.r.t to that point. i.e. as induction 
  # from H
  perms := 
    List(
      H.generators, 
      g -> MonAMat(ImageARep(g, R2)).perm
    );
  point := 
    First(
      [1..R2.degree], 
      i -> ForAll(perms, p -> i^p = i)
    );
  if point = false then
    return false;
  fi;
  M2 := TransitiveToInductionMonRep(R2, point);

  # test if the onedimensional representations in 
  # M1 and M2 are conjugated by s, calculate s
  s  := false;
  i  := 1;
  while s = false and i <= R1.degree do
    t := M1.rep.transversal[i];
    if 
      ForAll(
        H.generators,
        x -> 
          t*x/t in H and
          MonAMat(ImageARep(x, M2.rep.rep)) = 
          MonAMat(ImageARep(t*x/t, M1.rep.rep))
      )
    then
      s := t;
    else
      i := i + 1;
    fi;
  od;

  # if no s found, R1 and R2 are not equivalent
  if i > R1.degree then
    return false;
  fi;

  # calculate change of transversal
  T1 := M1.rep.transversal;
  T2 := s*M2.rep.transversal;
  pi := 
    ARepOps.ConjugationPermTransversalNC( 
      R1.source,
      H,
      T1,
      T2
    );
  Tpi := Permuted(T1, pi);
  D   := 
    List(
      List([1..R1.degree], i -> T2[i]/Tpi[i]),
      x -> ImageARep(x, M1.rep.rep)
    );

  return 
    MonAMatAMat(
      M1.conjugation^-1*
      AMatPerm(pi, R1.degree, R1.char)*
      DirectSumAMat(D)*
      M2.conjugation
    );
end;


#F ConjugationPermReps( <arep1>, <arep2> )
#F   returns for permutation areps <arep1>, <arep2> a 
#F   "perm"-amat p on <arep1>.degree many points such 
#F   that <arep1> ^ p = <arep2> and false if this is
#F   not possible. The areps must have common source 
#F   and characteristic.
#F

# The algorithm: 
#   Calculate the images of theGenerators and conjugate
#   them pointwise simultaneously onto eachother by the
#   function ConjugationPermLists above.

ConjugationPermReps := function ( R1, R2 )
  local images1, images2, p;

  if not ( IsARep(R1) and IsARep(R2) ) then
    Error("usage:  ConjugationPermReps( <arep1>, <arep2> )");
  fi;
  if not ( IsPermRep(R1) and IsPermRep(R2) ) then
    Error("<R1> and <R2> must be permreps");
  fi;
  if not R1.source = R2.source then
    Error("<R1> and <R2> must have the same source");
  fi;
  if not R1.char = R2.char then
    Error("<R1> and <R2> must have the same char");
  fi;

  # check degree
  if not R1.degree = R2.degree then
    return false;
  fi;

  R1      := PermARepARep(R1);
  R2      := PermARepARep(R2);
  images1 := R1.theImages;
  if R1.source.theGenerators = R2.source.theGenerators then
    images2 := R2.theImages;
  else
    images2 := 
      List(  
        ImageARep(R1.source.theGenerators, R2),
	PermAMat
      );
  fi;

  # determine conjugating permutation
  p := 
    ConjugationPermLists(
      images1,
      images2
    );
  if p = false then
    return false;
  fi;

  return AMatPerm(p, R1.degree, R1.char);
end;


#F TransversalChangeInductionARep( 
#F   <"induction"-arep>, <transversal> [, <hint> ] 
#F )
#F   Given an "induction"-ARep R,
#F     R = InductionARep(L, G, T)
#F   and a <transversal> of L.source\G, R is decomposed as
#F     R = InductionARep(L, G, <transversal>) ^ M.
#F   If L is a monrep (e.g. L.degree = 1), then M is a 
#F   "mon"-AMat, else an AMat with a structure similar to R.
#F   The <hint> "isTransversal" can be supplied to avoid
#F   testing it.
#F 

TransversalChangeInductionARep := function ( arg )
  local R, T, hint, i, j, T1, pi, Tpi, D, con;

  # dispatch
  if Length(arg) = 2 then
    R    := arg[1];
    T    := arg[2];
    hint := "no hint";
  elif Length(arg) = 3 then
    R    := arg[1];
    T    := arg[2];
    hint := arg[3];
  else
    Error(
      "usage :\n",
      "  TransversalChangeInductionARep( <induction-arep>, <transversal> [, <hint> ] )"
    );
  fi;

  # check arguments
  if not IsARep(R) and R.type = "induction" then
    Error("<R> must be an arep of type \"induction\"");
  fi;
  if not hint in ["no hint", "isTransversal"] then
    Error("hint must be \"isTransversal\"");
  fi;

  # check transversal T if no hint is given
  if hint = "no hint" then
    if ForAny(T, t -> not t in R.source) then
      Error("elements of <T> must lie in <R>.source");
    fi;
    if not Length(T) = Length(R.transversal) then
      Error("Length( <T> ) must be <R>.degree");
    fi;
    for i in [1..Length(R.transversal)] do
      for j in [i + 1..Length(R.transversal)] do
        if T[i]/T[j] in R.rep.source then
          Error("<T> must be a transversal of <R>.rep.source in <R>.source");
        fi;
      od;
    od;
  fi;

  # compute conjugating matrix
  T1 := R.transversal;
  pi := 
    ARepOps.ConjugationPermTransversalNC(
      R.source, R.rep.source, T, T1
    );
  Tpi := Permuted(T, pi);

  if IsPermRep(R) and R.rep.degree = 1 then
    D := IdentityMonAMat(R.degree, R.char);
  else
    D := 
      DirectSumAMat(
        List(
          List([1..Length(R.transversal)], i -> T1[i]/Tpi[i]),
          x -> PowerAMat(ImageARep(x, R.rep), -1, "invertible")
        )
      );
  fi;

  # conjugating matrix
  con := 
    TensorProductAMat(
      AMatPerm(pi, Length(R.transversal), R.char), 
      IdentityPermAMat(R.rep.degree, R.char)
    ) *
    D;

  # if R is monomial return with "mon"-AMat
  if IsMonRep(R) then
    return 
      ConjugateARep(
        InductionARep(
          R.rep,
          R.source,
          T
        ),
        MonAMatAMat(con),
        "invertible"
      );
  fi;
      
  return 
    ConjugateARep(
      InductionARep(
        R.rep,
        R.source,
        T
      ),
      con,
      "invertible"
    ); 

end;


# OTPDAbelianMonRepNC( <arep> )
#   actually means OuterTensorProductDecompositionAbelianMonRepNC
#   and has the same interface like OuterTensorProductDecompositionMonRep.
#   It deals with the special case, that <arep>.source is abelian.
#   In this case, R = <arep> decomposes the same way as G = <arep>.source
#   decomposes into a direct product G = N1 x .. x Nk.
#   The latter decomposition can be performed without computing
#   all normal subgroups od <arep>.source.
#   We use Martin Schoenerts function
#     IndependentGeneratorsAbelianPermGroup
#   on the regular permgroup associated with G.

# The function uses the following lemma:
#
# Lemma: R = L induction_T G, L rep of S <= G = N1 x .. x Nk.
#   Then R^M = L_1 induction_(T_1) N_1 # .. #  L_k induction_(T_k) N_k
#   with a monmat M, L_i = L restriction (S intersect Ni).
#

OTPDAbelianMonRepNC := function ( R )
  local G, RSG, L, S, PG, Ns, Rs, T, M;

  # decompose R into an induction
  G   := R.source;
  RSG := TransitiveToInductionMonRep(R);
  L   := RSG.rep.rep;
  S   := L.source;

  # decompose G into a direct product of cyclic groups
  # of prime power order using Martin Schoenerts function
  # for permgroups
  # all these groups are generated by one element
  PG := PermGroup(G);
  Ns := 
    List(
      IndependentGeneratorsAbelianPermGroup(PG),
      g -> GroupWithGenerators(Subgroup(G, [ Image(PG.bijection, g) ]))
    );

  # construct factors of RSG.rep corresponding
  #   G = N1 x .. x Nk
  # the factors are all of the form
  # (L restriction (S intersect Ni)) induction Ni
  # i = 1..k
  Rs :=
    List(
      Ns,
      n ->
        InductionARep(
          RestrictionARep(L, GroupWithGenerators(Intersection(S, n))),
          n
        )
    );

  # cartesian product of the transversals
  T := List(Cartesian(List(Rs, r -> r.transversal)), Product);

  # monomial matrix M from transversal change
  M := TransversalChangeInductionARep(RSG.rep, T).conjugation;

  return
    ConjugateARep( 
      OuterTensorProductARep(G, List(Rs, MonARepARep)),
      MonAMatAMat(M * RSG.conjugation),
      "invertible"
    );
end;


#F OuterTensorProductDecompositionMonRep( <arep> )
#F   decomposes the transitive monrep <arep> into a 
#F   conjugated (by a "mon"-AMat) outer tensorproduct 
#F   of "mon"-AReps as far as possible, namely
#F     <arep> =
#F       ConjugateARep(
#F         OuterTensorProductARep(
#F           <arep>.source,
#F           "mon"-ARep1, .., "mon"-ARepN
#F         ),
#F         M
#F       )
#F   with a monomial matrix M.
#F

# The algorithm:
#   1. Decompose RG as induction of a onedimensional 
#      representation L of S with a transversal T
#      conjugated by a diagonal matrix D.
#
#   2. Use the following lemma to check whether RG
#      decomposes as outer tensorproduct:
#      Lemma:
#        RG = RN1 # RN2 iff
#        a. G = N1 x N2
#        b. |S| = |N1 intersection S| * |N2 intersection S|
#      and calculate N1, N2, S1, S2
#   3. Calculate 
#        L1 = L restriction S1
#        L2 = L restriction S2
#   4. Construct
#        RN1    = L1 induction N1 with transversal T1
#        RN2    = L1 induction N2 with transversal T2
#        RN1xN2 = RN1 # RN2,
#      now we have
#        RG     = (L induction G with transversal T) ^ D
#        RN1xN2 = L induction G with transversal T1*T2
#   5. Map T12 on T to get monomial basechange M, what
#      means
#        RG = RN1xN2 ^ (M * D)
#   6. Try to decompose RN2 and collect factors.

# ...hier fehlt ein Satz ueber die Eindeutigkeit dieser Zerlegung,
# resp. die Unabhaengigkeit der Zerlegung von dem Zerlegungsweg.

OuterTensorProductDecompositionMonRep := function ( R )
  local 
    G,         # group represented
    S,         # stabilizer of the last point
    RSG,       # RG as conjugated induction from S
    Ns,        # normal subgroups of G
    factors,   # normal subgroups decomposing RG
    Rs,        # reps of the factors
    stop,      # flag to break loop
    i, j,      # counter
    N1, N2,    # normal subgroups decomposing RG
    S1, S2,    # Ni intersection S
    L, L1, L2, # onedim reps of S, S1, S2 
    RN1, RN2,  # induction from Li to Ni
    RN1xN2,    # OuterTensorProductARep(RN1, RN2)
    T,         # transversal of RG as induction of L
    T12,       # transversal of RN1xN2 as induction of L
    M;         # "mon"-AMat for transversal change 

  if not ( 
    IsARep(R) and
    IsMonRep(R) and
    IsTransitiveMonRep(R)
  ) then
    Error("<R> must be a transitive monrep");
  fi;
 
  # catch abelian case
  if IsAbelian(R.source) and IsPermRep(R) and IsFaithfulARep(R) then
    return OTPDAbelianMonRepNC(R);
  fi;

  R   := MonARepARep(R);
  RSG := TransitiveToInductionMonRep(R);
  G   := R.source;
  S   := RSG.rep.rep.source;
  Ns  := NormalSubgroups(G);

  # begin with i = 2 to avoid trivial factorization
  i := 2; 

  # calculate the first normal subgroup having
  # a direct tensor complement, note that the
  # normal subgroups have to be ordered with
  # respect to increasing size
  factors := false;
  stop    := false;
  while i < Length(Ns) and factors = false do
    j := i + 1;
    while j <= Length(Ns) and not stop do
      if not (
        Size(Ns[i]) * Size(Ns[j]) = Size(G) and
        IsTrivial(Intersection(Ns[i], Ns[j])) 
      ) then 
        j := j + 1;
      else

        # Si can be calculated via Stabilizer, too
        S1 := GroupWithGenerators(Intersection(Ns[i], S)); 
        S2 := GroupWithGenerators(Intersection(Ns[j], S)); 
        if not ( Size(S1) * Size(S2) = Size(S) ) then
          j := j + 1;
        else
          N1      := GroupWithGenerators(Ns[i]);
          N2      := GroupWithGenerators(Ns[j]);
          factors := [N1, N2];
          stop    := true;
        fi;
      fi;
    od;
    i := i + 1;
  od;

  if factors = false then
    return 
      ConjugateARep(
        OuterTensorProductARep(R),
        IdentityPermAMat(R.degree, R.char),
        "invertible"
      );
  fi;

  # set the fields N2.normalSubgroups which is needed
  # for further decomposition; use the lemma
  # Lemma: G = N1 x N2, then
  #   {N | N isNormalSubgroup Ni} =
  #   {N | N isNormalSubgroup G and N isSubset Ni} 
  N2.normalSubgroups := Filtered(Ns, N -> IsSubgroup(N2, N));

  # get onedimensional representation L, L1, L2 of S, S1, S2
  L   := RSG.rep.rep;
  L1  := RestrictionARep(L, S1);
  L2  := RestrictionARep(L, S2);

  # construct prototype equivalent to R
  RN1    := InductionARep(L1, N1);
  RN2    := InductionARep(L2, N2);
  RN1xN2 := OuterTensorProductARep(G, RN1, RN2);

  # RN1xN2 has to be conjugated onto RSG.rep. Both
  # reps are inductions of the same rep L, namely
  #   RN1xN2  = Induction(L, G, T12),
  #   RSG.rep = Induction(L, G, T)
  # where T12 is the product of the transversals of
  # RN1 and RN2. Map T12 onto T to get conjugating
  # monomial matrix M.
  T   := RSG.rep.transversal;
  T12 := 
    Concatenation(
      List(RN1.transversal, t -> t*RN2.transversal)
    );
  M := 
    TransversalChangeInductionARep(
       RSG.rep,
       T12,
      "isTransversal"
    ).conjugation;

  # try to decompose RN2 and construct recursively
  RN2 := OuterTensorProductDecompositionMonRep(RN2);

  return 
    ConjugateARep(
      OuterTensorProductARep(
        G,
        List(
          Concatenation(
            [RN1],
            RN2.rep.factors
          ),
        MonARepARep
       )
      ),
      MonAMatAMat(
        TensorProductAMat(
          IdentityPermAMat(RN1.degree, R.char),
          RN2.conjugation
        ) *
        M *
        RSG.conjugation
      ),
      "invertible"
    );
  
end;


#F InnerConjugationARep( <arep>, <supergroup>, <element> )
#F   calculates <arep> ^ <element> as representation of 
#F   <arep>.source ^ <element>, <element> must lie in <supergroup>.
#F   For a representation R of H <= G, and an 
#F   element t in G the representation R ^ t is defined as
#F     (R ^ t)(x) = R(t x t^-1) for all x in H ^ t.
#F   The returned arep is of type "perm", "mon" or "mat", 
#F   whatever poosible.
#F   If H is normal in G, then R ^ t is returned as rep of H, 
#F   else  H ^ t is constructed as a group with
#F     (H ^ t).theGenerators = H.theGenerators ^ t.
#F

InnerConjugationARep := function ( R, G, x )
  local K, Kx;

  if not( IsARep(R) and IsGroup(G) and x in G ) then
    Error(
      "usage: InnerConjugationARep( <arep>, <supergroup>, <element> )"
    );
  fi;
  if not ForAll(R.source.generators, g -> g in G) then 
    Error("<G> must contain <R>.source");
  fi;

  K := 
    GroupWithGenerators(
	  AsSubgroup(Parent(G), ShallowCopy(R.source))
	);
  if IsNormal(G, K) then
    Kx := K;
  else
    Kx := K ^ x;
	Kx.theGenerators := 
	  List(
	    K.theGenerators,
		g -> g ^ x
	  );
  fi;

  if IsPermRep(R) then
    return
      ARepByImages(
	    Kx,
        List(
          Kx.theGenerators, 
          g -> PermAMat((g ^ (x ^ -1)) ^ R)
        ),
        R.degree,
        R.char,
        "hom"
      );
  elif IsMonRep(R) then
    return
      ARepByImages(
	    Kx,
        List(
          Kx.theGenerators, 
          g -> MonAMat((g ^ (x ^ -1)) ^ R)
        ),
        "hom"
      );
  else
    return
      ARepByImages(
	    Kx,
        List(
          Kx.theGenerators, 
          g -> MatAMat((g ^ (x ^ -1)) ^ R)
        ),
        "hom"
      );
  fi;
end;


#F RestrictionInductionARep( <"induction"-arep>, <subgroup> )
#F   Given an <"induction"-arep> R of G, induced from a 
#F   onedimensional representation L of a subgroup H <= G 
#F   and a <subgroup> K of G, the function calculates 
#F   an arep equal to R decomposing (R restriction K) 
#F   according to Mackey's subgroup theorem. 
#F   If s_1, .., s_k represent the double cosets H\G/K, 
#F   the the decomposition is given by
#F     (R restriction K) = 
#F       ConjugateARep(
#F         DirectSumARep(
#F           InductionARep(R_i, K), 
#F           i = 1..k
#F         ),
#F         M
#F       ),
#F    where M is a monomial matrix and R_i are onedimensional 
#F    "mon"-areps obtained by restricting L^(s_i) to 
#F    (H^(s_i) intersect K).
#F           

# The algorithm: R is an induction of a onedimensional rep L of
#                H to G with transversal T. K is a subgroup of G.
#
#   a. Calculate a set S = (s_1, .., s_n) of representatives of 
#      the double cosets H\G/K.
#   b. Calculate transversals T_i of (H ^ s_i intersection K)\K.
#      T' = Concatenation(s_i * T_i, i = 1..n) is a transversal
#      of H\G.
#   c. Calculate the monomial matrix M to the tranversal change
#      T -> T'.
#   d. Calculate R_i = L ^ (s_i) restriction (H ^ s_i intersection K)\K
#      an make it a "mon"-arep.
#
#   R restriction K = 
#     ConjugateARep(
#       DirectSumARep(
#         InductionARep(R_i, K), 
#         i = 1..k
#       ),
#       M
#     )


RestrictionInductionARep := function ( R, K )
  local G, L, H, K, S, HsintK, Ts, T, M;

  if not IsARep(R) and IsGroup(K) then
    Error(
      "usage: ", 
      "  RestrictionInductionARep( <induction-arep>, <subgroup> )"
    );
  fi;
  if not R.type = "induction" then
    Error("<R> must be an induction");
  fi;
  if not R.rep.degree = 1 then
    Error("<R.rep> must have degree 1");
  fi;

  G := R.source;
  L := R.rep;
  H := L.source;
  if not (
    IsGroup(K) and
    ForAll(K.generators, g -> g in G)
  ) then
    Error("<K> must be a subgroup of <G>");
  fi;
  K := GroupWithGenerators(AsSubgroup(Parent(G), K));

  # use Mackey's theorem
  S      := List(DoubleCosets(G, H, K), d -> d.representative);
  HsintK := List(S, s -> Intersection(H^s, K));
  Ts     := List(HsintK, g -> Transversal(K, g));

  # transversal of H\G
  T := 
    Concatenation(
      List([1..Length(Ts)], i -> List(Ts[i], t -> S[i]*t))
    );

  # transversal change
  M := TransversalChangeInductionARep(R, T).conjugation;

  return 
    ConjugateARep(
      DirectSumARep(
        List(
          [1..Length(Ts)], 
          i -> 
            InductionARep(
              MonARepARep(
                RestrictionARep(
                  InnerConjugationARep(L, G, S[i]),
                  GroupWithGenerators(HsintK[i])
                )
              ),
              K,
              Ts[i]
            )
        )
      ),
      M,
      "invertible"
    );
end;


# AllMaximalNormalSubgroupsBetween( <group>, <subgroup> )
#   calculates all normal subgroups N of <group> with
#     <subgroup> <= N < <group> and (<group> : N) = prime,
#   if no such N exists, then false is returned.
#

AllMaximalNormalSubgroupsBetween := function ( G, H )
  local H, Nmin, GNmin, phi, L;

  if not( IsGroup(G) and IsGroup(H) ) then
    Error("usage: AllMaximalNormalSubgroupsBetween( <group>, <subgroup> )");
  fi;
  if not ForAll(H.generators, h -> h in G) then
    Error("<G> must contain <H>");
  fi;

  H     := AsSubgroup(Parent(G), H);
  Nmin  := NormalClosure(G, H);
  GNmin := G/Nmin;
  phi   := NaturalHomomorphism(G, GNmin);
  L     :=
    List(
      Filtered(
        NormalSubgroups(GNmin), 
        N -> IsPrime(Index(GNmin, N))
      ),
      N -> PreImage(phi, N)
    );
  if L = [ ] then
    return false;
  fi;

  return L;
end;


# OneMaximalNormalSubgroupBetween( <group>, <subgroup> )
#   calculates one normal subgroup N of <group> with
#     <subgroup> <= N < <group> and (<group> : N) = prime,
#   if no such N exists, then false is returned.
#

OneMaximalNormalSubgroupBetween := function ( G, H )
  local H, Nmin, GNmin, GNminag, phi, pos, N;

  if not( IsGroup(G) and IsGroup(H) ) then
    Error("usage: OneMaximalNormalSubgroupBetween( <group>, <subgroup> )");
  fi;
  if not ForAll(H.generators, h -> h in G) then
    Error("<G> must contain <H>");
  fi;

  H     := AsSubgroup(Parent(G), H);
  Nmin  := NormalClosure(G, H);
  if Nmin = G then 
    return false;
  fi;
  GNmin := FactorGroup(G, Nmin);
  phi   := NaturalHomomorphism(G, GNmin);

  # catch the solvable case
  if IsSolvable(GNmin) then
    GNminag := AgGroup(GNmin);
    N       := CompositionSeries(GNminag)[2];
    return PreImage(phi, Image(GNminag.bijection, N));
  fi;

  pos   :=
    PositionProperty(
      NormalSubgroups(GNmin), 
      N -> IsPrime(Index(GNmin, N))
    );
  if pos = false then
    return false;
  fi;
  N := NormalSubgroups(GNmin)[pos];

  return PreImage(phi, N);
end;


#F kbsARep( <arep> )
#F   determines the conjugated block structure of <arep>
#F   (cf. kbs in permblk.g). Note that if <arep> is 
#F   monomial, the kbs is exactly the list of orbits
#F   of <arep> on [1..R.degree].
#F

kbsARep := function ( R )
  if not IsARep(R) then
    Error("usage: kbsARep( <arep> )");
  fi;

  if IsMonRep(R) then
    return
      List( 
	Orbits(
	  Group(
	    List(
	      MonARepARep(R).theImages, 
	      m -> m.perm
	    ), 
	    ()
	  ), 
	  [1..R.degree]
	),
	Set
      );
   fi;

   return kbsAMat(List(R.source.theGenerators, g -> g ^ R));
end;


#F RestrictionToSubmoduleARep( <arep>, <list> [, <hint> ] )
#F   calculates the restriction of <arep> to 
#F   the submodule generated by the basevectors
#F   with the indices in <list>. 
#F   Note that the word "restriction" here is not related to
#F   a restriction to a subgroup.
#F   The optional hint "hom" indicates, that the restriction
#F   yields a representation, i.e. that <list> 
#F   is a union of lists in the kbs of <arep>.
#F   The restriction is of type "perm", "mon", "mat", if
#F   <arep> is a perm-, mon-, matrep.
#F

RestrictionToSubmoduleARep := function ( arg )
  local R, list, hint;

  if Length(arg) = 2 then
    R    := arg[1];
    list := arg[2];
    hint := "no hint";
  elif Length(arg) = 3 then
    R    := arg[1];
    list := arg[2];
    hint := arg[3];
  else
    Error(
      "usage: RestrictionToSubmoduleARep( <arep>, <list-of-posints> [, <hint> ] )"
    );
  fi;

  if not IsARep(R) then
    Error("<R> must be an arep");
  fi;
  if not (
    IsList(list) and
    ForAll(list, i -> IsInt(i) and 1 <= i and i <= R.degree)
  ) then
    Error("<list> must contain ints in [1..<R>.degree");
  fi;
  if not hint in ["no hint", "hom"] then
    Error("<hint> must be \"hom\" or \"no hint\"");
  fi;

  if IsPermRep(R) then
    R := PermARepARep(R);
    return 
      ARepByImages(
        R.source, 
        List(R.theImages, p -> Permutation(p, list)),
        Length(list),
        R.char,
        hint
      );
  fi;
  if IsMonRep(R) then
    R := MonARepARep(R);
    return
      ARepByImages(
        R.source,
        List(
          R.theImages, 
          m -> Mon(Permutation(m.perm, list), Sublist(m.diag, list))
        ),
        hint
      );
  fi;
  R := MatARepARep(R);
  return
    ARepByImages(
      R.source,
      List(
        R.theImages,
        m -> List(Sublist(m, list), l -> Sublist(l, list))
      ),
      hint
    );
end;


#F kbsDecompositionARep( <arep> )
#F   decomposes <arep> into a conjugated direct sum 
#F   according to the kbs (cf. permblk.g) as far as 
#F   possible
#F     <arep> = 
#F       ConjugateArep(
#F         DirectSumARep( <arep1>, .., <arepN> ),
#F         P
#F       )
#F   with a permutation matrix P. For monreps this
#F   function does exactly the same as the function
#F   OrbitDecompositionMonRep.
#F   The <arepi> are of type "perm", "mon", "mat", if
#F   <arep> is  a perm-, mon-, matrep resp.
#F

kbsDecompositionARep := function ( R )
  local sortperm;

  if not IsARep(R) then
    Error("usage: DirectSumDecompositionARep( <arep> )");
  fi;

  sortperm := PermList(Concatenation(kbsARep(R)));
  return
    ConjugateARep(
      DirectSumARep(
	List(
	  kbsAMat(List(R.source.theGenerators, g -> g ^ R)),
	  k -> RestrictionToSubmoduleARep(R, k, "hom")
	)
      ),
      AMatPerm(sortperm, R.degree, R.char),
      "invertible"
    );
end;


# ARepOps.inductionBlocks( <list-of-degs>, <index> )
#   Given reps R1,..,Rn of H <= G of degrees <list-of-degs>, 
#   (G : H) = <index>, the function calculates a permutation
#   pi, such that the direct sum of R1,..,Rn induced to G 
#   equals the direct sum of the inductions conjugated by pi.

ARepOps.inductionBlocks := function ( L, n )
  local sumL, sum, allblocks, i, j;

  if not( 
    IsList(L) and 
    Length(L) >= 1 and
    ForAll(L, d -> IsInt(d) and d > 0) and 
    IsInt(n) and
    n > 0
  ) then
    Error("usage: ARepOps.inductionBlocks( <list-of-posint>, <posint> )");
  fi;

  # prepare list of cumulated sum of L
  sumL := [ 0 ];
  for i in [1..Length(L) - 1] do
    Add(sumL, L[i] + sumL[i]);
  od;
  sum := Sum(L);

  # list of indices corresponding to the summands
  allblocks := List([1..Length(L)], i -> [ ]);
  for i in [1..Length(L)] do
    for j in [0..n - 1] do
      Append(
        allblocks[i], 
        [j * sum + sumL[i] + 1..j * sum + sumL[i] + L[i]]
      );
    od;
  od;

  return 
    MappingPermListList(
      [1..n * sum], 
      Concatenation(allblocks)
    );
end;


#F ExtensionOnedimensionalAbelianRep( <arep>, <group> )
#F   calculates an extension of an <arep> of degree 1 of 
#F   a subgroup H of <group>, if <group>/kernel( <arep> ) is abelian. 
#F   The extension is a "mon"-arep and chosen to be over the 
#F   smallest possible extension field. No character theory is used. 
#F   This function only works for charcteristic zero.
#F

ExtensionOnedimensionalAbelianRep := function ( R, G )
  local K, H, GH, phi, GHag, CGHag, i, ts, ps, gens, L;

  if not IsARep(R) and IsGroup(G) then
    Error("usage: ExtensionOnedimensionalAbelianRep( <arep>, <group> )");
  fi;
  if not R.degree = 1 then
    Error("<R> must have degree 1");
  fi;
  if not R.char = 0 then
    Error("<R>.char must be 0");
  fi;
  if not ForAll(R.source.theGenerators, h -> h in G) then
    Error("<R>.source must be a subgroup of <G>");
  fi;

  # check, if G/kernel(R) is abelian
  if not IsAbelian(G) then
    K := AsSubgroup(Parent(G), KernelARep(R));
    if not ( IsNormal(G, K) and IsAbelian(G/K) ) then
      Error("<G>/kernel( <R> ) must be abelian");
    fi;
  fi;

  # catch trivial case
  if ARepOps.IsTrivialOneRep(R) then
    return TrivialMonARep(G, 1, R.char);
  fi;

  # calculate the factor group and make it
  # an aggroup to compute the composition series
  H               := AsSubgroup(Parent(G), ShallowCopy(R.source));
  H.theGenerators := ShallowCopy(R.source.theGenerators);
  GH              := G/H;
  phi             := NaturalHomomorphism(G, GH);
  GHag            := AgGroup(GH);
  CGHag           := CompositionSeries(GHag);  

  # representatives and primes
  ts := [ ];
  ps := [ ];
  for i in [1..Length(CGHag) - 1] do
    Add(ts, First(CGHag[i].generators, g -> not g in CGHag[i + 1]));
    Add(ps, Size(CGHag[i])/Size(CGHag[i + 1]));
  od;
  ts := 
    List(
      Reversed(ts), 
      t -> PreImagesRepresentative(phi, Image(GHag.bijection, t))
    );
  ps := Reversed(ps);

  # extend stepwise
  L    := ShallowCopy(MonARepARep(R));
  H    := ShallowCopy(H);
  gens := H.theGenerators;
  for i in [1..Length(ts)] do
    Add(gens, ts[i]);
    H               := Subgroup(G, gens);
    H.theGenerators := ShallowCopy(gens);
    L := 
      ARepByImages(
        ShallowCopy(H), 
        Concatenation(
          L.theImages, 
          [ Mon(
              [RootOfRootOfUnity(MonAMat((ts[i] ^ ps[i]) ^ L).diag[1], ps[i])]
            ) ]
        ),
        "hom"
      );
  od;

  # the extension
  G := GroupWithGenerators(G);
  return 
    ARepByImages(
      G,
      List(G.theGenerators, g -> MonAMat(g ^ L)), 
      "hom"
    );
end;

# funktioniert bisher nur fuer den Fall:
#   <irr-arep> monomial und Erweiteurng monomial

# ExtensionByIntertwiningSpaceNC( <irr-arep>, <supergroup>, <element> )
#   calculates an extension of <irr-arep>, which must be 
#   irreducible, to <supergroup>. The source of <irr-arep>
#   must be normal of prime index in <supergroup>.
#   <element> is an element of <supergroup> generating the
#   factor group. If no extension exists, the function returns 
#   false. According to Clifford, the extension exists, iff
#   <irr-arep> is invariant (up to conjugation) under inner 
#   conjugation by <element>.
#   So far this function only works in the case:
#     <irr-arep> monomial and extension monomial.
#   The function returns the image of the extension on <element>
#   as a mon.


ExtensionByIntertwiningSpaceNC := function( R, G, t )
  local N, p, A, Ap;

  if not IsMonRep(R) then
    return false;
  fi;

  G := GroupWithGenerators(G);
  N := AsSubgroup(Parent(G), R.source);
  p := Index(G, N);

  # catch degree 1 case
  if R.degree = 1 then
    return 
      Mon( 
        [ RootOfRootOfUnity(MonAMat((t ^ p) ^ R).diag[1], p) ]
      );
  fi;

  # matrix generating the onedimensional 
  # intertwining space
  A := IntertwiningSpaceARep(InnerConjugationARep(R, G, t), R)[1];

  # if A is not mon, we return false
  if not IsMonMat(A) then
    return false;
  fi;

  A  := MonAMat(A);
  Ap := A ^ p;

  # correction factor
  return 
    RootOfRootOfUnity(MonAMat((t ^ p) ^ R).diag[1] / Ap.diag[1], p) * A;
end;

# DecompositionPermRepByRegAbNormalSubgroupNC( <arep>, <normal_subgroup> )
#   decomposes the transitive permrep <arep> by using the 
#   normal subgroup N = <normal_subgroup> of R.source, which has the 
#   following properties:
#     1. N/kernel(R) is abelian
#     2. N/kernel(R) is represented regularly by R
#   These properties and the arguments are not checked 
#   by the function.
#

DecompositionMonRep := "defined below";

DecompositionPermRepByRegAbNormalSubgroupNC := function ( R, N )
  local less, permblocks, R1, H, GK, phi, HK, NK, R2, R2NK, 
    con1, DNK, irrs, chars, con2, n, CNK, HKop, phiop, 
    op1, orbs, con3, stabs, tvs, extirrs, irr, stab, 
    allirrs, images, g, i, blocks, con4, con5;

  # a function to compare reps
  # via the character
  less := function ( R1, R2 )
    if not (
      IsARep(R1) and 
      IsARep(R2) and
      IsIdentical(R1.source, R2.source)
    ) then
      Error("<R1> and <R2> must be areps of the same source");
    fi;

    # make the trivial rep the smallest
    if ARepOps.IsTrivialOneRep(R1) then 
      return true;
    elif ARepOps.IsTrivialOneRep(R2) then 
      return false;
    fi;

    # compare degrees
    if R1.degree < R2.degree then
      return true;
    elif R2.degree < R1.degree then
      return false;
    fi;

    # decide by character
    # return CharacterARep(R1) <  CharacterARep(R2);
    return true;
  end;

  # given a list of positive integers L and a 
  # permutation p, permblocks construct a permutation
  # on [1..Sum(L)], which permutes succeding blocks
  # of lengths in L as p does.
  permblocks := function ( L, p )
    local n, B, i;

    n := 0;
    B := [ ];
    for i in L do
      Add(B, [n + 1..n + i]);
      n := n + i;
    od;

    return PermList(Concatenation(Permuted(B, p)));
  end;    

  # decompose R into an induction,
  # H is the stabilizer, 
  # R.source/kernel(R) = HK semidirect NK
  R1  := TransitiveToInductionMonRep(R).rep;
  H   := R1.rep.source;
  GK  := R.source/KernelARep(R);
  phi := NaturalHomomorphism(R.source, GK);
  HK  := Image(phi, H);
  NK  := Image(phi, N);

  # calculate the restriction of R1 to N
  # the restriction is transitive and
  # N/kernel(R) is regularly represented by R
  R2   := RestrictionInductionARep(R1, GroupWithGenerators(N));
  con1 := R2.conjugation;
  R2   := R2.rep.summands[1];

  # view R2 as a rep R2NK of NK
  R2NK := 
    InductionARep(
      TrivialPermARep(GroupWithGenerators(TrivialSubgroup(NK))),
      GroupWithGenerators(NK),
      List(R2.transversal, t -> Image(phi, t))
    );

  # decompose R2NK
  DNK   := DecompositionMonRep(R2NK);
  irrs  := DNK.rep.summands;
  chars := List(irrs, r -> CharacterARep(r).values);
  con2  := DNK.conjugation.element;
  n     := R.degree;

  # the operation of HK on NK via conjugation
  # note, that NK is abelian
  CNK   := List(ConjugacyClasses(NK), c -> c.representative);
  HKop  := Operation(HK, CNK);
  phiop := OperationHomomorphism(HK, HKop);

  # the operation of HKop on the characters
  op1 := function(i, p)
    return Position(chars, Permuted(chars[i], p));
  end;

  # orbits of the irrs under HK
  orbs  := Orbits(HKop, [1..n], op1);
  con3  := 
    AMatPerm(
      MappingPermListList([1..n], Concatenation(orbs))^-1, 
      n
    );

  # stabilizers and transversals of HKop/stabilizers
  stabs := List(orbs, orb -> Stabilizer(HKop, orb[1], op1));
  tvs   := 
    List(
      orbs, 
      orb -> 
        List(
          orb,
          i -> RepresentativeOperation(HKop, orb[1], i, op1)
        )
    );

  # translation from HKop to HK
  stabs := List(stabs, s -> PreImage(phiop, s));
  tvs   := List(tvs, tv -> List(tv, t -> PreImage(phiop, t)));

  # translation from HK to H
  stabs := 
    List(
      stabs, 
      s -> PreImage(phi, s)
    );
  tvs   := List(tvs, tv -> List(tv, t -> PreImagesRepresentative(phi, t)));

  # add N to the stabs to obtain the stabilizer in R.source
  stabs := 
    List(
      stabs, 
      s -> 
        GroupWithGenerators(
          Subgroup(R.source, Concatenation(s.generators, N.generators))
        )
    );

  # extend first irrs in orbit to their stabilizers
  # note, that all irrs are monomial
  extirrs := [ ];
  for i in [1..Length(orbs)] do
    irr    := irrs[orbs[i][1]];

    # translate irr to a rep of N
    irr := 
      ARepByImages(
        N,
        List(N.theGenerators, g -> MonAMat(Image(phi, g)^irr)),
        "hom"
      );

    stab   := stabs[i];
    images := [ ];
    for g in stab.theGenerators do
      if g in N then
        Add(images, MonAMat(g^irr));
      elif g in H then
        Add(images, Mon( [ AMatOps.OneNC(R.char) ] ));
      else
        Error("check algorithm");
      fi;
    od; 
    Add(
      extirrs,
      ARepByImages(GroupWithGenerators(stab), images, "hom")
    );
  od;

  # calculate irrs of R
  allirrs := 
    List(
      [1..Length(extirrs)], 
      i -> MonARepARep(InductionARep(extirrs[i], R.source, tvs[i]))
    );

  # sort allirrs, note, that the irrs
  # are pairwise inequivalent
  blocks := List(allirrs, r -> r.degree);
  con4   := [1..Length(allirrs)];
  SortParallel(
    allirrs,
    con4,
    less
  );
  con4 := 
    AMatPerm(
      permblocks(blocks, PermList(con4) ^ -1) ^ -1,
      R.degree,
      R.char
    );

  # precompute conjugation for fast simplifying
  con5 := con1 ^ -1 * con2 * con3 * con4;
  con5.isMonMat := false;

  return 
    ConjugateARep(
      DirectSumARep(allirrs),
      PowerAMat(
        SimplifyAMat(con5),
        -1,
        "invertible"
      )
    );
end;


# FindSparseInvertible( <amats> )
#   finds a sparse element of the (matrix) vector space
#   given by the base <amats> which is a list of amats.
#   It assumes, that <amats> cosist of very sparse matrices
#   which have entries at different positions.
#   The function forms sums of subsets of <amats> of 
#   increasing length and checks for full rank.
#   If an invertible is found, it is returned as a
#   structured amat, else false is returned.

FindSparseInvertible := function ( L )
  local dims, char, n, k, found, inds, pos, A, D;

  # check argument
  if not ( IsList(L) and ForAll(L, IsAMat) ) then
    Error("<L> must be a list of amats");
  fi;

  dims := L[1].dimensions;
  char := L[1].char;
  
  if not ForAll(L, A -> A.dimensions = dims and A.char = char) then
    Error("amats in <L> must have common size and char");
  fi;

  n     := Length(L);
  k     := 1;
  found := false;

  # check whether sum of k elements is invertible
  # and increase k
  repeat
    inds := Combinations([1..n], k);
    pos  := 
      PositionProperty(
        inds, 
        c -> RankMat(Sum(List(Sublist(L, c), MatAMat))) = dims[1]
      );
    if pos <> false then
      A     := Sum(Sublist(L, inds[pos]), MatAMat);
      found := true;
    fi;
    k := k + 1;
  until k > n or found;

  # return structured amat or false
  # we don't use AMatSparseMat, since we know,
  # that the matrix is invertible
  if found then
    D := DirectSummandsPermutedMat(A);
    return 
      SimplifyAMat(
        AMatPerm(D[1], dims[1], char) *
        DirectSumAMat(List(D[2], AMatMat)) *
        AMatPerm(D[3],  dims[1], char)
      );
  else
    return false;
  fi;
end;


# FindRandomInvertible( <amats> )
#   finds an invertible element of the (matrix) vector space
#   given by the base <amats> which is a list of amats.
#   The function forms random linear combinations of <amats>
#   until the result is invertible.

FindRandomInvertible := function ( L )
  local dims, char, n, A;

  # check argument
  if not ( IsList(L) and ForAll(L, IsAMat) ) then
    Error("<L> must be a list of amats");
  fi;

  dims := L[1].dimensions;
  char := L[1].char;
  
  if not ForAll(L, A -> A.dimensions = dims and A.char = char) then
    Error("amats in <L> must have common size and char");
  fi;

  n     := Length(L);
  repeat
    A := 
      AMatMat(
        Sum(
          List(
            L, 
            m -> Random([1..n])*MatAMat(m)
          )
        )
      );
  until IsInvertibleMat(A);

  return A;
end;

# InfoLatticeDec prints (little) info,
# InfoLatticeDec1 is for debugging purposes.
if not IsBound(InfoLatticeDec) then
  InfoLatticeDec := Ignore;
fi;
if not IsBound(InfoLatticeDec1) then
  InfoLatticeDec1 := Ignore;
fi;

#F DecompositionMonRep( <arep> [, <hint> ] )
#F   decomposes the monomial <arep> over characteristic zero
#F   into irreducibles and determines a highly structured 
#F   decomposition matrix A. 
#F   More precisely <arep> is decomposed as
#F     <arep> = 
#F       ConjugateARep(
#F         DirectSumARep(R_i, i = 1..k),
#F         A ^ -1
#F       ) 
#F   where all R_i are irreducible. Since the algorithm is
#F   not able to decompose every monomial <arep>, false
#F   is returned in the case of failure.
#F   If <arep> is transitive, then the R_i are ordered by 
#F   degree with the trivial onerep being the smallest rep 
#F   (of course this is only a partial ordering).
#F   In this case (transitivity) equivalent R_i are equal.
#F   If the <hint> "noOuter" is supplied, the decomposition
#F   is executed without any decomposition into an outer
#F   tensor product. Supplying the hint may speed up the
#F   calculation for the price of a suboptimal (concerning the 
#F   structure) decomposition matrix.
#F   Note, that the decomposition matrix A is accessible by 
#F   A = R.conjugation.element. A is simplified by the 
#F   function SimplifyAMat.
#F   The structure of A represents a fast algorithm for 
#F   multiplication with A.
#F   Note that not every monomial representation can be decomposed,
#F   but at least any monomial rep of any solvable group.
#F   In the case that R is a regular representation of a solvable
#F   group the matrix A represents a fast Fourier transform 
#F   for R.source.
#F

# The algorithm: (R is a monomial representation of G)
#
#   Case 1: R is already irreducible.
#
#     The identity matrix of suitable size decomposes. 
#     Note that a permrep is irreducible iff degree = 1, 
#     but monreps also may be irreducible with larger degree.
#
#   Case 2: R is not transitive.
#
#     Conjugate orbits with a permutation in a row and recurse
#     with the transitive constituents.
#
#   Case 3: R is a double transitive permrep
#
#     SOR decomposes. The irreducibles are found by conjugation.
#
#   Case 4: R is a permrep of prime degree 
#
#     Theorem: Let R an induced monomial rep of prime degree p, 
#       and G = G/Kernel(R).
#       Then R is double transitive or G contains a normal
#       regular (up to conjugation by a monomial matrix)
#       Zp, which is exactly the p-Sylow subgroup of G.
#
#     Lemma: Let R be a monrep, and H an abelian  subgroup of 
#        R/Kernel(R) represented regularly by R. Then any 
#        decomposition matrix of (R restriction H) is one of R
#
#     a. Conjugate R by a diagonal matrix to be an induced
#        rep R1 = (L induction G).
#
#     b. Calculate the p-Sylow subgroup Zp' of G/Kernel(R)
#        and the corresponding Zp in G.
#
#     c. Conjugate (R1 restriction Zp) by a monomial matrix onto
#        R2 = (1 induction Zp) where 1 is the onerep of Kernel(R).
#        
#     d. Conjugate the image of any generator of Zp onto (1, ..,p),
#        now the DFT_p decomposes and we obtain a decomposition
#        marix of R by the lemma.
#
#     e. In general, the irreducibles must be constructed by
#        conjugation.
#
#     The case, where R is a permrep and R/Kernel(R) ~= Zp is
#     treated seperately, since in this case the irreducible
#     components of R can be constructed without conjugating
#     R by the decomposition matrix.
#
#   Case 5: G/Kernel(R) is abelian.
#
#     Lemma: Let R be a transitive monrep and G/Kernel(R) abelian.
#       Then R can be decomposed (by a diagonal matrix) into 
#       the (inner tensor-) product of a onedimensional rep L and 
#       a permrep. As L serves any extension of the onedimensional 
#       rep from which R is induced.
#
#     a. If R is no permrep then make R an induction 
#          R = (L induction by T G)^D1
#        and extend L to L1 with the function 
#        ExtensionOnedimensionalAbelianRep above.
#        Recurse with the permrep 
#          (L induction by T G)^diag(L1(t), t in T)
#     b. If R is a permrep and G/Kernel(R) a prime power
#        return the structured p-power DFT.
#     c. There is no other case, since any permrep of an abelian
#        group can be decomposed as an outer tensor product
#        of permreps of cyclic groups of p-power order.
#        
#   Case 6: R is a primitive permrep and a minimal normal subgroup of
#     G/Kernel(R) is solvable.
#
#     Theorem: Let R be a primitive monrep and G = G/Kernel(R)
#       contain a minimal normal subgroup N, that is solvable.
#       Then N ~= Zp^k and N is represented regularly by R. If
#       S is a the stabilizer of one point w.r.t. R, then G = SN.
#
#     a. Calculate a minimal normal subgroup N of G and test 
#        whether it is abelian!
#
#     b. decompose R by restriction to N
#
#   Case 7: R is a conjugated outer tensorproduct.
#
#     a. Conjugate R by a monomial matrix to be an outer 
#        tensorproduct R = (R1 # .. # Rk).
#
#     b. Recurse with the factors. The irreducibles of R are 
#        obtained by constructing all outer tensorproduct of
#        tuples of irreducibles of the factors.
#
#     c. The tensorproduct of the decomposition matrices of
#        the factors decomposes R, but only up to a permutation.
#
#   Case 8: "induction recursion"
#           There is a normal subgroup N of prime index p 
#           between G and the stabilizer S of R.
#
#     a. Conjugate R by a monomial matrix onto
#          RSNG = (RS induction N) induction G
#     b. Conjugate R1 by a monomial matrix onto
#          RSNbyTtoG = (RS induction N) induction_T G,
#        where T is of the form T = [t^0, .., t^(p - 1)]
#     c. Decompose RSN with A into irreducibles (recursion)
#          RSN ^ A = RSNdec
#
#     Use Clifford's theorem:
#     Theorem: Let N be normal in G of prime index p and r be 
#       an irreducible rep of N. For any t in G\N,
#       T = {t ^ i | i = 0..p - 1} is a transversal of G\N.
#       Let r ^ g: x -> r(gxg^-1) for g in G denote the 
#       (inner) conjugated rep of N by g. Of course
#       all the r ^ (t ^ i) are irreducible.
#       Then exactly one of the following two cases applies:
#       1. The r ^ (t ^ i) are pairwise inequivalent. Then
#          (r induction G) is irreducible and (r induction_by_T G)
#          is an extension of the direct sum of the r ^ (t ^ i).
#       2. All r ^ (t ^ i) are equivalent. Then there are exactly
#          p pairwise inequivalent extensions of r to G given by
#          l ^ i * r1 where r1 is one extension and l ^ i are the 
#          p characters of G/N. (r induction G) decomposes into those.
#     
#     d. Conjugate RSNdec by a permutation P to order the irreducibles, 
#        such that first come the extendable irreducibles, and then 
#        the not extendables, called inducables, collected in groups 
#          r...r, r^t...r^t, ... ...,r^(t^(p-1))..r^(t^(p-1))
#        up to equivalence!
#        Note that:
#          1. To decide whether a given irreducible r belongs to
#             extendables or inducables it is sufficient to check
#             whether r ^ t ~= r.
#          2. A strong necessary condition for the equation above
#             is, that the ordered multisets of the character values
#             have to be equal.
#          3. t induces a permutation tperm on the conjugacy classes
#             of N. The following lemma yields a fast method to
#             decide, whether r ^ t ~= r.
#             Lemma: if t in G permutes the conjugacy classes of N 
#               as the permutation p, then
#                 chi.values ^ p = (chi ^ t).values,
#               where chi is a character of N.
#          4. If G is abelian, then all irrs of N are 
#             extendables (clear, since an abelian group has only
#             onedimensional irrs).
#          5. It is convenient to precompute the collected character
#             values to obtain a fast decision for inner conjugacy.
#          6. If an inducable is contained in RSNdec, then it is
#             not necessary, that all inner conjugates are present
#             or have the same multiplicity.
#     e. Conjugate the r_i, such that equality in the equation above 
#        holds. This means, that inner conjugates are now equal! to
#        inner conjugates. Here, the conjugating matrices are found by solving
#        linear equations. The direct sum of the conjugations 
#        prepended by an identity matrix on the extendables yield
#        a matrix D.
#        The matrix A * P * D is a decomposition matrix for RSN, too.
#     f. Calculate a permutation matrix P1 conjugating the induction
#        of the decomposed RSN^APD onto the direct sum of the inductions
#        of the blocks. Here, all extendables are viewed as one block, 
#        every inducable yields a single block.
#     g. Extend the direct sum of all extendables R1 to R1ext. 
#        There are two ways to do that
#          1. Minkwitz's extension formula which needs a summation 
#             over N.
#          2. Calculating the intertwining space (linear equations)
#             of r and r ^ t according to the following lemma
#             Lemma: Let N normal subgroup of index p in G, 
#               T = {t ^ 0, .., t ^ (p - 1)} a transversal of G/N 
#               and r an irreducible rep of N with r ~= r ^ t. 
#               According to Clifford r has an extension R to G.
#               Let M in Int(r ^ t, r), which is unique up to a 
#               scalar. Then r(t ^ p) = c * M ^ p. Let cp be a pth
#               root of c and define
#                 R(t ^ i * n) = cp ^ i * r(n), for n in N.
#               R extends r to G.
#        Method 2 is better if r has small degree.
#     h. Calculate the image timage of t under that extension
#     i. Calculate the p irrs lin_1, ..,lin_p of G coming from G/N.
#     j. Now the component of R given by the induction of 
#        all extendables decomposes by
#          DirectSumAMat(List([0..(p - 1)], i -> timage ^ i)) *
#	   TensorProductAMat(
#	     DFTAMat(p, R.char), 
#	     IdentityPermAMat(RSNG.rep.rep.degree, R.char)
#	   )
#        into directsum lin_i * R1ext, i = 1..p.
#        using the lemma
#        Lemma: H <= G, R rep of H with extension R1 to G, 
#          T transversal of H\G. Then
#          (R induction_T G)^D = (1_H induction_T G) tensor R1,
#            D = directsum R1(t), t in T>
#      k. The irreducibles of R are given by the lin_i * R1ext
#         and the inductions of the inducables
#      l. Make the induction of inner conjugated inducables 
#         equal using the lemma
#         Lemma: r ^ (t ^ i) induction_T = (r induction_T G) ^ A,
#         where A is defines as
#           A = (1..p) ^ (-i) tensor one_deg(r) *
#               one_((p - i) * deg(r)) dirsum 
#               one_i tensor r(t ^ p)
#      m. Sort the irreducibles and calculate the sorting 
#         permutation.
#
#   Case 9: "switch recursion"
#           G is solvable and there is a normal subgroup N of G
#           of prime index p which does not contain the 
#           stabilizer S of R.
#
#     a. Calculate N with the property mentioned. Since we have
#        passed Case 8, the second element of any composition 
#        series of G can be chosen. Choose t in G\N.
#     b. Conjugate R by a monomial matrix onto
#          RSG = (RS induction G), 
#        such that
#          RN = RSG restriction N = 
#            (RS restriction (S intersection N)) induction N.
#     c. Decompose RN with A into irreducibles
#          RN ^ A = irrs
#     d. Use Clifford's Theorem (see above, "induction recursion")
#        as in the "induction recursion" to decide which irrs
#        are extendable to G and which not. Latter are called inducables.
#     e. Determine a permutation matrix P, such that the irrs
#        are ordered: first the extendables, then the inducables
#        grouped as
#          (r^n), (r^t)^n, .., (r^(t^(p-1)))^n,
#        with common multiplicity n.
#     f. Calculate a blockdiagonal matrix D, making the inner
#        conjugates equal to inner conjugates.
#        A*P*D is a decomposition matrix for RN, too.
#     g. Decompose (partially) RSG with A. This is expensive and the
#        reason for the great difference in speed between the
#        "induction recursion" and the "switch recursion".
#          RSG ^ A = summandsDR
#        The summands correspond to the groups of equal extendables
#        resp. inner conjugate inducables.
#        The remaining task is the decomposition of summandsDR.
#     h. Case 1: summandsDR[i] belongs to a group of extendables
#          Determine and decompose the character of summandsDR[i] 
#          and compute a decomposed prototype extsummand by extending the 
#          extendables in the right way. If the extension is not homogeneous,
#          (check by the scalar product of the the character with itself)
#          then the chartable is computed.
#          Now summandsDR[i] has to be conjugated onto extsummand.
#          Since the restrictions of summandsDR[i] and extsummand to N
#          are equal, any such matrix is of the form
#            (B tensor one_d), 
#          where d is the degree of any extendable.
#          To attack directly B we:
#            1. Choose g in G\N with the property that the character
#               of one (and hence any) extension of an extendable
#               on g is <> 0.
#            2. Calculate the image of g under summandsDR[i] and 
#               extsummand and calculate the partial trace, i.e.
#               divide both into (d x d) submatrices, wherefrom
#               we take the trace, see lemma.
#
#               Lemma: (B tensor one_d) in Int(summandsDR[i], extsummand)
#                 <=>  (B tensor one_d) in Int(summandsDR[i](g), extsummand(g))
#                 <=>  B in Int(T(summandsDR[i](g)), T(extsummand(g)),
#               where T maps a (nd x nd) matrix onto a (n x n) matrix
#               by dividing it into (d x d) matrices and calculating the 
#               trace of these.
#         
#            3. To find B use first the function FindSparseInvertible to get
#               something sparse and maybe structured. Else use
#               FindRandomInvertible.
#         Case 2: summandsDR[i] belongs to a group of inducables ordered as
#                   (r^n), (r^t)^n, .., (r^(t^(p-1)))^n,
#                 A matrix (B1 dirsum .. Bp) tensor one_d transforms
#                 this sum into (r^n) induction G which is decomposed by
#                 a permutation matrix into (r induction G)^n.
#                 Bi can be read of summandsDR[i](t^k), k = 0..p-1:
#                 the (k+1)th block of size nd x nd in the first block
#                 row of this image is Bl tensor one_d.
#


DecompositionMonRep := function ( arg )
  local 
    G,              # R.source
    less,           # function to compare irr. reps
    monormatify,    # function to flatten reps
    permblocks,     # function to calculate blockpermutations
    partialTrace,   # function for the partial trace
    R,              # the rep
    hint,           # the hint
    K,              # the kernel
    AGK, AGK1,      # aggroups
    psi,            # corresponding bijection
    gens,           # minimal generating set
    R1, R2,         # R decomposed
    Ds,             # List of decomposed reps
    irrs, irrs1,    # irreducible reps
    GK,             # R.source/Kernel(R)
    phi,            # hom R.source -> GK
    ZpK,            # normal cyclic subgroup of GK of prime order p
    irr,            # an irredcuible rep
    ind,            # index of a generator
    gen,            # generator of a group
    genim,          # images of generators
    im,             # image of a generator
    imagegrp,       # group generated by the images
    twiddle,        # function for twiddles in primepower case
    Ts,             # twiddle matrices
    Zpk,            # cyclic group of order p ^ k
    Zp,             # preimage of ZpK under phi
    Sn,             # symmetric group
    M,              # matrix
    L,              # onedimensional rep
    Lext,           # extension of onedimensional rep to G
    D,              # decomposed rep
    blocks,         # blocks of decomposition
    nrfacs,         # number of tensorfactors
    degs,           # degrees of irrs
    perm,           # permutation
    kbss,           # kbs's of decomposed tensorfactors
    kbs1, kbs2,     # kbs's
    sum1, sum2,     # off-
    sum3,           # sets
    n, d, e, c,     # counter
    inds,           # index vectors
    RSG,            # RSG = R, RSG = (L_S induction G) ^ con1 
    con1,           # a diagonal matrix
    N,              # normal subgroup of R.source, index is prime
    RSNG,           # RSNG = RSG.rep, 
                    # RSNG = ((L_S induction N) induction G) ^ con2
    con2,           # a monomial matrix
    p,              # (R.source : N)
    t,              # element of R.source\N
    T,              # T = {t ^ 0, ..,t ^ (p - 1)}, TV of R.source\N
    RSNbyTtoG,      # RSNG = RSNbyTtoG, upper induction with TV T
    con3,           # RSNbyTtoG.conjugation, a monomial matrix
    testelms,
    perms,
    chars,
    char,
    char1,
    collirrs,       # irrs collected   
    rn,             # pair in collirrs 
    pos,            # position of an irr
    extendables,    # list of extendable r in irrs
    inducables,     # list of lists of the form 
                    #   [r_i | i in [0..p-1], r_i ~= r_1 ^ (t ^ i) ],
                    # with those r in irrs being not extendable
    extpermlist,    # indices of extendables in irrs
    indpermlist,    # indices of inducables in irrs
    ccs,            # conjugacy classes of N
    cc,             # conjugacy class in N
    tperm,          # permutation of t on ccs
    i, j, k, l,     # counter
    stop,           # boolean to exit a loop
    sortperm,       # permutation ordering irrs 
    extdeg,         # entire degree of extendables
    perm1, perm2,   # sorting
    perm3,          # permutations
    extrs,          # extended extendable
    extextendables, # extended extendables
    indrs,          # induced inducable
    indinducables,  # induced inducables
    indextendables, # induced extendables
    pcycle,         # p-cycle (1..p)
    corrperm,       # perm correcting induced inducables
    corrpermdegs,   # corresponding degrees
    corrdiag,       # blocks to correct induced inducables
    tprs,           # t^p evaluated at an inducable
    mult,           # multiplicity of a group of inducables
    indcons,        # making inner conjugates
    alldegs,        # blocks of decomposed lower rep
    rs,             # list of equivalent irrs
    ers,            # monormatified rs
    lins,           # onedim reps of R.source/N
    g, g1,          # group elements
    primeroots,     # list of p-th roots of unity
    allcons,        # list of matrices conjugating r_i onto r_1 ^ (t ^ i)
    lrs,            # l in lin times rs in extextendables
    cons,           # matrix conjugating r_i onto r_1 ^ (t ^ i)
    con4,           # permutation matrix
    con5,           # direct sum of allcons
    con6,           # permutation matrix
    con7,           # matrix with few blocks
    con8,           # permutation matrix
    con9,           # permutation matrix to sort irrs
    allirrs,        # irrs of induction
    allirrs1,       # monormatified irrs
    r,              # element in irrs
    chi,            # character of r
    timage,         # t ^ RSNdecext
    timage1,        # partial image of t
    con01, con02,   # correcture matrices 
    con03, con04,   # constructed for
    con05, con06,   # fast simplifying
    NK, NKs,        # normal subgroups of R.source/kernel(R)
    RN,             # restriction of R to N
    DR,             # R decomposed with the dec. matrix of N
    summandsDR,     # the summands of DR
    deg,            # degree of an irr
    chiirrs,        # character
    s,              # scalar product of two characters
    chin,           # character
    rchin,          # rep with character
    extsummand,     # direct sum of irrs with homogenous restriction
    intsummand,     # intertwining matrix
    extind,         # extended inducable
    extinds,        # sum of extended inducables
    M1, M2;         # matrices

  # a function to compare reps
  # via the character
  less := function ( R1, R2 )
    if not (
      IsARep(R1) and 
      IsARep(R2) and
      IsIdentical(R1.source, R2.source)
    ) then
      Error("<R1> and <R2> must be areps of the same source");
    fi;

    # make the trivial rep the smallest
    if ARepOps.IsTrivialOneRep(R1) then 
      return true;
    elif ARepOps.IsTrivialOneRep(R2) then 
      return false;
    fi;

    # compare degrees
    if R1.degree < R2.degree then
      return true;
    elif R2.degree < R1.degree then
      return false;
    fi;

    # decide by character
    # return CharacterARep(R1) <  CharacterARep(R2);
    return true;
  end;

  # given a list of positive integers L and a 
  # permutation p, permblocks construct a permutation
  # on [1..Sum(L)], which permutes succeding blocks
  # of lengths in L as p does.
  permblocks := function ( L, p )
    local n, B, i;

    n := 0;
    B := [ ];
    for i in L do
      Add(B, [n + 1..n + i]);
      n := n + i;
    od;

    return PermList(Concatenation(Permuted(B, p)));
  end;    


  # a function to convert an arep to a "mon"-arep
  # if possible, else to a "mat"-arep
  monormatify := function ( R )
    if IsMonRep(R) then
      return MonARepARep(R);
    fi;
    return MatARepARep(R);
  end;

  # a function to calculate the "partial trace",
  # given a square matrix of degree divisible by d,
  # the matrix is divided into d x d matrices,
  # then these are subsituted by their trace
  partialTrace := function( M, d )
    local n, Mtr, i, j;

    n   := DimensionsMat(M)[1]/d;
    Mtr := List([1..n], i -> [ ]);
    for i in [1..n] do
      for j in [1..n] do
	Mtr[i][j] := 
          Sum(List([1..d], l -> M[(i - 1)*d + l][(j - 1)*d + l]));
      od;
    od;

    return Mtr;
  end;

  # here starts the function
  # ------------------------

  # dispatch
  if Length(arg) = 1 then
    R    := arg[1];
    hint := "no hint";
  elif Length(arg) = 2 then
    R    := arg[1];
    hint := arg[2];
  else
    Error("usage: DecompositionMonRep( <arep> [, <hint> ] )");
  fi;

  # check arguments
  if not IsARep(R) then
    Error("usage: DecompositionMonRep( <arep> )");
  fi;

  # avoid cases with char <> 0
  if R.char <> 0 then
    Error("<R>.char must be zero");
  fi;
  if not hint in ["noOuter", "no hint"] then
    Error("hint must be \"noOuter\"");
  fi;
  if not IsMonRep(R) then
    Error("<R> must be a monrep");
  fi;

  G := R.source;
  if not IsSolvable(G) then
    return false;
  fi;
  InfoLatticeDec1("#I+ check if faithful\n");

  # R is not faithful
  # -----------------
  if not IsFaithfulARep(R) then

    # compute kernel and factor group
    K   := KernelARep(R);
    GK  := G/K;
    phi := NaturalHomomorphism(G, GK);
    InfoLatticeDec(
      "#I not faithful: ", 
      Size(G), " -> ", Size(G)/Size(K), " (group sizes)\n"
    );

    # at the moment only the solvable case is of interest
    if IsSolvable(G) or IsSolvable(GK) then
    
      # compute an aggroup for GK with
      # minimal generating set
      # note that AGK and AGK1 are equal
      InfoLatticeDec1("#I+ construct aggroup\n");
      AGK  := AgGroup(GK);
      psi  := AGK.bijection;
      gens := MinimalGeneratingSet(AGK);
      if Length(gens) = 0 then
        gens := [AGK.identity];
      fi;
      AGK1 := GroupWithGenerators(gens);

      # construct representation of AGK1
      R1 := 
        ARepByImages(
          AGK1,
          List(
            gens,
            g -> 
              MonAMat(
                PreImagesRepresentative(phi, Image(psi, g)) ^ R
              )
          ),
          "faithful"
        );

      # decompose
      D := DecompositionMonRep(R1);

      # change to G, produce monreps if possible
      InfoLatticeDec1("#I+ translate to original group\n");
      L    := [ ];
      gens := List(G.theGenerators, g -> PreImage(psi, Image(phi, g)));
      for r in D.rep.summands do
        if IsMonRep(r) then
          Add(
            L,
            ARepByImages(
              G,
              List(gens, g -> MonAMat(g ^ r)),
              "hom"
            )
          );
        else
          Add(
            L,
            ARepByImages(
              G,
              List(gens, g -> MatAMat(g ^ r)),
              "hom"
            )
          );
         fi;
      od;

      # return result
      InfoLatticeDec("#I rep of degree ", R.degree, " completed\n");
      return
        ConjugateARep(
          DirectSumARep(L),
          D.conjugation
        );
    fi;
  fi;

  # R irreducible
  # -------------
  if IsIrreducibleARep(R) then
    InfoLatticeDec("#I irreducible\n");
    return 
      ConjugateARep(
        DirectSumARep(monormatify(R)),
        IdentityPermAMat(R.degree, R.char) ^ -1,
        "invertible"
      );
  fi;

  # R is not transitive: orbit decomposition
  # ----------------------------------------
  if not IsTransitiveMonRep(R) then
    R1   := OrbitDecompositionMonRep(R);
    InfoLatticeDec("#I orbit decomposition: ", R.degree, " -> ");
    InfoLatticeDec(R1.rep.summands[1].degree);
    for i in [2..Length(R1.rep.summands)] do
      InfoLatticeDec(" + ", R1.rep.summands[i].degree);
    od;
    InfoLatticeDec(" (degrees)\n");
    if hint = "noOuter" then
      Ds   := List(R1.rep.summands, r -> DecompositionMonRep(r, "noOuter"));
    else
      Ds   := List(R1.rep.summands, DecompositionMonRep);
    fi;

    if false in Ds then
      return false;
    fi;

    InfoLatticeDec1("#I+ simplifying decomposition matrix\n");
    con1 := 
      SimplifyAMat(
        R1.conjugation ^ -1 * 
        DirectSumAMat(List(Ds, r -> r.conjugation.element))
      );

    InfoLatticeDec("#I rep of degree ", R.degree, " completed\n");
    return
      ConjugateARep(
        DirectSumARep(
          List(
            Concatenation(List(Ds, r -> r.rep.summands)),
            monormatify
          )
        ),
        PowerAMat(con1, -1, "invertible"),
        "invertible"
      );
  fi;

  # R.source is abelian: recurse with permrep contained
  # ---------------------------------------------------
  if IsAbelian(G) and not IsPermRep(R) then

    # we extract the onedimensional rep
    # and recurse with the permrep
    # (note that we have prime power degree,
    # otherwise it would be tensor decomposable)
    # L induction G -> (1 induction G) tensor Lext

    InfoLatticeDec("#I proper abelian monrep\n");

    # extract permrep contained in R
    R1 := TransitiveToInductionMonRep(R);

    # extend onedimensional rep
    con1 := R1.conjugation;
    Lext := ExtensionOnedimensionalAbelianRep(R1.rep.rep, G);
    con2 := 
      DirectSumAMat(
        List(R1.rep.transversal, t -> SimplifyAMat(t ^ Lext))
      );

    # decompose the permrep
    D := 
      DecompositionMonRep(
        InductionARep(
          TrivialMonARep(R1.rep.rep.source, 1, R.char),
          G,
          R1.rep.transversal
        )
      );

    con3 := con1 ^ -1 * con2 * D.conjugation.element;
    con3.isMonMat := false;

    # irrs of r
    # note, that the trivial onerep can not
    # occur among the irrs, since Lext is not 
    # the onerep, hence the irrs are sorted
    irrs := 
      List(
        D.rep.summands, 
        r -> MonARepARep(InnerTensorProductARep(Lext, r))
      );

    InfoLatticeDec1("#I+ simplifying decomposition matrix\n");
    con3 := SimplifyAMat(con3);
    InfoLatticeDec("#I rep of degree ", R.degree, " completed\n");
    return 
      ConjugateARep(
        DirectSumARep(irrs),
        PowerAMat(con3, -1, "invertible")
      );
  fi;

  # outer tensorproduct: recurse with factors
  # -----------------------------------------
  # in the abelian case we always want to try this decomposition
  # because its fast (not all normal subgroups are computed)
  if hint <> "noOuter" or IsAbelian(G) then
    R1     := OuterTensorProductDecompositionMonRep(R);
    nrfacs := Length(R1.rep.factors);
    if nrfacs > 1 then
      InfoLatticeDec(
	"#I outer tensorproduct: ",
	R.degree, " -> " 
      );
      for i in [1..nrfacs - 1] do
	InfoLatticeDec(R1.rep.factors[i].degree);
	InfoLatticeDec(" * ");
      od;
      InfoLatticeDec(R1.rep.factors[nrfacs].degree);
      InfoLatticeDec(" (degrees)\n");

      # calculate permutation conjugating irreducibles
      # of the tensorproduct in a row. Note, that the
      # irreducibles are ordered lexicographically with
      # respect to the ordering of the irreducibles of
      # the factors
      Ds := List(R1.rep.factors, r -> DecompositionMonRep(r));
      if false in Ds then
        return false;
      fi;
      InfoLatticeDec1("#I+ factors decomposed\n");

      # kbs of the factors
      kbss := [ ];
      for i in [1..nrfacs] do
	kbss[i] := [ ];
	degs    := List(Ds[i].rep.summands, r -> r.degree);
	n       := 0;
	for d in degs do
	  Add(kbss[i], [n + 1..n + d]);
	od;
      od;

      perm1 := ( );
      L     := kbss[nrfacs];
      sum1  := R1.rep.factors[nrfacs].degree;

      # sort from the rear
      for n in [nrfacs - 1, nrfacs - 2..1] do
	kbs1 := kbss[n];
	kbs2 := List(L, l -> List(l, i -> i ^ perm1));
	L    := [ ];
	sum2 := 0;
	for d in kbs1 do
	  sum3 := 0;
	  for e in kbs2 do
	    Add(
	      L, 
	      sum2 + 
	      sum3 +
	      Concatenation(
		List(
		  [0..Length(d) - 1], 
		  j -> [j * sum1 + 1..j * sum1 + Length(e)]
		)
	      )
	    );
	    sum3 := sum3 + Length(e);
	  od;
	  sum2 := sum2 + Length(d) * sum1;
	od;
	sum1 := sum1 * R1.rep.factors[n].degree;
	perm1 := 
	  perm1 * 
	  TensorProductPerm(
	    [R1.degree/sum1,                            sum1], 
	    [            (), PermList(Concatenation(L)) ^ -1]
	  );
      od;

      # collect irrs of the factors by equality
      collirrs := [ ];
      blocks   := [ ];
      for i in [1..nrfacs] do
	irrs        := Ds[i].rep.summands;
	collirrs[i] := [ ];
	blocks[i]   := [ ];
	j           := 0;
	pos         := 0;
	while j < Length(irrs) do
	  j := j + 1;
	  r := irrs[j];
	  n := 1;
	  while j < Length(irrs) and irrs[j + 1] = r do
	    n := n + 1;
	    j := j + 1;
	  od;
	  Add(collirrs[i], [r, n]);
	  Add(blocks[i], [pos + 1..pos + n]);
	  pos := pos + n;
	od;
      od;

      # perm to bring equivalent tensor products together
      inds  := 
	Cartesian(
	  List(
	    [1..nrfacs], 
	    i -> [1..Length(Ds[i].rep.summands)]
	  )
	);
      perm2 := [ ];
      for l in Concatenation(List(Cartesian(blocks), Cartesian)) do
	Add(perm2, Position(inds, l));
      od;
      perm2 := PermList(perm2) ^ -1;

      blocks := 
	List(
	  inds, 
	  l -> 
	    Product(
	      List([1..nrfacs], i -> Ds[i].rep.summands[ l[i] ].degree)
	    )
	);

      # construct irrs
      InfoLatticeDec1("#I+ ");
      for i in [1..nrfacs - 1] do
	InfoLatticeDec1(Length(Ds[i].rep.summands), " * ");
      od;
      InfoLatticeDec1(Length(Ds[nrfacs].rep.summands), " many irrs\n");
      perm2 := permblocks(blocks, perm2) ^ -1;
      irrs  := [ ];
      for l in Cartesian(collirrs) do
	Add(
	  irrs, 
	  [ monormatify(
	      OuterTensorProductARep(G, List(l, p -> p[1]))
	    ),
	    Product(List(l, p -> p[2]))
	  ]
	);
      od;

      # sort irrs
      InfoLatticeDec1("#I sorting irrs\n");
      perm3  := [1..Length(irrs)];
      blocks := List(irrs, r -> r[1].degree * r[2]);
      SortParallel(
	irrs, 
	perm3, 
	function(r1, r2) return less(r1[1], r2[1]); end
      );
      perm3 := permblocks(blocks, PermList(perm3) ^ -1) ^ -1;
      con1  := 
	R1.conjugation ^ -1 *
	SimplifyAMat(TensorProductAMat(List(Ds, d -> d.conjugation.element))) *
	AMatPerm(perm1 * perm2 * perm3, R.degree, R.char);

      # set the field .isMonMat, because it is 
      # expensive to check
      con1.isMonMat := 
	ForAll(List(Ds, d -> d.conjugation.element), IsMonMat);

      InfoLatticeDec1("#I+ simplifying decomposition matrix\n");
      con1 := SimplifyAMat(con1);

      InfoLatticeDec("#I rep of degree ", R.degree, " completed\n");
      return 
	ConjugateARep(
	  DirectSumARep(
	    Concatenation(
	      List(irrs, r -> List([1..r[2]], i -> r[1]))
	    )
	  ),
	  PowerAMat(con1, -1, "invertible"),
	  "invertible"
	);
    fi;
  fi;

  # R.source is abelian and R is a permrep: use formula
  # ---------------------------------------------------
  if IsAbelian(G) and IsPermRep(R) then

    # now R is a regular rep of a Z_(p^k)
    # (otherwise it would be tensor decomposable)
    R1       := PermARepARep(R);
    imagegrp := Group(R1.theImages, ());

    ind := 
      PositionProperty(
        R1.theImages, 
        g -> OrderPerm(g) = R.degree
      );
  
    # prime and exponent
    p := SmallestRootInt(R.degree);
    k := LogInt(R.degree, p);
    InfoLatticeDec("#I cyclic prime power: ", p, "^", k, "\n");

    # construct via formula from 
    # Sebastian's thesis
    # conjugate onto (1..n)
    InfoLatticeDec1("#I+ computing conjugating perm\n");
    gen  := R1.theImages[ind];
    Sn   := SymmetricGroup(R.degree);
    con1 := 
      AMatPerm(
        RepresentativeOperation(
          Sn, 
          gen, 
          CyclicGroup(R.degree).1
        ),
        R.degree,
        R.char
      );

    # the twiddles
    InfoLatticeDec1("#I+ computing twiddles\n");
    twiddle := function( i, j )
      return 
        Int((j mod p ^ (i + 1))/p ^ i) * 
        (j mod p ^ i) * 
        p ^ (k - 1 - i);
    end;      
    Ts := 
      List(
        [k - 1, k - 2..1], 
        i -> 
          DiagonalAMat(
            List([0..p ^ k - 1], 
            j -> RootOfUnity(p ^ k, R.char) ^ twiddle(i, j))
          )
      );

    # the sorting permutation (p-adic bit reversal)
    InfoLatticeDec1("#I+ compute bit reversal\n");
    con2 := [ ];
    for i in [1..R.degree] do
      Add(
        con2,
        Sum(
          List(
            [0..k - 1], 
            j -> (Int((i - 1)/(p ^ j)) mod p) * p ^ (k - 1 - j) 
          )
        )
      );
    od;
    con2 := AMatPerm(PermList(con2 + 1), R.degree, R.char);

    # the decomposition matrix for R
    con3 := [ ];
    for i in [1..k - 1] do
      Add(
        con3, 
        TensorProductAMat(
          IdentityPermAMat(p ^ (i - 1), R.char),
          DFTAMat(p, R.char),
          IdentityPermAMat(p ^ (k - i), R.char)
        )
      );
      Add(con3, Ts[i]);
    od;
    Add(
      con3, 
      TensorProductAMat(
        IdentityPermAMat(p ^ (k - 1), R.char),
        DFTAMat(p, R.char)
      )
    );
   
    # note that con3 is in simplified form
    con3 := con1 * Product(con3) * con2;
    con3.isMonMat := false;
    con3.isSimplified := true;
    con3.isSimplifiedFirstParse := true;

    # the irrs
    # construct first the generating irreducible irr,
    # all the others are powers
    InfoLatticeDec1("#I+ compute irrs\n");
    Zpk   := GroupWithGenerators(Subgroup(G, [ G.theGenerators[ind] ] ));
    irr   := ARepByImages(Zpk, [ Mon([RootOfUnity(p ^ k, R.char)]) ], "hom");
    genim := List(G.theGenerators, g -> MonAMat(g ^ irr));
    irrs := 
      List(
        [0..p ^ k - 1], 
        i -> 
          ARepByImages(
            G, 
            List(genim, g -> g ^ i),
            "hom"
          )
      );

    InfoLatticeDec1("#I+ simplifying decomposition matrix\n");
    con3 := SimplifyAMat(con3);
    InfoLatticeDec("#I rep of degree ", R.degree, " completed\n");
    return 
      ConjugateARep(
        DirectSumARep(irrs),
        PowerAMat(con3, -1, "invertible"),
        "invertible"
      );
  fi;

  # preparation for induction resp. switch recursion
  RSG  := TransitiveToInductionMonRep(R);
  con1 := RSG.conjugation;
  N    := 
    OneMaximalNormalSubgroupBetween(
      RSG.source, 
      RSG.rep.rep.source
    );

  # induction recursion: recurse with lower induction
  # -------------------------------------------------
  if N <> false then

    # decompose into double induction via N
    N    := GroupWithGenerators(N);
    RSNG := InsertedInductionARep(RSG.rep, N);
    con2 := RSNG.conjugation;
    p    := RSNG.degree/RSNG.rep.rep.degree;
    InfoLatticeDec(
      "#I induction recursion: ",
      R.degree, " -> ", R.degree/p
    );
    InfoLatticeDec(" * ", p, " (degrees)\n");

    # change transversal of upper induction
    # to {t^0, .., t^(p - 1)}. Since p is prime
    # by construction, any t in G\N
    # can be taken. Take a generator for
    # better evaluation.
    t         := First(G.theGenerators, g -> not g in N);
    T         := List([0..(p - 1)], i -> t ^ i);
    RSNbyTtoG := 
      TransversalChangeInductionARep(
        RSNG.rep, 
        T, 
        "isTransversal"
      );
    con3      := RSNbyTtoG.conjugation;

    # decompose lower induction
    InfoLatticeDec1("#I+ recurse\n");
    if hint = "noOuter" then
      D := DecompositionMonRep(RSNG.rep.rep, "noOuter");
    else
      D := DecompositionMonRep(RSNG.rep.rep);
    fi;
    if D = false then
      return false;
    fi;
    irrs := D.rep.summands;

    # blocks of the decomposition
    blocks := List(irrs, r -> r.degree);

    # check for irrs r, whether r ~= r ^ t, collect 
    # those in the list extendables, the others in
    # the list inducables. Use the fact, that the
    # irreducibles are ordered.
    # In order to check the equivalence above use 
    # the fact, that
    #   1. inner conjugates have (up to ordering)
    #      the same multisets of character values.
    #   2. if t permutes the conjugacy classes by 
    #      conjugation as the permutation p, then
    #        (chi ^ t).values =
    #        chi.values ^ p.
    #
    # extendables is a list of pairs [r, n], such that
    #   1. r is a rep in irrs equal to all its
    #      inner conjugates.
    #   2. n is the multiplicity of r in irrs.
    #   3. r's in different pairs are not equivalent.
    #   
    # inducables is a list of pairs [R, ns], such that
    #   1. R is a list of maximal length p containing 
    #      inner conjugates, more precisely
    #        R[i]  = R[1] ^ (t ^ (i-1)), if present.
    #      Note, that the list may contain holes!
    #   2. r's in different R's are not equivalent and
    #      no (inner) conjugates.
    #   3. ns is a list of length p containing the 
    #      multiplicities of the reps in R.
    # The positions of the r in irrs are collected in
    # extpermlist and indpermlist.
    extendables := [ ];
    extpermlist := [ ];
    inducables  := [ ];
    indpermlist := [ ];

    # collect irrs by equality in collirrs  
    # collirrs is a list of pairs [r, n], 
    # with the properties
    #   1. r's in different pairs are not 
    #      equivalant.
    #   2. r occurs with multiplicity n in irrs.
    # use the fact, that the irrs are ordered
    collirrs := [ ];
    i        := 0;
    while i < Length(irrs) do
      i := i + 1;
      r := irrs[i];
      n := 1;
      while i < Length(irrs) and irrs[i + 1] = r do
        n := n + 1;
        i := i + 1;
      od;
      Add(collirrs, [r, n]);
    od;

    InfoLatticeDec1("#I+ consider ", Length(collirrs), " many irrs\n");

    # note that at this point, G is never abelian, but N
    # might be, in which case computing all characters of
    # all irrs produces a lot of (square degree) numbers
    # we use the fact that in the latter case, all characters
    # are homomorphisms, which means that deciding on extendability
    # as well as inner conjugation can be done on the set of generators
    if IsAbelian(N) then
      
      # create the list of group elements on which we test
      # extendability. It is the generators expanded by 
      # conjugation with t, compute also the permutation
      # tperm arising from conjugation by t
      testelms := [ ];
      degs     := [ ];
      perms    := [ ];
      pcycle   := PermList(Concatenation([2..p], [1]));
      for g in N.theGenerators do      
        if g ^ t <> g then
          for i in [0..p - 1] do
            Add(testelms, g ^ (t ^ i));
          od;
          Add(degs, p);
          Add(perms, pcycle);
        else
          Add(testelms, g);
          Add(degs, 1);
          Add(perms, ());
        fi;
      od;
      tperm := DirectSumPerm(degs, perms);

      # compute the "character" with respect to this elements
      chars := 
        List(
          collirrs,
          r ->
            List(
              testelms,
              x -> TraceAMat(x ^ r[1])
            )
        );

    else # N is not abelian          

      # calculate permutation tperm on the conjugacy classes
      # of N induced by conjugation with t, tperm has only 
      # cycles of length p or 1
      ccs   := ConjugacyClasses(N);
      tperm := [ ];
      for cc in ccs do
        Add(
          tperm, 
          PositionProperty(
            ccs, 
            c -> cc.representative ^ t in c
          )
        );
      od;
      tperm := PermList(tperm);

    fi;

    # precompute the collected charactervalues
    # as third component in rn
    if IsAbelian(N) then

      # use "shorter" character
      for i in [1..Length(collirrs)] do 
        Add(collirrs[i], Collected(chars[i]));
      od;

    else

      # use ordinary character
      for rn in collirrs do
        Add(rn, Collected(CharacterARep(rn[1]).values));
      od;

    fi;

    pos := 1;
    for i in [1..Length(collirrs)] do 

      rn := collirrs[i];

      if IsAbelian(N) then
        char := chars[i];
      else
        char := CharacterARep(rn[1]).values;
      fi;

      if Permuted(char, tperm) = char then

        # rn is equivalent to all inner conjugates,
        # hence extendable
        Add(extendables, [rn[1], rn[2]]);
        Add(extpermlist, [pos..pos + rn[2] - 1]);
        pos := pos + rn[2];

      else

        # rn is not equivalent to all inner conjugates,
        # hence the induction is irreducible
        j    := 1;
        stop := false;
        while not stop do

          if not IsBound(inducables[j]) then

            # rn[1] is no (inner) conjugated to any
            # other seen so far
            inducables[j]  := [ [ rn[1] ], [ rn[2] ] ];
            indpermlist[j] := [ [pos..pos + rn[2] - 1] ];
            pos            := pos + rn[2];

            # memorize collected character values at
            # this place for fast comparison
            inducables[j][3] := rn[3];
            stop             := true;

            # in case N abelian, memorize index of the irr
            # to find the "shorter" character in chars
            if IsAbelian(N) then
              inducables[j][1][1].tmpIndex := i;
            fi;

          elif 

            # necessary for reps r1, r2 to be (inner) conjugated
            # is that the unordered multisets of the character
            # values are identical
            rn[3] <> inducables[j][3]

          then

            j := j + 1;

          else

            # now rn[1] is a serious candidate to be an (inner)
            # conjugate of inducables[j][1][1]
            k := 0;

            # character of inducables[j][1][1]
            if IsAbelian(N) then
              char1 := chars[inducables[j][1][1].tmpIndex];
            else
              char1 := CharacterARep(inducables[j][1][1]).values;
            fi;

            while 
              not Permuted(char1, tperm ^ k) = char and
              k < p
            do
              k := k + 1;
            od;
            if k = p then
              j := j + 1;
            else
              inducables [j][1][k + 1] := rn[1];
              inducables [j][2][k + 1] := rn[2];
              indpermlist[j][k + 1]    := [pos..pos + rn[2] - 1];
              pos                      := pos + rn[2];
              stop                     := true;
            fi;
          fi;
        od;
      fi;
    od;

    InfoLatticeDec1("#I+ ", Length(extendables), " many extendables\n");
    InfoLatticeDec1("#I+ ", Length(inducables), " many groups of inducables\n");

    # unbind memorization of collected
    # character values
    for rs in inducables do
      Unbind(rs[3]);
    od;

    # permutation conjugating irrs in desired order:
    #   first the extendables, then the inducables
    # this yields a perm matrix con4
    sortperm := 
      permblocks(
	blocks, 
	PermList(
	  Concatenation(
	    Concatenation(extpermlist),
	    Concatenation(
	      List(
		indpermlist,
		Concatenation
	      )
	    ) 
          )
        ) ^ -1
      ) ^ -1;
    con4   := AMatPerm(sortperm, R.degree/p, R.char);

    # check for theGenerators, in which residue class
    # they are and store index in inds
    inds := [ ];
    for g in G.theGenerators do
      g1   := g;
      i    := 0;
      while not g1 in N do
        g1 := g1/t;
        i  := i + 1;
      od;
      Add(inds, i);
    od;

    # extend extendables by intertwining space 
    # (avoids calculation of character table)
    # for small degrees, else with Minkwitz
    InfoLatticeDec1("#I+ extending ", Length(extendables), " extendables\n");
    extextendables:= [ ];
    
    # the following loop can be exchanged with the piece
    # of code at the end of this file, to do every extension
    # with Minkwitz. It seems, both variants are equally fast
    for rs in extendables do

      # catch trivial case
      if ARepOps.IsTrivialOneRep(rs[1]) then

        extrs := TrivialMonARep(G, 1, R.char);

      else

        # try to extend by intertwining space, this works
        # at the moment iff rs and the extension is monomial
        timage := ExtensionByIntertwiningSpaceNC(rs[1], G, t);

        if timage <> false then

          # extension is monomial
          extrs  := 
	    ARepByImages(
	      G,
	      List(
		[1..Length(G.theGenerators)],
		i -> 
                  MonAMat((G.theGenerators[i]/(t ^ inds[i])) ^ rs[1]) *
                  timage ^ inds[i] 
              ),
	      "hom"    
	    );

        else
      
          # use Minkwitz
          chi   := OneExtendingCharacter(CharacterARep(rs[1]), G);
          extrs := ExtensionARep(rs[1], chi);

        fi;

      fi;

      # add extensions with multiplicities
      Add(
        extextendables, 
        [extrs, rs[2]]
      );
    od;

    # calculate the image of t under the
    # entire extension of the extendables
    InfoLatticeDec1("#I+ image of t under extension\n");
    timage := [ ];
    for rs in extextendables do
      timage1 := t ^ rs[1];
      for i in [1..rs[2]] do
        Add(timage, timage1);
      od;
    od;

    # note, that no extendable has to be present
    # in the permutation case, the onerep is 
    # extendable, of course
    if Length(timage) > 0 then
      timage := DirectSumAMat(timage);
    else
      timage := false;
    fi;

    # calculate matrices to conjugate the inner
    # conjugates to be equal to inner conjugates,
    # first the inner conjugates are made "equal" to
    # inner conjugates by con5, then a permutation
    # con6 and a block diagonal con7 is calculated
    # to equalize the inductions of the inner conjugates
    # alldegs is the blockstructure of the decomposed
    # lower rep, where all extendables are viewed as one block.
    InfoLatticeDec1("#I+ extending ", Length(inducables), " inducables\n");      
    extdeg        := Sum(List(extendables, rs -> rs[1].degree * rs[2]));
    if extdeg > 0 then
      alldegs     := [extdeg];
    else
      alldegs     := [ ];
    fi;
    indcons       := [ ];
    indinducables := [ ];
    pcycle        := PermList(Concatenation([2..p], [1]));
    corrperm      := [ ];
    corrpermdegs  := [ ];
    corrdiag      := [ ];
    for rs in inducables do

      # calculate matrices conjugating 
      # rs[1][i + 1] onto rs[1][1] ^ (t ^ i)
      cons := [ IdentityPermAMat(rs[1][1].degree, R.char) ];
      for i in [1..p - 1] do
        if IsBound(rs[1][i + 1]) then 
          cons[i + 1] := 
	    IntertwiningSpaceARep(
              rs[1][i + 1], 
	      InnerConjugationARep(rs[1][1], G, t ^ i)
	    )[1];
        fi;
      od;

      # correcture matrix
      mult := 0;
      tprs := (t^p)^rs[1][1];
      for i in [1..p] do
        if IsBound(rs[2][i]) then
          mult := mult + rs[2][i];
          for j in [1..rs[2][i]] do
            Add(corrpermdegs, p * rs[1][i].degree);
            Add(
              corrperm, 
              TensorProductPerm([p, rs[1][i].degree], [pcycle^(i - 1), ()])
            );
            Append(
              corrdiag, 
              Concatenation(
                List([1..i - 1], k -> tprs), 
                List(
                  [1..p - i + 1], 
                  k -> IdentityPermAMat(rs[1][1].degree, R.char)
                )
              )
            );
            Add(alldegs, rs[1][i].degree);
            Add(indcons, cons[i]);
          od;
        fi;
      od;

      # result of induction with multiplicities
      indrs := InductionARep(rs[1][1], G, T);
      Add(indinducables, [indrs, mult]);
    od;

    # correcture matrix making inner conjugates
    # "equal" to inner conjugates
    if extdeg > 0 then
      con5   := 
	DirectSumAMat(
	  Concatenation(
	    [IdentityPermAMat(extdeg, R.char)], 
	    indcons
	  )
	);
    else
      con5 := DirectSumAMat(indcons);
    fi;
     
    # permutation and blockdiagonal matrix 
    # equalizing inductions of inner conjugates
    con6 := 
      AMatPerm(
        DirectSumPerm(
          [ extdeg * p,
            R.degree - extdeg * p
          ],
          [ (),
            DirectSumPerm(corrpermdegs, corrperm)
          ]
        ),
        R.degree,
        R.char
      );
    if extdeg > 0 then
      con7 := 
	DirectSumAMat(
	  Concatenation(
	    [IdentityPermAMat(extdeg * p, R.char)],
	    corrdiag
	  )
	);
    else
      con7 := DirectSumAMat(corrdiag);
    fi;

    # permutation conjugating the induction of the 
    # direct sum of irreducibles onto the direct sum
    # of the inductions
    # the extendables are viewed as one block
    con8 := AMatPerm(ARepOps.inductionBlocks(alldegs, p), R.degree, R.char);

    # construct onedim reps of G coming
    # from G/N
    lins := 
      List(
        [0.. p - 1], 
        i -> 
          ARepByImages(
            G, 
            List(
              [1..Length(G.theGenerators)], 
              k -> Mon( [(RootOfUnity(p, R.char) ^ i) ^ inds[k]] )
            ),
            "hom"
          )
      );

    # irrs of R
    InfoLatticeDec1("#I+ constructing irrs\n");
    indextendables := [ ];
    for l in lins do
      for rs in extextendables do
        lrs := InnerTensorProductARep(l, rs[1]);
        Add(indextendables, [lrs, rs[2]]);
      od;
    od;
    allirrs := Concatenation(indextendables, indinducables);
    
    # permutation to sort irrs
    InfoLatticeDec1("#I+ sorting irrs\n");
    perm1  := [1..Length(allirrs)];
    blocks := List(allirrs, rs -> rs[1].degree * rs[2]);
    SortParallel(
      allirrs, 
      perm1, 
      function(rs1, rs2) return less(rs1[1], rs2[1]); end
    );
    perm1 := permblocks(blocks, PermList(perm1) ^ -1) ^ -1;
    con9  := AMatPerm(perm1, R.degree, R.char);

    # monormatify allirrs
    InfoLatticeDec1("#I+ monormatifying irrs\n");
    allirrs1 := [ ];
    for rs in allirrs do
      ers := monormatify(rs[1]);
      for i in [1..rs[2]] do
        Add(allirrs1, ers);
      od;
    od;

  # decomposition matrix for R
  # some parts are precalculated for faster
  # simplifying afterwards
  if extdeg > 0 then
    if R.degree - p * extdeg > 0 then
      con01 := 
	  DirectSumAMat(
	    Concatenation(
	      List([0..(p - 1)], i -> timage ^ i),
	      [IdentityPermAMat(R.degree - p * extdeg, R.char)]
	    )
	  );
      con02 :=
	  DirectSumAMat(
	    TensorProductAMat(
	      DFTAMat(p, R.char),
	      IdentityPermAMat(extdeg, R.char)
	    ),
	    IdentityPermAMat(R.degree - p * extdeg, R.char)
	  );
    else
      con01 := 
	  DirectSumAMat(
	    List([0..(p - 1)], i -> timage ^ i)
	  );
      con02 :=
	  TensorProductAMat(
	    DFTAMat(p, R.char),
	    IdentityPermAMat(extdeg, R.char)
	  );
    fi;
  fi;

  con03 := MonAMatAMat((con3 * con2 * con1) ^ -1);
  if IsMonMat(con5) then
    con04 := 
      TensorProductAMat(
	IdentityPermAMat(p, R.char),
	D.conjugation.element
      );
    con05 := 
      MonAMatAMat(
        TensorProductAMat(
	  IdentityPermAMat(p, R.char),
	  con4 * con5
        ) *
        con8 ^ -1 *
        con6 
      );
  else
    con04 := 
      TensorProductAMat(
	IdentityPermAMat(p, R.char),
	D.conjugation.element * con4 * con5
      );
    con05 := MonAMatAMat(con8 ^ -1 * con6);
  fi;

  if extdeg > 0 then
    con06 := con03 * con04 * con05 * con7 * con01 * con02 * con9;
  else
    con06 := con03 * con04 * con05 * con7 * con9;
  fi;
  con06.isMonMat := false;
  
  InfoLatticeDec1("#I+ simplifying decomposition matrix\n");
  con06 := SimplifyAMat(con06);

  InfoLatticeDec("#I rep of degree ", R.degree, " completed\n");
  return
    ConjugateARep(
      DirectSumARep(allirrs1),
      PowerAMat(con06, -1, "invertible"),
      "invertible"
    );

  fi;


  # switch recursion: recurse with restriction
  # ------------------------------------------
  # R.source is solvable
  # recurse with normal subgroup of prime index
  if IsSolvable(R.source) then

    # get normal subgroup of prime index
    N := 
      GroupWithGenerators(
        CompositionSeries(G)[2]
      );
    p := Index(R.source, N);
    InfoLatticeDec(
      "#I switch recursion (degree = ",
      R.degree,
      "): ",
      Size(G), 
      " -> ", 
      p,
      " * ",
      Size(N),
      " (group sizes)\n"
    );
    InfoLatticeDec1("#I+ prime index: ", p, "\n");

    # recurse with restriction
    # note that restriction is transitive, too
    R1   := RestrictionInductionARep(RSG.rep, N);
    con2 := R1.conjugation;
    if Length(R1.rep.summands) <> 1 then
      Error("<RSG> restriction <N> should be transitive");
    fi;
    RN   := R1.rep.summands[1];
    if hint = "noOuter" then
      D := DecompositionMonRep(RN, "noOuter");
    else
      D := DecompositionMonRep(RN);
    fi;
    if D = false then
      return false;
    fi;
    con3 := D.conjugation.element;
    irrs := D.rep.summands;
    # blocks of the decomposition
    blocks := List(irrs, r -> r.degree);

    # collect irrs by equality in collirrs
    # collirrs is a list of pairs [r, n], 
    # with the properties
    #   1. r's in different pairs are not 
    #      equivalant.
    #   2. r occurs with multiplicity n in irrs.
    # use the fact, that the irrs are ordered
    collirrs := [ ];
    i        := 0;
    while i < Length(irrs) do
      i := i + 1;
      r := irrs[i];
      n := 1;
      while i < Length(irrs) and irrs[i + 1] = r do
        n := n + 1;
        i := i + 1;
      od;
      Add(collirrs, [r, n]);
    od;

    # catch element in G\N and build transversal of G/N
    t := First(G.theGenerators, g -> not g in N);
    T := List([0..p - 1], i -> t^i);

    # check for irrs r, whether r ~= r ^ t, collect 
    # those in the list extendables, the others in
    # the list inducables. Use the fact, that the
    # irreducibles are ordered.
    # In order to check the equivalence above use 
    # the fact, that
    #   1. inner conjugates have (up to ordering)
    #      the same multisets of character values.
    #   2. if t permutes the conjugacy classes by 
    #      conjugation as the permutation p, then
    #        (chi ^ t).values =
    #        chi.values ^ p.
    #
    # extendables is a list of pairs [r, n], such that
    #   1. r is a rep in irrs equal to all its
    #      inner conjugates.
    #   2. n is the multiplicity of r in irrs.
    #   3. r's in different pairs are not equivalent.
    #   
    # inducables is a list of pairs [R, n], where
    #   1. R is a list of length p, such that
    #        R[1] ^ (t^(i-1)) ~= R[i]
    #   2. n is the common! multiplicity of the R[i]'s, i = 1..p
    #      The multiplicity is common, since RN is extendable to G!
    #   3. R[i]'s in different pairs are no inner conjugates.
    # The positions of the r in irrs are collected in
    # extpermlist and indpermlist. The aim is to order the 
    # inducables into groups of inner conjugates
    #   r^(t^0)..r^(t^(p-1))
    # to extend these groups simultaneously by the induction of 
    # the first
    extendables := [ ];
    extpermlist := [ ];
    inducables  := [ ];
    indpermlist := [ ];

    # note that at this point, G is never abelian, but N
    # might be, in which case computing all characters of
    # all irrs produces a lot of (square degree) numbers
    # we use the fact that in the latter case, all characters
    # are homomorphisms, which means that deciding on extendability
    # as well as inner conjugation can be done on the set of generators
    if IsAbelian(N) then
      
      # create the list of group elements on which we test
      # extendability. It is the generators expanded by 
      # conjugation with t, compute also the permutation
      # tperm arising from conjugation by t
      testelms := [ ];
      degs     := [ ];
      perms    := [ ];
      pcycle   := PermList(Concatenation([2..p], [1]));
      for g in N.theGenerators do      
        if g ^ t <> g then
          for i in [0..p - 1] do
            Add(testelms, g ^ (t ^ i));
          od;
          Add(degs, p);
          Add(perms, pcycle);
        else
          Add(testelms, g);
          Add(degs, 1);
          Add(perms, ());
        fi;
      od;
      tperm := DirectSumPerm(degs, perms);

      # compute the "character" with respect to this elements
      chars := 
        List(
          collirrs,
          r ->
            List(
              testelms,
              x -> TraceAMat(x ^ r[1])
            )
        );

    else # N is not abelian          

      # calculate permutation tperm on the conjugacy classes
      # of N induced by conjugation with t, tperm has only 
      # cycles of length p or 1
      ccs   := ConjugacyClasses(N);
      tperm := [ ];
      for cc in ccs do
        Add(
          tperm, 
          PositionProperty(
            ccs, 
            c -> cc.representative ^ t in c
          )
        );
      od;
      tperm := PermList(tperm);

    fi;

    # precompute the collected charactervalues
    # as third component in rn
    if IsAbelian(N) then

      # use "shorter" character
      for i in [1..Length(collirrs)] do 
        Add(collirrs[i], Collected(chars[i]));
      od;

    else

      # use ordinary character
      for rn in collirrs do
        Add(rn, Collected(CharacterARep(rn[1]).values));
      od;

    fi;

    pos := 1;
    for i in [1..Length(collirrs)] do

      rn := collirrs[i];

      if IsAbelian(N) then
        char := chars[i];
      else
        char := CharacterARep(rn[1]).values;
      fi;

      if Permuted(char, tperm) = char then

	# rn is equivalent to all inner conjugates,
	# hence extendable
	Add(extendables, [rn[1], rn[2]]);
	Add(extpermlist, [pos..pos + rn[2] - 1]);
	pos := pos + rn[2];

      else

	# rn is not equivalent to all inner conjugates,
	# hence the induction is irreducible
	j    := 1;
	stop := false;
	while not stop do
	  if not IsBound(inducables[j]) then

	    # rn[1] is no (inner) conjugated to any
	    # other seen so far
	    inducables[j]  := [ [ rn[1] ], rn[2] ];
            indpermlist[j] := [ ];
            indpermlist[j][1] := [pos..pos + rn[2] - 1];
            pos               := pos + rn[2];

	    # memorize collected character values at
	    # this place for fast comparison
	    inducables[j][3] := rn[3];
	    stop             := true;

            # in case N abelian, memorize index of the irr
            # to find the "shorter" character in chars
            if IsAbelian(N) then
              inducables[j][1][1].tmpIndex := i;
            fi;

	  elif 

	    # necessary for reps r1, r2 to be (inner) conjugated
	    # is that the unordered multisets of the character
	    # values are identical
	    rn[3] <> inducables[j][3]

	  then

	    j := j + 1;

	  else

	    # now rn[1] is a serious candidate to be an (inner)
	    # conjugate of inducables[j][1][1]
	    k := 0;

            # character of inducables[j][1][1]
            if IsAbelian(N) then
              char1 := chars[inducables[j][1][1].tmpIndex];
            else
              char1 := CharacterARep(inducables[j][1][1]).values;
            fi;

            while 
              not Permuted(char1, tperm ^ k) = char and
              k < p
            do
	      k := k + 1;
	    od;
	    if k = p then

              # rn[1] is no inner conjugate
	      j := j + 1;

	    else

              # rn[1] is inner conjugate
	      inducables [j][1][k + 1] := rn[1];
              indpermlist[j][k + 1]    := [pos..pos + rn[2] - 1];
              pos                      := pos + rn[2];
	      stop                     := true;
	    fi;
	  fi;
	od;
      fi;
    od;

    # unbind memorization of collected
    # character values
    for rs in inducables do
      Unbind(rs[3]);
    od;

    # some info
    InfoLatticeDec1(
      "#I+ extendables (deg, mult): ",
      List(extendables, p -> [p[1].degree, p[2]]),
      "\n"
    );
    InfoLatticeDec1(
      "#I+ inducables (deg, mult): ",
      List(inducables, p ->  [p[1][1].degree, p[2]]),
      "\n"
    );

    # permutation conjugating irrs in desired order:
    # first the extendables, then the inducables
    # in groups r, r^t, .., r^(t^(p-1)),
    # this yields a perm matrix con4
    sortperm := 
      permblocks(
	blocks, 
	PermList(
	  Concatenation(
	    Concatenation(extpermlist),
	    Concatenation(
	      List(
		indpermlist,
		Concatenation
	      )
	    ) 
          )
        ) ^ -1
      ) ^ -1;
    con4 := AMatPerm(sortperm, R.degree, R.char);

    # calculate matrices to conjugate the inner
    # conjugates to be equal to inner conjugates,
    # this yields a matrix con5
    extdeg := Sum(List(extendables, rs -> rs[1].degree * rs[2]));
    con5   := [ IdentityPermAMat(extdeg, R.char) ];
    for rs in inducables do

      # calculate matrices conjugating 
      # rs[1][i + 1] onto rs[1][1] ^ (t ^ i)
      # note that the (one) generator of the intertwining space
      # must be invertible
      Add(con5, IdentityPermAMat(rs[1][1].degree * rs[2], R.char));
      for i in [1..p - 1] do
        Add(
          con5,
          TensorProductAMat(
            IdentityPermAMat(rs[2], R.char),
            IntertwiningSpaceARep(
              rs[1][i + 1], 
	      InnerConjugationARep(rs[1][1], G, t ^ i)
	    )[1]
          )
        );
      od;

    od;

    con5 := SimplifyAMat(DirectSumAMat(con5));

    # info
    InfoLatticeDec1(
      "#I+ equalizer for groups of inducables: ",
      con5,
      "\n"
    );

    # the matrix con2^-1*con3*con4*con5 decomposes RSG.rep 
    # into blocks, one block for each group of equal
    # extendables, one block for each group of inner
    # conjugate inducables,
    # determine these blocks
    pos    := 1;
    blocks := [ ];
    for i in [1..Length(extendables)] do
      Add(
        blocks, 
        [pos..pos + extendables[i][1].degree*extendables[i][2] - 1]
      );
      pos := pos + extendables[i][1].degree*extendables[i][2];
    od;
    for i in [1..Length(inducables)] do
      Add(
        blocks, 
        [pos..pos + inducables[i][1][1].degree*inducables[i][2]*p - 1]
      );
      pos := pos + inducables[i][1][1].degree*inducables[i][2]*p;
    od;

    # info
    InfoLatticeDec1(
      "#I+ blocks of decomposed extension: ",
      blocks,
      "\n"
    );

    # decompose rep RSG.rep of G partially with 
    # the decomposition matrix of RN,
    # here an expensive conjugation has to be performed
    DR         := RSG.rep^(con2^-1*con3*con4*con5);
    summandsDR := 
      List(
        blocks, 
        b -> RestrictionToSubmoduleARep(MatARepARep(DR), b, "hom")
      );
    InfoLatticeDec1("#I+ extension decomposed\n");

    # extend the decomposed rep of N to G
    # extend the extendables in the right way,
    # the inducables by induction
    # con6 will be the matrix which decomposes
    # DR entirely
    con6    := [ ];
    allirrs := [ ];
    InfoLatticeDec1("#I+ consider extendable blocks\n");
    for i in [1..Length(extendables)] do

      # degree and multiplicityy
      deg  := extendables[i][1].degree;
      mult := extendables[i][2];

      # info
      InfoLatticeDec1(
        "#I+ deg = ", deg, ", mult = ", mult, "\n"
      );

      # decompose the character
      # check if character is homogeneous
      chi        := CharacterARep(summandsDR[i]);
      InfoLatticeDec1("#I+ <chi, chi> = ", ScalarProduct(chi, chi), "\n");
      chiirrs    := [ ];
      if ScalarProduct(chi, chi) = mult^2 then
 
        # chi is homogeneous = r^n, hence the extending
        # character can be deduced by dividing the
        # multiplicity
        chiirrs[1] := [Character(chi.source, 1/mult * chi.values), mult];

      else

        # char table must be computed (sigh!)
        for r in Irr(G) do
          if Degree(r) = deg then
            s := ScalarProduct(chi, r);
            if s > 0 then
              Add(chiirrs, [r, s]);
            fi;
          fi;
        od;

      fi;

      # construct reps for the irreducible 
      # components of chi, which are all extensions
      # of extendables[i][1]
      extsummand := [ ];
      for chin in chiirrs do
        rchin := monormatify(ExtensionARep(extendables[i][1], chin[1]));
        Add(allirrs, [rchin, chin[2]]);
        for j in [1..chin[2]] do
          Add(extsummand, rchin);
        od;
      od;

      # correcture matrix
      if Length(chiirrs) = 1 then
        
        # the extension is homogenous, hence
        # there is nothing to correct
        Add(
          con6, 
          IdentityPermAMat(mult * deg, R.char)
        );

      else

        # search g in G\N, such that the character
        # chiirrs[1] (and hence all chiirrs) 
        # does not vanish,
        # refer to the description of the algorithm
        # above for the reason
        g := 
          First(
            ConjugacyClasses(G),
            cc -> 
              not cc.representative in N and 
              cc.representative^chiirrs[1][1] <> 0
          ).representative;

        # image of g under summandsDR[i] and 
        # the prototype extsummand
        M1 := partialTrace(MatAMat(g^summandsDR[i]), deg);
        M2 := DiagonalMat(List(extsummand, r -> TraceAMat(g^r)));

        # determine the "intertwining space"
        # of the two matrices
        intsummand := ConjugationMat([ M1 ], [ M2 ]);
        InfoLatticeDec1("#I+ intsummand: ", intsummand, "\n");

        # determine an invertible element
        # in intsummand, try first to find a sparse one, 
        # and if that fails, choose randomly
        M := FindSparseInvertible(intsummand);
        if M = false then
          M := FindRandomInvertible(intsummand);
        fi;

	Add(
	  con6, 
	  TensorProductAMat(
	    M,
	    IdentityPermAMat(deg, R.char)
	  )
	);

      fi;

    od;

    # con8 is one on the extendables and
    # for the inducables permuted the induction of
    # the sum onto the sum of the inductions
    con8 := [ IdentityPermAMat(extdeg, R.char) ];

    # extend the inducables by induction of the first
    InfoLatticeDec1("#I+ consider inducables\n");
    for i in [1..Length(inducables)] do
      
      extinds := [ ];
      extind  := monormatify(InductionARep(inducables[i][1][1], G, T));
      Add(allirrs, [extind, inducables[i][2]]);
      for j in [1..inducables[i][2]] do
        Add(extinds, extind);
      od;

      # the decomposition matrix for 
      # summandsDR[Length(extendables) + i]
      # can be read of
      deg  := inducables[i][1][1].degree;
      mult := inducables[i][2];
      InfoLatticeDec1("#I+ deg = ", deg, ", mult = ", mult, "\n");
      cons := [ IdentityPermAMat(deg * mult) ];
      for k in [1..p - 1] do
        Add(
          cons,
          SimplifyAMat(
            TensorProductAMat(
              SubmatrixAMat(
                T[k + 1] ^ summandsDR[Length(extendables) + i],
                List([1..mult], j -> (j - 1) * deg + 1),
                List([1..mult], j -> k * mult * deg + (j - 1) * deg + 1)
              ),
              IdentityPermAMat(deg, R.char)
            )
          )
        );
        InfoLatticeDec1("#I+ decomposer: ", cons, "\n");
      od;

      # invert summands
      cons := List(cons, InverseAMat);

      # add to con6
      Add(
        con6, 
        SimplifyAMat(DirectSumAMat(cons))
      );

      # permuting the induction of the sum onto
      # the sum of the inductions
      L :=
        Concatenation(
          List(
            [1..mult],
            k ->
              Concatenation(
                List(
                  [1..p],
                  i -> (k - 1) * deg + (i - 1) * mult * deg + [1..deg]
                )
              )
          )
        );
      Add(
        con8, 
        AMatPerm(
          MappingPermListList(L, [1..mult * deg * p]),
          mult * deg * p,
          R.char
        )
      );
    od;

    con6 := SimplifyAMat(DirectSumAMat(con6));
    con8 := SimplifyAMat(DirectSumAMat(con8));

    # permutation to sort irrs
    perm1  := [1..Length(allirrs)];
    blocks := List(allirrs, rs -> rs[1].degree * rs[2]);
    SortParallel(
      allirrs, 
      perm1, 
      function(rs1, rs2) return less(rs1[1], rs2[1]); end
    );
    perm1 := permblocks(blocks, PermList(perm1) ^ -1) ^ -1;
    con7  := AMatPerm(perm1, R.degree, R.char);

    allirrs1 := 
      Concatenation(
        List(
          allirrs, 
          rs -> List([1..rs[2]], i -> rs[1])
        )
      );

    InfoLatticeDec1("#I+ simplifying decomposition matrix\n");
    con1 := 
      SimplifyAMat(
        MonAMatAMat(con1^-1*con2^-1)*
        con3*con4*con5*con6*con8*con7
      );

    InfoLatticeDec("#I rep of degree ", R.degree, " completed\n");
    return 
      ConjugateARep(
        DirectSumARep(allirrs1),
        PowerAMat(con1, -1, "invertible"),
        "invertible"
      );
  fi;

  return false;
end;


# Miscellaneous
# -------------
# Piece of code, which can be used in induction recursion
# to compute extensions via Minkwitz rather than using the
# intertwining space.
#
#    for rs in extendables do
#
#      # if deg = 1 construct directly
#      if rs[1].degree = 1 then
#
#          timage := 
#            Mon( 
#              [ RootOfRootOfUnity(MatAMat((t ^ p) ^ rs[1])[1][1], p) ]
#            );
#          extrs  := 
#	    ARepByImages(
#	      G,
#	      List(
#		[1..Length(G.theGenerators)],
#		i -> 
#                  MonAMat((G.theGenerators[i]/(t ^ inds[i])) ^ rs[1]) *
#                  timage ^ inds[i] 
#              ),
#              "hom"    
#	    );
#
#      else
#    
#        # use Minkwitz
#        chi   := OneExtendingCharacter(CharacterARep(rs[1]), G);
#        extrs := ExtensionARep(rs[1], chi);
#
#      fi;
#
#      # add extensions with multiplicities
#      Add(
#        extextendables, 
#        [extrs, rs[2]]
#      );
#    od;
