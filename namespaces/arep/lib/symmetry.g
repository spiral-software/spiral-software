# -*- Mode: shell-script -*-
# Determination of Symmetry as a Pair of Representations
# MP, 12.12.97, GAPv3.4

# Literature:
#   T. Minkwitz: PhD. Thesis, University of Karlsruhe, 1993
#   S. Egner   : PhD. Thesis, University of Karlsruhe, 1997
#   M. Pueschel: PhD. Thesis, University of Karlsruhe, 1998

# Nach Submission (AREP 1.1)
# --------------------------
# 18.05.99: PermIrredSymmetry und PermIrredSymmetry1 
#           testet nun ob die Matrix invertierbar ist.
#           Kleinen Bug in beiden Funktionen entfernt:
#           im Nicht-Erfolgsfall muss eine leere Liste
#           zurueckgegeben werden.

# Nach AREP 1.1 (AREP 1.2)
# ------------------------
# 29.08.99: Mon2IrredSymmetry(1) von Sebastian eingebaut.
#

#F Computing Symmetry of Matrices
#F ==============================
#F

#F PermPermSymmetry( <mat/amat> )
#F   calculates the perm-perm symmetry of the <mat/amat> M
#F   as a pair [R1, R2] of "perm"-areps of the same group G.
#F   This means, that M is in the intertwining space 
#F   Int(R1, R2), i.e.
#F     R1(g) * M = M * R2(g), for all g in G.
#F   If G is solvable, then it is an aggroup.
#F

PermPermSymmetry := function ( A )
  local G, char, Gag, phi;

  # check argument
  if IsAMat(A) then
    A := MatAMat(A);
  fi;
  if not IsMat(A) then
    Error("<A> must be a matrix");
  fi;

  # calculate perm-perm symmetry
  G    := PermPermSym(A);
  char := Characteristic(DefaultField(A[1][1]));

  # switch to an aggroup, if possible
  if IsSolvable(G) then
    Gag := GroupWithGenerators(AgGroup(G));
    phi := Gag.bijection;

    return
      [ ARepByImages(
          Gag,
          List(
            Gag.theGenerators, 
            g -> PermPermSymL(G, Image(phi, g))
          ),
          G.dimensionsMat[1],
          char,
          "hom"
        ),
        ARepByImages(
          Gag,
          List(
            Gag.theGenerators, 
            g -> PermPermSymR(G, Image(phi, g))
          ),
          G.dimensionsMat[2],
          char,
          "hom"
        )
      ];
  else
    G := GroupWithGenerators(G);

    return
      [ ARepByImages(
          G,
          PermPermSymL(G, G.theGenerators),
          G.dimensionsMat[1],
          char,
          "hom"
        ),
        ARepByImages(
          G,
          PermPermSymR(G, G.theGenerators),
          G.dimensionsMat[2],
          char,
          "hom"
        )
      ]; 
  fi;
end;


#F MonMonSymmetry( <mat/amat> )
#F   calculates the mon-mon symmetry of the <mat/amat> A
#F   as a pair [R1, R2] of "mon"-areps of the same group G.
#F   This means, that A is in the intertwining space 
#F   Int(R1, R2), i.e.
#F     R1(g) * A = A * R2(g), for all g in G.
#F   If G is solvable, then it is an aggroup.
#F   The function only works for characteristic zero.
#F

MonMonSymmetry := function ( A )
  local G, char, Gag, phi;

  # check argument
  if IsAMat(A) then
    A := MatAMat(A);
  fi;
  if not IsMat(A) then
    Error("<A> must be a matrix");
  fi;
  char := Characteristic(DefaultField(A[1][1]));

  # require char = 0
  if char <> 0 then
    Error("characteristic of <A> must be zero");
  fi;

  # calculate perm-perm symmetry
  G    := MonMonSym(A);

  # switch to an aggroup, if possible
  if IsSolvable(G) then
    Gag := GroupWithGenerators(AgGroup(G));
    phi := Gag.bijection;

    return
      [ ARepByImages(
          Gag,
          List(
            Gag.theGenerators, 
            g -> MonMonSymL(G, Image(phi, g))
          ),
          "hom"
        ),
        ARepByImages(
          Gag,
          List(
            Gag.theGenerators, 
            g -> MonMonSymR(G, Image(phi, g))
          ),
          "hom"
        )
      ];
  else
    G := GroupWithGenerators(G);

    return
      [ ARepByImages(
          G,
          MonMonSymL(G, G.theGenerators),
          "hom"
        ),
        ARepByImages(
          G,
          MonMonSymR(G, G.theGenerators),
          "hom"
        )
      ]; 
  fi;
end;


#F PermIrredSymmetry( <mat/amat> [, <maxblocksize> ] )
#F   calculates a list containing all the non-trivial 
#F   perm-irred symmetries of <mat/amat>, where the degree
#F   of at least one irreducible is <= <maxblocksize>. 
#F   The default for <maxblocksize> is 2 because of the 
#F   expensive computation.
#F   A perm-irred symmetry is a pair [R1, R2] of areps of the same 
#F   group G, where R1 is a "perm"-arep and R2 is a direct sum 
#F   of irreducible "mat"-areps, conjugated by a "perm"-amat.
#F   M = <mat/amat> is in the intertwining space of R1 and R2, i.e.
#F     R1(g) * M = M * R2(g), for all g in G.
#F   If G is solvable, then it is an aggroup.
#F   The function only works for characteristic zero.
#F

PermIrredSymmetry := function ( arg )
  local A, k, dim, char, Gs, G, RG, Rs, irrs, nrirrs, b, perm;

  # decode and check arguments
  if Length(arg) = 1 then
    A := arg[1];
    if IsAMat(A) then
      A := MatAMat(A);
    fi;
    if not IsMat(A) then
      Error("<A> must be a matrix or an amat");
    fi;
    dim := DimensionsMat(A);
    if dim[1] <> dim[2] or RankMat(A) < dim[1] then
      return [ ];
    fi;
    k := 2;
  elif Length(arg) = 2 then
    A := arg[1];
    k := arg[2];
    if IsAMat(A) then
      A := MatAMat(A);
    fi;
    if not IsMat(A) then
      Error("<A> must be a matrix or an amat");
    fi;
    dim := DimensionsMat(A);
    if dim[1] <> dim[2] or RankMat(A) < dim[1] then
      return [ ];
    fi;
  else
    Error("usage: PermIrredSymmetry( <mat/amat> [, <maxblocksize> ] )");
  fi;
  char := Characteristic(DefaultField(A[1][1]));

  # require char = 0
  if char <> 0 then
    Error("characteristic of <A> must be zero");
  fi;

  # calculate non-trivial perm-block symmetry
  Gs := 
    Filtered(
      PermBlockSymBySubsets(A, [1..k]),
      G -> not IsTrivial(G)
    );
  Gs := List(Gs, GroupWithGenerators);

  # check each one for irreducibility
  # compute first the number of irreducibles
  # components of the right rep and compare
  # with length of the kbs
  Rs := [ ];
  for G in Gs do
    RG := NaturalARep(G, dim[1], char);
    nrirrs := 
      Sum(
        List(
          Irr(G),
          chi -> ScalarProduct(chi, CharacterARep(RG))
        )
      );
    if nrirrs = Length(G.kbsM) then
      irrs := [ ];
      for b in G.kbsM do
        Add(
          irrs,
          ARepByImages(
            G,
            List(
              G.theGenerators,
              g -> PermBlockSymR(G, b, g)
            ),
            "hom"
          )
        );
      od;
      perm := 
        MappingPermListList(
          [1..dim[1]],
          Concatenation(G.kbsM)
        );
      Add(
        Rs, 
        [ RG,
          ConjugateARep(
            DirectSumARep(irrs),
            AMatPerm(perm, dim[1], char)
          )
        ]
      );
    fi;
  od;

  return Rs;  
end;


# PermIrredSymmetry1 does the same as PermIredSymmetry, but
# throws away all symmetries with a block of size > <maxblocksize>.
# The reason is: every unitary matrix with a column only
# containing ones has a perm-irred symmetry with G = S_n.
# Checking for irreducibility (via character) slows down substantially.
# Note, however, that a perm-irred symmetry with all blocks
# of size < <maxblocksize> is only found, iff splitting off one block
# results in all blocks being split off.


#F PermIrredSymmetry1( <mat/amat> [, <maxblocksize> ] )
#F   calculates a list containing all the non-trivial 
#F   perm-irred symmetries of <mat/amat>, where the degrees
#F   of all irreducibles is <= <maxblocksize>
#F   (This is the difference to PermIrredSymmetry).
#F   The default for <maxblocksize> is 2 because of the 
#F   expensive computation.
#F   A perm-irred symmetry is a pair [R1, R2] of areps of the same 
#F   group G, where R1 is a "perm"-arep and R2 is a direct sum 
#F   of irreducible "mat"-areps, conjugated by a "perm"-amat.
#F   M = <mat/amat> is in the intertwining space of R1 and R2, i.e.
#F     R1(g) * M = M * R2(g), for all g in G.
#F   If G is solvable, then it is an aggroup.
#F   The function only works for characteristic zero.
#F

PermIrredSymmetry1 := function ( arg )
  local A, k, dim, char, Gs, G, RG, Rs, irrs, nrirrs, b, perm;

  # decode and check arguments
  if Length(arg) = 1 then
    A := arg[1];
    if IsAMat(A) then
      A := MatAMat(A);
    fi;
    if not IsMat(A) then
      Error("<A> must be a matrix or an amat");
    fi;
    dim := DimensionsMat(A);
    if dim[1] <> dim[2] or RankMat(A) < dim[1] then
      return [ ];
    fi;
    k := 2;
  elif Length(arg) = 2 then
    A := arg[1];
    k := arg[2];
    if IsAMat(A) then
      A := MatAMat(A);
    fi;
    if not IsMat(A) then
      Error("<A> must be a matrix or an amat");
    fi;
    dim := DimensionsMat(A);
    if dim[1] <> dim[2] or RankMat(A) < dim[1] then
      return [ ];
    fi;
  else
    Error("usage: PermIrredSymmetry1( <mat/amat> [, <maxblocksize> ] )");
  fi;
  char := Characteristic(DefaultField(A[1][1]));

  # require char = 0
  if char <> 0 then
    Error("characteristic of <A> must be zero");
  fi;

  # calculate non-trivial perm-block symmetry
  Gs := 
    Filtered(
      PermBlockSymBySubsets(A, [1..k]),
      G -> ForAll(G.kbsM, b -> Length(b) <= k) and not IsTrivial(G)
    );
  Gs := List(Gs, GroupWithGenerators);

  # check each one for irreducibility
  # compute first the number of irreducibles
  # components of the right rep and compare
  # with length of the kbs
  Rs := [ ];
  for G in Gs do
    RG := NaturalARep(G, dim[1], char);
    nrirrs := 
      Sum(
        List(
          Irr(G),
          chi -> ScalarProduct(chi, CharacterARep(RG))
        )
      );
    if nrirrs = Length(G.kbsM) then
      irrs := [ ];
      for b in G.kbsM do
        Add(
          irrs,
          ARepByImages(
            G,
            List(
              G.theGenerators,
              g -> PermBlockSymR(G, b, g)
            ),
            "hom"
          )
        );
      od;
      perm := 
        MappingPermListList(
          [1..dim[1]],
          Concatenation(G.kbsM)
        );
      Add(
        Rs, 
        [ RG,
          ConjugateARep(
            DirectSumARep(irrs),
            AMatPerm(perm, dim[1], char)
          )
        ]
      );
    fi;
  od;

  return Rs;  
end;


#F Mon2IrredSymmetry( <mat/amat> [, <maxblocksize> ] )
#F   calculates a list containing all non-trivial Mon2-Irred-Symmetries
#F   of the given square and invertible matrix M such that there is a block
#F   of size <= maxblocksize (default is 2 to avoid expensive calculations).
#F     A Mon2-Irred-Symmetry is a pair [L, R] of AReps of a common group G
#F   such that L is a monomial ARep and R is a direct sum irreducibles 
#F   conjugated with a permutation matrix such that L^AMat(M) = R.
#F   The matrix M must be in characteristic zero and L contains only 
#F   entries [-1, 0, 1].
#F

Mon2IrredSymmetry := function ( arg )
  local 
    M, maxblocksize, # arguments
    reps,            # result, pair of AReps
    lat, G,          # lattice of groups, G in lat
    Gbs, Gb,         # block parts of G, an element of Gbs
    L,               # left hand representation
    Rs, R,           # right blocks, combined reps
    n;               # degree of M

  # decode and check arg
  if Length(arg) = 1 then

    # <mat/amat>
    M            := arg[1];
    maxblocksize := 2;

  elif Length(arg) = 2 then

    # <mat/amat>, <maxblocksize>
    M            := arg[1];
    maxblocksize := arg[2];
  
  else
    Error("usage: Mon2IrredSymmetry( <mat/amat> [, <maxblocksize> ] )");
  fi;
  if IsAMat(M) then
    M := MatAMat(M);
  fi;
  if not IsMat(M) and 0*M[1][1] = 0 then
    Error("<M> must be a matrix in characteristic zero");
  fi;
  if not IsInt(maxblocksize) then
    Error("<maxblocksize> must be integer");
  fi;
  n := Length(M);

  # use Mon2BlockSym and filter out trivial groups
  lat := Mon2BlockSymBySubsets(M, [1..maxblocksize]);
  lat := Filtered(lat, G -> not Size(G) in [1, Factorial(n)*2^n]);

  # find the fully decomposed ones
  for G in lat do
    CharTable(G);
  od;

  CompletedMon2BlockSym(lat);
  lat := Filtered(lat, G -> ForAll(G.charactersM, IsIrreducible));

  # make representations
  reps := [ ];
  for G in lat do

    # fix a generating set of G
    G.theGenerators := G.generators;

    # compute groups for the blocks
    Gbs := List(G.kbsM, b -> BlockMon2BlockSym(G, b));

    # the left representation
    L := ARepByImages(G, Mon2MatSymL(Gbs[1], G.theGenerators), "hom");

    # the right representation
    R := [ ];
    for Gb in Gbs do
      Add(R, ARepByImages(G, Mon2MatSymR(Gb, G.theGenerators), "hom"));
    od;
    R := 
      ConjugateARep(
        DirectSumARep(R),
        AMatPerm(PermList(Concatenation(G.kbsM)), n)
      );

    # add to reps
    Add(reps, [L, R]);
  od;

  return reps;
end;

# simple example:
#   sym := Mon2IrredSymmetry(DCT_IV(8));
#   List(sym, LR -> LR[1]^AMatMat(DCT_IV(8)) = LR[2]);
#   # hopefully: [true, true]


# The reason for the function Mon2IrredSymmetry1 is the same as
# for PermIrredSymmetry1, explained above.


#F Mon2IrredSymmetry1( <mat/amat> [, <maxblocksize> ] )
#F   calculates a list containing all non-trivial Mon2-Irred-Symmetries
#F   of the given square and invertible matrix M such that the degrees
#F   of all irreducibles is <= maxblocksize 
#F   (default is 2 to avoid expensive calculations).
#F     A Mon2-Irred-Symmetry is a pair [L, R] of AReps of a common group G
#F   such that L is a monomial ARep and R is a direct sum irreducibles 
#F   conjugated with a permutation matrix such that L^AMat(M) = R.
#F   The matrix M must be in characteristic zero and L contains only 
#F   entries [-1, 0, 1].
#F

Mon2IrredSymmetry1 := function ( arg )
  local 
    M, maxblocksize, # arguments
    reps,            # result, pair of AReps
    lat, G,          # lattice of groups, G in lat
    Gbs, Gb,         # block parts of G, an element of Gbs
    L,               # left hand representation
    Rs, R,           # right blocks, combined reps
    n;               # degree of M

  # decode and check arg
  if Length(arg) = 1 then

    # <mat/amat>
    M            := arg[1];
    maxblocksize := 2;

  elif Length(arg) = 2 then

    # <mat/amat>, <maxblocksize>
    M            := arg[1];
    maxblocksize := arg[2];
  
  else
    Error("usage: Mon2IrredSymmetry1( <mat/amat> [, <maxblocksize> ] )");
  fi;
  if IsAMat(M) then
    M := MatAMat(M);
  fi;
  if not IsMat(M) and 0*M[1][1] = 0 then
    Error("<M> must be a matrix in characteristic zero");
  fi;
  if not IsInt(maxblocksize) then
    Error("<maxblocksize> must be integer");
  fi;
  n := Length(M);

  # use Mon2BlockSym and filter out trivial groups
  lat := Mon2BlockSymBySubsets(M, [1..maxblocksize]);
  lat := Filtered(lat, G -> not Size(G) in [1, Factorial(n)*2^n]);

  # filter w.r.t to block size
  lat := 
    Filtered(
      lat, 
      G -> ForAll(G.kbsM, b -> Length(b) <= maxblocksize)
    );

  # find the fully decomposed ones
  for G in lat do
    CharTable(G);
  od;
  if lat = [ ] then
    return [ ];
  fi;
  CompletedMon2BlockSym(lat);
  lat := Filtered(lat, G -> ForAll(G.charactersM, IsIrreducible));

  # make representations
  reps := [ ];
  for G in lat do

    # fix a generating set of G
    G.theGenerators := G.generators;

    # compute groups for the blocks
    Gbs := List(G.kbsM, b -> BlockMon2BlockSym(G, b));

    # the left representation
    L := ARepByImages(G, Mon2MatSymL(Gbs[1], G.theGenerators), "hom");

    # the right representation
    R := [ ];
    for Gb in Gbs do
      Add(R, ARepByImages(G, Mon2MatSymR(Gb, G.theGenerators), "hom"));
    od;
    R := 
      ConjugateARep(
        DirectSumARep(R),
        AMatPerm(PermList(Concatenation(G.kbsM)), n)
      );

    # add to reps
    Add(reps, [L, R]);
  od;

  return reps;
end;
