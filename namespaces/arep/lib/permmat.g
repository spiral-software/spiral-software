# -*- Mode: shell-script -*-
# Determination of Perm-Irred-Symmetry
# SE, 07.02.97 - , GAPv3.4

# SE, 7.--9.2.97, GAP v3.4
# SE, MP, 10.2.97: Fehler; Young-Gr. falsch berechnet

# SelectBaseFromList( <list> )
#   selects a base from the <list> of vectors considering
#   one after the other. The function a set of indices into
#   the list to pick out linear independent elements.
#

SelectBaseFromList := function ( list )
  local indices, base, i;

  if not IsMat(list) then
    Error("<list> must be a list of vectors");
  fi;

  indices := [ ];
  base    := [ ];
  for i in [1..Length(list)] do
    Add(base, ShallowCopy(list[i]));
    TriangulizeMat(base);
    if ForAll(base[Length(base)], x -> x = 0*x) then
      Unbind(base[Length(base)]);
    else
      Add(indices, i);
    fi;
  od;
  return indices;
end;

#F PermMatSym( <mat> )
#F PermMatSymL( <sym>, <x> )
#F PermMatSymR( <sym>, <x> )
#F   the Perm-Mat-symmetry of the (nr x nc)-matrix mat
#F   where nr >= nc and mat does not contain linear
#F   dependent columns. The call PermMatSym(mat) returns
#F   a permgroup sym on [1..nr] such that for all x in sym
#F
#F       MatPerm(PermMatSymL(sym, x), nr) * mat 
#F     = mat * PermMatSymR(sym, x).
#F
#F   (You can use PermutedMat for the left part.) The
#F   group record sym contains the additional fields
#F     sym.matrix       identical to mat
#F     sym.baseIndices  set of row indices forming a base
#F     sym.baseMatrix   Sublist(mat, baseIndices)^-1
#F   to be used in PermMatSymL and PermMatSymR. The <x>
#F   passed to PermMatSymL or PermMatSymR can also be
#F   a list of permutations or a permgroup.
#F

if not IsBound(InfoPermMat1) then
  InfoPermMat1 := Ignore;
fi;
if not IsBound(InfoPermMat2) then
  InfoPermMat2 := Ignore;
fi;

PermMatSymL := function ( sym, x )
  if IsPerm(x) then
    return x;
  elif IsList(x) then
    return x;
  elif IsPermGroup(x) then
    if not ( IsPermGroup(sym) and IsBound(sym.matrix) ) then
      Error("<sym> must be permgroup with sym.matrix etc.");
    fi;
    x.matrix      := sym.matrix;
    x.baseIndices := sym.baseIndices;
    x.baseMatrix  := sym.baseMatrix;
    return x;
  else
    Error("<x> must be perm, list-of-perm or permgroup");
  fi;
end;

PermMatSymR := function ( sym, x )
  if IsPerm(x) then
    if not ( IsPermGroup(sym) and IsBound(sym.matrix) ) then
      Error("<sym> must be permgroup with sym.matrix etc.");
    fi;
    return
      sym.baseMatrix * 
      Sublist(sym.matrix, OnTuples(sym.baseIndices, x));
  elif IsList(x) then
    return List(x, x1 -> PermMatSymR(sym, x1));
  elif IsPermGroup(x) then
    return 
      Group(
        PermMatSymR(sym, x.generators),
        sym.baseMatrix^0
      );
  else
    Error("<x> must be perm, list-of-perm or permgroup");
  fi;
end;

PermMatSym := function ( M0 )
  local
    sym0,      # the symmetry of M0; resul
    nr0, nc0,  # nr. of rows, columns of M0
    M,         # sorted selection of unequal rows of M0
    sym,       # Perm-Mat-symmetry of M
    part,      # partition index underlying M
    setM,      # the set of components of M
    idxM,      # M with entries mapped to [1..Length(setM)]
    nr, nc,    # nr. of rows, columns of M
    I,         # a set of indices for rows of M
    Ic,        # the complement Difference([1..nr], I)
    InvMI,     # precomputed matrix M[I,*]^-1
    MInvMI,    # precomputed matrix M * M[I,*]^-1
    computeL,  # func to compute L from OnTuples(I, L^-1)
    extendSym, # func to extend sym recursively
    complete;  # func to complete sym into sym0

  # decode and check arg
  if not IsMat(M0) then
    Error("<M0> must be matrix");
  fi;
  nr0 := Length(M0);
  nc0 := Length(M0[1]);
  if not nr0 >= nc0 then
    Error("<M0> must be matrix with more rows than columns");
  fi;
  InfoPermMat1("#I considering (", nr0, " x ", nc0, ")-matrix\n");

  # partition the set of rows of M0 wrt. equality;
  # arrange the blocks in part such that M is sorted
  part := PartitionIndex(M0);
  M    := Sublist(M0, List(part, b -> b[1]));
  part := Permuted(part, Sortex(M));
  nr   := Length(M);
  nc   := Length(M[1]);
  setM := Union(M);
  idxM := List(M, Mi -> List(Mi, Mij -> Position(setM, Mij)));
  InfoPermMat1("#I blocks of equal rows  = ", part, "\n");

  # choose linear independent rows I
  I  := SelectBaseFromList(M);
  Ic := Difference([1..nr], I);
  InfoPermMat1("#I chosen base of blocks = ", I, "; \c");
  InvMI  := Sublist(M, I)^-1;
  MInvMI := M * InvMI; # (1)
  InfoPermMat1("matrices precomputed\n");

  # computeL( <IInvL> )
  #   determines L such that I^(L^-1) = IInvL and
  #     M * M[I,*]^-1 * M[I^(L^-1),*] = L^-1 * M.
  #   The permutation L is uniquely determined if it
  #   exists because M does not contain duplicate rows.
  #   If no such L exists then false is returned.

  # Implementation notes
  #   (1) M * M[I,*]^-1 is precomputed.
  #   (2) M * M[I,*]^-1 * M[IInvL,*] is computed component
  #       per component; if a value occurs which is not in M
  #       then there is no permutation L.
  #   (3) Look up the index xi of the row 
  #         ( M * M[I,*]^-1 * M[IInvL,*] )[i,*]
  #       in the rows of M. Then i^(L^-1) = xi if there is an
  #       L at all. Hence, if no xi exists then there is no L.
  #   (4) Furthermore, L only maps blocks of M0 which contain
  #       the same number of equal rows (of M0).

  computeL := 
    function ( IInvL )
      local 
        invL,    # OnTuples([1..nr], L^-1)
        setInvL, # set of points in invL
        xi,      # row (L^-1 * M)[i,*] or its index into idxM
        xij,     # component xi[j] or its index into setM
        i, j, k; # index counters

      # use IInvL directly
      invL := [ ];
      for k in [1..Length(I)] do
        invL[ I[k] ] := IInvL[k];
      od;
      setInvL := Set(IInvL);

      # find the preimage on the rest
      for i in Ic do

        # compute xi = ( M * M[I,*]^-1 * M[I^(L^-1),*] )[i,*]
        # which should equal ( L^-1 * M )[i,*] = M[i^(L^-1),*]
        xi := [ ];
        for j in [1..nc] do

          # compute xij = ( M * M[I,*]^-1 * M[I^(L^-1),*] )[i,j] 
          # which equals ( L^-1 * M )[i,j] for the correct L
          xij := MInvMI[i][1] * M[IInvL[1]][j];
          for k in [2..nc] do
            xij := xij + MInvMI[i][k] * M[IInvL[k]][j];
          od;

          # check if xij is a number from M at all, (2)
          xij := Position(setM, xij);
          if xij = false then
            return false;
          fi;

          Add(xi, xij);
        od;

        # check if xi is a new row from M mapping blocks
        # of equal number of rows of M0, (3), (4)
        xi := Position(idxM, xi);
        if not ( 
          xi <> false and 
          Length(part[i]) = Length(part[xi]) and
          not xi in setInvL
        ) then
          return false;
        fi;

        invL[i] := xi;
        AddSet(setInvL, xi);
      od;
      return PermList(invL)^-1;
    end;

  # extendSym( <J> )
  #   extends the group sym by adding all L which
  #   map Sublist(I, [1..Length(J)])^(L^-1) = J. 
  #   The list J is an initial part of IInvL.

  # Implementation notes
  #   (5) I is a base for sym. Hence, only preimages J
  #       of I which are not in Orbit(sym, I, OnTuples)
  #       have to be considered at any moment.

  extendSym := function ( J )
    local n, i, L;

    if Length(J) = Length(I) then

      # find an L with I^(L^-1) = J if possible
      InfoPermMat2(J, " \c");
      L := computeL(J);
      if L <> false and not L in sym then
        sym := Closure(sym, L);
      fi;
      return;

    fi;

    n := Length(J)+1;
    for i in [1..nr] do
      if 
        Length(part[i]) = Length(part[I[n]]) and # (4)
        not i in J     # J does not contain duplicates
      then

        # backtracking on IInvL
        J[n] := i;
        if
          RepresentativeOperation(
            sym, 
            Sublist(I, [1..n]), 
            J, 
            OnTuples
          ) = false # (5)
        then
          extendSym(J);
        fi;
        Unbind(J[n]);

      fi;
    od;
  end;

  InfoPermMat2("#I considering \c");
  sym := Subgroup(SymmetricGroup(nr), [ ]);
  extendSym([ ]);
  InfoPermMat2("\n");

  # complete( <sym>, <part> )
  #   remapps <sym> to act on the blocks in part and
  #   adds symmetric groups within the blocks (the 
  #   Young group of <part>).

  complete := function ( sym, part )
    local
      grp,  # the resulting grup
      base, # base of the result
      sgs,  # strong generating set of the result
      b,    # a block in part
      k;    # index running through b

    # get base and strong generating set from sym
    base := Base(sym);
    sgs  := PermGroupOps.StrongGenerators(sym);

    # map base, sgs into action on Union(part)
    base  := List(base, ib -> part[ib][1]);
    sgs   := 
      List(
        sgs,
        g -> 
          MappingPermListList(
            Concatenation(part), 
            Concatenation(Permuted(part, g))
          )
      );

    # add symmetric groups on the blocks
    for b in part do
      if Length(b) > 1 then
        if b[1] in base then
          Append(base, Sublist(b, [2..Length(b)-1]));
        else
          Append(base, Sublist(b, [1..Length(b)-1]));
        fi;
        for k in [1..Length(b)-1] do
          Add(sgs, (b[k], b[Length(b)]));
        od;
      fi;
    od;

    grp := Group(sgs, ());
    MakeStabChainStrongGenerators(grp, base, sgs);
    return grp;
  end;

  sym0 := complete(sym, part);
  sym0.matrix      := M0;
  sym0.baseIndices := List(I, i -> part[i][1]);
  sym0.baseMatrix  := InvMI;

  InfoPermMat1("#I => size of sym = ", Size(sym0), "\n\n");
  return sym0;
end;

#F PermMatSymNormalL( <sym> )
#F   the subgroup ker of sym = PermMatSym(mat) containing
#F   all those L for which PermMatSymR(sym, L) = IdMat.
#F

PermMatSymNormalL := function ( sym )
  local ker, part, sgs, b, k;

  # construct the Young group of the row partition
  part := PartitionIndex(sym.matrix);
  sgs  := [ ];
  for b in part do
    for k in [1..Length(b)-1] do
      Add(sgs, (b[k], b[Length(b)]));
    od;
  od;

  # add base and strong generators and specific information
  ker := Subgroup(sym, sgs);
  ker.matrix      := sym.matrix;
  ker.baseIndices := sym.baseIndices;
  ker.baseMatrix  := sym.baseMatrix;
  return ker;
end;

# Bsp.
# M0 := List(TransposedMat(DCT(6)), x -> Sublist(x, [3,4]));
# PM := [ (), (1,3)(2,5)(4,6), (1,4)(3,6), (1,6)(2,5)(3,4) ];

# L1 := (1,2,3,4,5,6);
# R1 := [[0, 1], [E(3), 0]];
# M0 := List([1..6], i -> List([1..2], j -> Random(CF(5))));
# M0 := 1/6*Sum(List([0..5], k -> MatPerm(L1^k,6)*M0*R1^(-k)));
