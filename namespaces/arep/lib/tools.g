# -*- Mode: shell-script -*-
# Universal Tools 
# SE, MP, 26.02.96 - , GAPv3.4

# 26.2.96 first version
# 14.5.97 DiagonalMat, NrMovedPointsPerm

#F Universal tools
#F ===============
#F

#F PartitionIndex( <list> )
#F   compute a partition index B for the given list L with
#F   respect to the equivalence relation '='. That means that
#F   B is a set of sets of integers such that L[i] = L[j] 
#F   for all i, j in b for all b in B.
#F

PartitionIndex := function (list0)
  local list, partition, block, backperm, k;

  if Length(list0) = 0 then
    return [];
  fi;

  list      := ShallowCopy(list0);
  backperm  := Sortex(list)^-1;
  partition := [ ];
  block     := [ 1 ];
  for k in [2..Length(list)] do
    if list[k] = list[block[1]] then
      Add(block, k);
    else
      Add(partition, block);
      block := [ k ];
    fi;
  od;
  Add(partition, block);

  return Set(List(partition, b -> Set(OnTuples(b, backperm))));
end;

#F PartitionRefinement( <partition1>, <partition2> )
#F   compute the common refinement of the two partitions of
#F   the same range.
#F

PartitionRefinement := function (partition1, partition2)
  local partition, block, block1, block2;

  partition := Set([]);
  for block1 in partition1 do
    for block2 in partition2 do
      block := Intersection(block1, block2);
      if Length(block) > 0 then
        AddSet(partition, block);
      fi;
    od;
  od;
  return partition;
end;

#F DirectProductSymmetricGroups( <set-of-set-of-int> ) 
#F   constructs the direct product of the symmetric groups on
#F   the given orbits and adds a base and strong generating set.
#F

DirectProductSymmetricGroups := function ( orbits )
  local group, base, gens, b, n, k;

  base := [ ];
  gens := [ ];
  for b in orbits do
    n := Length(b);
    if n > 1 then
      for k in [1..n-1] do
        Add(base, b[k]);
        Add(gens, (b[k], b[n]));
      od;
    fi;
  od;
  group := Group(gens, ());
  MakeStabChainStrongGenerators(group, base, gens);
  return group;
end;


#F Constructions for matrices
#F ==========================
#F

#F DiagonalMat( <list> )
#F   the diagonal matrix with <list> on the diagonal.
#F

DiagonalMat := function ( diag )
  local A, i;

  if not ( 
    IsList(diag) and 
    Length(diag) >= 1 and 
    ForAll(diag, x -> x in FieldElements)
  ) then
    Error("<diag> must be a non-empty list of scalars");
  fi;

  A := IdentityMat(Length(diag), DefaultField(diag));
  for i in [1..Length(diag)] do
    A[i][i] := diag[i];
  od;
  return A;  
end;

#F DirectSumMat( <list-of-mat>) 
#F DirectSumMat( <mat>, .., <mat> )
#F   form the direct sum of matrices (block diagonal).
#F

DirectSumMat := function (arg)
  local S, A, i, z, columnsS, columnsA, row;

  if Length(arg) > 1 then
    return DirectSumMat(arg);
  fi;
  if not (Length(arg[1]) > 0 and ForAll(arg[1], IsMat)) then
    Error(
      Concatenation(
        "usage: DirectSumMat( <list-of-mat> )\n",
        "       DirectSumMat( <mat>, .., <mat> )"
      )
    );
  fi;

  S := List(arg[1][1], ShallowCopy);
  for i in [2..Length(arg[1])] do
    A := List(arg[1][i], ShallowCopy);

    # S := diag(S, A);
    columnsS := Length(S[1]);
    columnsA := Length(A[1]);
    z := List([1..columnsA], i -> 0*S[1][1]);
    for row in S do
      Append(row, z);
    od;
    z := List([1..columnsS], i -> 0*A[1][1]);
    for row in A do
      Add(S, Concatenation(z, row));
    od;
  od;
  return S;
end;

#F TensorProductMat( <list-of-mat> )
#F TensorProductMat( <mat>, .., <mat> )
#F   constructs the Kronecker product of the
#F   matrices given; (the leftmost is the outermost
#F   structure)
#F

TensorProductMat := function ( arg )
  local TA, T, rT, cT, iT, jT, A, rA, cA, iA, jA, k;
  
  if Length(arg) > 1 then
    return TensorProductMat(arg);
  fi;
  if not (Length(arg[1]) > 0 and ForAll(arg[1], IsMat)) then
    Error(
      Concatenation(
        "usage: TensorProductMat( <list-of-mat> )\n",
        "       TensorProductMat( <mat>, .., <mat> )"
      )
    );
  fi;

  T := arg[1][1];
  for k in [2..Length(arg[1])] do
    A := arg[1][k];

    # T := T kronecker* A
    rT := Length(T);
    cT := Length(T[1]);
    rA := Length(A);
    cA := Length(A[1]);
    TA := NullMat(rT*rA, cT*cA);
    for iT in [1..rT] do
      for iA in [1..rA] do
	for jT in [1..cT] do
	  for jA in [1..cA] do
	    TA[(iT-1)*rA + iA][(jT-1)*cA + jA] := T[iT][jT] * A[iA][jA];
	  od;
	od;
      od;
    od;
    T := TA;
  od;

  return T;
end;


#F Permutation matrices
#F ====================
#F
#F MatPerm( <perm>, <degree> [, <char/field>] )
#F   constructs the permutation matrix A for the permutation g
#F   such that MatPerm(g1*g2, n) = MatPerm(g1, n)*MatPerm(g2, n).
#F   Note that PermMat(MatPerm(g, n)) = g for all permutations g.
#F   This means MatPerm(g, n) = [ delta[i^g, j] ]_ij for example
#F   MatPerm((1,2,3), 3) = 
#F     [ [0, 1, 0], 
#F       [0, 0, 1], 
#F       [1, 0, 0] ].
#F

MatPerm := function (arg)
  local s, d, c, lmp;

  lmp := function (x)
    if x = () then 
      return 0; 
    else 
      return LargestMovedPointPerm(x); 
    fi;
  end;

  # check arguments; get permutation s, degree d, and char c
  if Length(arg) in [2, 3] and arg[1] = false then
    return false;
  elif
    Length(arg) = 2 and
    IsPerm(arg[1]) and 
    IsInt(arg[2]) and
    arg[2] >= lmp(arg[1])
  then
    s := arg[1];
    d := arg[2];
    c := 0;
  elif 
    Length(arg) = 3 and
    IsPerm(arg[1]) and 
    IsInt(arg[2]) and
    arg[2] >= lmp(arg[1])
  then
    s := arg[1];
    d := arg[2];
    if IsField(arg[3]) then
      c := Characteristic(arg[3]);
    elif arg[3] = 0 then
      c := 0;
    elif IsInt(arg[3]) and arg[3] > 0 and IsPrimeInt(arg[3]) then
      c := arg[3];
    else
      Error("<arg[3]> must be 0, prime or field");
    fi;
  else
    Error("usage: MatPerm( <perm>, <degree> [, <char/field> ] )");
  fi;

  if not (
    IsPerm(s) and 
    IsInt(d) and 
    d >= lmp(s)
  ) then
    Error("usage: MatPerm( <perm>, <degree> )");
  fi;

  if c = 0 then
    return Permuted(IdentityMat(d, 0), s^-1);
  else
    return Permuted(IdentityMat(d, GF(c)), s^-1);
  fi;
end;

#F PermMat( <square-mat> )
#F   constructs the permutation represented by the permutation
#F   matrix given or returns false if the matrix is no 
#F   permutation matrix. The convention used matches MatPerm
#F   that is PermMat(MatPerm(g, n)) = g for all permutations g.
#F

PermMat := function ( A )
  local s,     # s[i] is index of 1 in A[i]
        zeros, # number of 0s in A[i]
        Ai,    # A[i]
        j;     # counter 

  if A = false then
    return false;
  fi;
  if not (IsMat(A) and Length(A) = Length(A[1])) then
    Error("usage: PermMat( <square-mat> )");
  fi;
  s := [];
  for Ai in A do
    zeros := 0;
    for j in [1..Length(Ai)] do
      if Ai[j] = 0*Ai[j] then
        zeros := zeros+1;
      elif Ai[j] = Ai[j]^0 then
        Add(s, j);
      else
        return false;
      fi;
    od;
    if zeros <> Length(Ai)-1 then
      return false;
    fi;
  od;
  if Set(s) <> [1..Length(A)] then
    return false;
  fi;
  return PermList(s);
end;

#F PermutedMat( <perm>, <mat>, <perm> )
#F   given sL, A, sR compute MatPerm(sL, nr) * A * MatPerm(sR, nc)
#F   for the (nr x nc)-matrix A. This function is more efficient
#F   than its definition.
#F

PermutedMat := function ( sL, A, sR )
  local nr, nc, sRInv;

  if not ( IsMat(A) and IsPerm(sL) and IsPerm(sR) ) then
    Error("usage: PermutedMat( <perm>, <mat>, <perm> )");
  fi;
  nr := Length(A);
  nc := Length(A[1]);
  if not ( 
    (sL = () or LargestMovedPointPerm(sL) <= nr) and
    (sR = () or LargestMovedPointPerm(sR) <= nc)
  ) then
    Error("permutation moves to large points");
  fi;
  sRInv := sR^-1;
  return List([1..nr], r -> List([1..nc], c -> A[r^sL][c^sRInv]));
end;

#F DirectSumPerm(ns, gs)
#F   constructs the permutation which encodes the direct sum of the
#F   permutation matrices for the permutations gs of degrees ns.
#F   gs is a list of permutations, ns is a list of positive integers 
#F   and Length(ns) = Length(gs).
#F

DirectSumPerm := function (ns, gs)
  local s, k, i,t;

  if not (
    IsList(ns) and 
    IsList(gs) and 
    Length(ns) = Length(gs) and
    ForAll(ns, n -> IsInt(n) and n >= 0) and
    ForAll(gs, g -> IsPerm(g)) and
    ForAll([1..Length(ns)], 
      i -> gs[i] = () or 
           ns[i] >= LargestMovedPointPerm(gs[i])
    )
  ) then
    Error("usage: DirectSumPerm( <lst-of-degs> <lst-of-perm> )");
  fi;

  s := [];
  k := 0;
  for i in [1..Length(ns)] do
    for t in [1..ns[i]] do
      Add(s, k + t^gs[i]);
    od;
    k := k + ns[i];
  od;
  return PermList(s);
end;

#F TensorProductPerm(ns, gs)
#F   constructs the permutation matrix for the Kronecker product
#F   of the permutation matrices for the permutations in gs of
#F   degree given by the corresponding ns.
#F

TensorProductPerm := function (ns, gs)
  local n,s, i,ni,gi;

  if not (
    IsList(ns) and 
    IsList(gs) and 
    Length(ns) = Length(gs) and
    ForAll(ns, n -> IsInt(n) and n > 0) and
    ForAll(gs, g -> IsPerm(g)) and
    ForAll(
      [1..Length(ns)], 
      i -> gs[i] = () or ns[i] >= LargestMovedPointPerm( gs[i] )
    )
  ) then
    Error("usage: TensorProductPerm( <lst-of-degs> <lst-of-perm> )");
  fi;

  n := 1;
  s := [1];
  for i in [Length(ns), Length(ns)-1 .. 1] do
    gi := gs[i];
    ni := ns[i];

    # (n, s) := (ni, gi) tensor* (n, s)
    s := Concatenation( List([1..ni], k -> (k^gi - 1)*n + s) );
    n := ni*n;     
  od;
  return PermList(s);
end;

#F MovedPointsPerm( <perm> )
#F NrMovedPointsPerm( <perm> )
#F   the set or nr. of moved points of <perm>.
#F

MovedPointsPerm := function ( x )
  local mp, i;

  if x = () then
    return [ ];
  else
    mp := [ ];
    for i in [1..LargestMovedPointPerm(x)] do
      if i^x <> i then
        Add(mp, i);
      fi;
    od;
    IsSet(mp);
    return mp;
  fi;
end;

NrMovedPointsPerm := function ( x )
  local mp, i;

  if x = () then
    return 0;
  else
    mp := 0;
    for i in [1..LargestMovedPointPerm(x)] do
      if i^x <> i then
        mp := mp + 1;
      fi;
    od;
    return mp;
  fi;
end;

#F PermOfCycleType( <cycle-type> )
#F   the smallest permutation of cycle type ct.
#F   The cycle type ct is of the form 
#F     [ .., [ cyclelength, nr ], ..]
#F

PermOfCycleType := function ( ct )
  local deg, L, n, c, i;

  if not (
    IsList(ct) and
    ForAll(
      ct, 
      c -> 
        IsList(c) and 
        Length(c) = 2 and
        ForAll(c, x -> IsInt(x) and x > 0))
    ) then
    Error("<ct> must be a cycle type");
  fi;

  deg := Sum(List(ct, Product));
  L   := [ ];
  n   := 0;
  for c in ct do
    for i in [1..c[1] * c[2]] do
      L[n + i] := n + i + 1;
    od;
    for i in [1..c[2]] do
      L[n + i * c[1]] := n + i * c[1] - c[1] + 1;
    od;
    n := n + c[1] * c[2];
  od;
  
  return PermList(L);
end;


#F Block structure of matrices
#F ===========================
#F

# Lemma1:
#   If A has a non-trivial permuted block structure then
#   |rbs(A)| = |cbs(A)| > 1.
# Proof1:
#   By rearranging rows and columns suitably we may assume
#   A = diag(A1, .., Ar), r > 1, is maximally decomposed. 
#   Then |cbs(A)| = |rbd(A)| > 1 because otherwise a block
#   would split further. q.e.d.


#F SupportVector( <vec> )
#F   computes the set of indices i such that x[i] <> 0.
#F

SupportVector := function ( x )
  return Set( Filtered([1..Length(x)], i -> x[i] <> 0*x[i]) );
end;


#F RowBlockStructureMat( <mat> )
#F   is ColumnBlockStructureMat( TransposedMat( A ) ).
#F   Use PermutedMat(PermList(Concatenation( CBS(A) )), A, ())
#F   to arrange the row blocks in standard order.
#F

RowBlockStructureMat := function ( A )
  local rbs,  # list of sets of row indices; result
        b,    # b[r] is set of row indices containing r
        bc,   # block merged by column c
        Ac,   # A[*][c]
        r, c; # a row, column index

  if not IsMat(A) then
    Error("usage: RowBlockStructureMat( <mat> )");
  fi;

  b := List([1..Length(A)], r -> Set([ r ]));
  for c in [1..Length(A[1])] do
    bc := Set([ ]);
    for r in [1..Length(A)] do
      if A[r][c] <> 0*A[r][c] then
        UniteSet(bc, b[r]);
      fi;
    od;
    for r in bc do
      b[r] := bc;
    od;
  od;
  rbs := Set( b );
  Sort(
    rbs,
    function (b1, b2)
      if Length(b1) < Length(b2) then return true;  fi;
      if Length(b1) > Length(b2) then return false; fi;
      return b1 < b2;
    end
  );
  return rbs;
end;


#F ColumnBlockStructureMat( <mat> )
#F   computes a sorted list of sets of column indices
#F   for the column blocks of the matrix. The result
#F   is sorted such that small sets are in front and
#F   the sets of equal size are in standard ('<') order.
#F   Use PermutedMat((), A, PermList(Concatenation( CBS(A) ))^-1)
#F   to arrange the column blocks in standard order.
#F

ColumnBlockStructureMat := function ( A )
  local cbs, # list of sets of column indices; result
        b,   # b[c] is set of column indices containing c
        br,  # block merged by row r
        Ar,  # A[r]
        c;   # a column index

  if not IsMat(A) then
    Error("usage: ColumnBlockStructureMat( <mat> )");
  fi;

  b := List([1..Length(A[1])], c -> Set([ c ]));
  for Ar in A do
    br := Set([ ]);
    for c in SupportVector(Ar) do
      UniteSet(br, b[c]);
    od;
    for c in br do
      b[c] := br;
    od;
  od;
  cbs := Set( b );
  Sort(
    cbs,
    function (b1, b2)
      if Length(b1) < Length(b2) then return true;  fi;
      if Length(b1) > Length(b2) then return false; fi;
      return b1 < b2;
    end
  );
  return cbs;
end;


#F BlockStructureMat( <mat> )
#F   computes a pair [rbs, cbs] of matching row and column
#F   block structures of the matrix. That is |rbs| = |cbs| and
#F   the set of pairs (i, j) of indices for the k-th block
#F   is exactly Cartesian(rbs[k], cbs[k]).
#F

# BlockStructureMat reduces the matrix to a possibly smaller
# matrix without zero rows and zero columns. Then it calls
# BlockStructureMat1 to do the job on the smaller matrix.

BlockStructureMat1 := function ( A )
  local nr, nc,      # number of rows, columns of A
        rbs, cbs,    # row and column block structure
        nb,          # number of blocks
        br, c, cbs1, # temporaries
        brZ, bcZ;    # the all-zeros block

  # get size of A
  nr := Length(A);
  nc := Length(A[1]);

  # compute rbs and cbs first
  rbs := RowBlockStructureMat(A);
  if Length(rbs) = 1 then
    return [ [[1..nr]], [[1..nc]] ];
  fi;
  cbs := ColumnBlockStructureMat(A);
  if Length(cbs) <> Length(rbs) then
    return [ [[1..nr]], [[1..nc]] ];
  fi;
  nb := Length(rbs); # = Length(cbs);

  # rearrange cbs such that Cartesian(rbs[i], cbs[i]) is a 
  # block for all i in [1..nb]
  cbs1 := [ ];
  for br in rbs do
    c :=
      First(
        [1..nc],
        c -> ForAny(br, r -> A[r][c] <> 0*A[r][c])
      );
    Add(cbs1, First(cbs, bc -> c in bc));
  od;

  return [rbs, cbs1];
end;

BlockStructureMat := function ( A )
  local 
    bs,       # the result
    A1,       # reduced matrix
    bs1,      # the result on A1
    nr,  nc,  # nr. of rows, columns of A
    rz,  cz,  # the zero block
    rzc, czc; # the complement block

  if not IsMat(A) then
    Error("usage: BlockStructureMat( <mat> )");
  fi;
  nr := Length(A);
  nc := Length(A[1]);

  # find the zero rows, columns
  rz := Filtered([1..nr], r -> ForAll([1..nc], c -> A[r][c] = 0*A[r][c]));
  cz := Filtered([1..nc], c -> ForAll([1..nr], r -> A[r][c] = 0*A[r][c]));
  if rz = [ ] and cz = [ ] then
    return BlockStructureMat1(A);
  fi;
  rzc := Difference([1..nr], rz);
  czc := Difference([1..nc], cz);

  # construct smaller matrix
  A1 :=
    List(
      Sublist(A, rzc),
      Ar -> Sublist(Ar, czc)
    );
  
  # delegate work to BlockStructureMat1
  bs1 := BlockStructureMat1(A1);
  
  # remap the points of bs1 
  bs := 
    [ List(bs1[1], rb -> Sublist(rzc, rb)),
      List(bs1[2], cb -> Sublist(czc, cb))
    ];

  # add the zero block
  Add(bs[1], rz);
  Add(bs[2], cz);
  return bs;
end;


