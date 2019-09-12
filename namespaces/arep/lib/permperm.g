# -*- Mode: shell-script -*-
# Determination of Perm-Perm-Symmetry
# SE, 09.03.96 - , GAPv3.4

# SE, GAP v3.4
# SE,      9.3.96  erste Version
# SE,     10.2.97  umgestellt auf "L M = M R" Konvention 
# SE, MP, 12.12.97 'leonsym.g' sauber integriert
# SE, 16.01.01     PermPermSymLeon() verbessert: Option "-mb:<int>" wird
#   an Leon's 'desauto' durchgegeben damit auch groessere Basislaengen
#   moeglich sind (default war 62); L M = M R wird gecheckt
# SE, 06.02.01    PermPerm verwendet FewGenerators

# Literature:
#   G. Butler: Fundamental algorithms for permutation groups, 
#              LNCS 559.

if not IsBound(InfoPermSym1) then
  InfoPermSym1 := Ignore;
fi;
if not IsBound(InfoPermSym2) then
  InfoPermSym2 := Ignore;  
fi;

#F FewGenerators( <group> [, <trials> ] )
#F   a small generating set for the given group. The function does not
#F   try very hard to minimize the number of generators, it just takes
#F   random elements until they generate the group. If the optional
#F   argument trials (default 3) is passed the process is repeated for
#F   the specified number of iterations.
#F     The function memorizes the result in the field G.fewGenerators.
#F   If G.fewGenerators is already bound, then the function still runs
#F   the specified number of iterations to improve it and if it finds
#F   something shorter it also updates G.fewGenerators. Hence, you must
#F   not assume that G.fewGenerators is constant.
#F

FewGenerators := function ( arg )
  local
    G, trials, # arguments
    S,         # result
    H,         # a subgroup of G
    t;         # the trial

  # decode arg
  if Length(arg) = 1 then
    G      := arg[1];
    trials := 3;
  elif Length(arg) = 2 then
    G      := arg[1];
    trials := arg[2];
  else
    Error("usage: FewGenerators( <group> [, <trials> ] )");
  fi;

  # check arguments
  if not IsGroup(G) then
    Error("<G> must be a group");
  fi;
  if not ( IsInt(trials) and trials >= 0 ) then
    Error("<trials> must be a non-negative integer");
  fi;

  # use memorization
  if not IsBound(G.fewGenerators) then
    G.fewGenerators := Generators(G);
  fi;

  # improve
  for t in [1..trials] do
    H := Subgroup(G, []);
    while not Index(G, H) = 1 do
      H := Closure(H, Random(G));
    od;
    if Length(H.generators) < Length(G.fewGenerators) then
      G.fewGenerators := H.generators;
    fi;
  od;
  return G.fewGenerators;
end;


#F PermPermSym( <mat> )
#F PermPermSymL( <sym>, <x> )
#F PermPermSymR( <sym>, <x> )
#F   the Perm-Perm-symmetry of the matrix mat. If mat is a 
#F   (nr x nc)-matrix then the PermPermSym(mat) returns a 
#F   permgroup sym on [1..nr+nc] which represents the Perm-
#F   Perm-symmetry of mat. The two functions PermPermSymL 
#F   and PermPermSymR are the projections onto the left and 
#F   right component. In fact, for all x in sym
#F  
#F       MatPerm(PermPermSymL(sym, x), nr) * mat
#F     = mat * MatPerm(PermPermSymR(sym, x), nc).
#F
#F   (Work with PermutedMat(L, mat, R^-1) = mat to check.) 
#F   The group record sym contains the additional field
#F     sym.dimensionsMat = [nr, nc]
#F   to be used in PermPermSymL and PermPermSymR. The <x>
#F   passed to the projections may also be a list or a 
#F   permgroup to be projected.
#F      If the global variable UseLeon is set to true
#F   then the function PermPermSym tries to call a 
#F   C-implementation by Jeffrey Leon instead of using 
#F   the GAP-program. In this case the external C-programs
#F   'desauto', 'leonin' and 'leonout'  has to be available.
#F

if not IsBound(InfoPermSym1) then InfoPermSym1 := Ignore; fi;
if not IsBound(InfoPermSym2) then InfoPermSym2 := Ignore; fi;

PermPermSymL := function ( sym, x )
  local nr;

  if IsPerm(x) then
    if not ( IsPermGroup(sym) and IsBound(sym.dimensionsMat) ) then
      Error("<sym> must be permgroup with sym.dimensionsMat");
    fi;
    nr := sym.dimensionsMat[1];
    return RestrictedPerm(x, [1..nr]);
  elif IsList(x) then
    return List(x, x1 -> PermPermSymL(sym, x1));
  elif IsPermGroup(x) then
    return Group(PermPermSymL(sym, x.generators), ());
  else
    Error("<x> must be perm, list-of-perm or permgroup");
  fi;
end;

PermPermSymR := function ( sym, x )
  local nr, nc;

  if IsPerm(x) then
    if not ( IsPermGroup(sym) and IsBound(sym.dimensionsMat) ) then
      Error("<sym> must be permgroup with sym.dimensionsMat");
    fi;
    nr := sym.dimensionsMat[1];
    nc := sym.dimensionsMat[2];
    return PermList(OnTuples([nr+1..nr+nc], x) - nr);
  elif IsList(x) then
    return List(x, x1 -> PermPermSymR(sym, x1));
  elif IsPermGroup(x) then
    return Group(PermPermSymR(sym, x.generators), ());
  else
    Error("<x> must be perm, list-of-perm or permgroup");
  fi;
end;

PermPermSymLeon := "defined below";
if not IsBound(UseLeon) then
  UseLeon := false;
fi;


PermPermSym_checkAndPolish := function ( G, M ) 
  local G1;

  # check if G is a PermGroup
  if not IsPermGroup(G) then
    Error("Panic! PermPermSym failed: <G> not a PermGroup");
  fi;

  # change the generating set of G by a short one 
  InfoPermSym1("#I   searching for few generators\n");
  G1 := Group(FewGenerators(G), ());
  MakeStabChainStrongGenerators(
    G1,
    Base(G),
    PermGroupOps.StrongGenerators(G)
  );
  G1.dimensionsMat := G.dimensionsMat;
 
  # check if L*M= M*R for all the generators
  InfoPermSym1(
    "#I   checking ", Length(G1.generators), " generators for PermPermSym\n"
  );
  if not 
    ForAll(
      G1.generators,
      g -> PermutedMat(PermPermSymL(G, g), M, PermPermSymR(G, g)^-1) = M
    )
  then
    Error("Panic! PermPermSym failed: <G> contains non-symmetries\n");
  fi;
  return G1; 
end;


PermPermSym := function ( A )
  local 
    nr, nc,     # number of rows, columns of A
    Br, Bc,     # partitions of rows, columns wrt. totals
    Er, Ec,     # partitions of rows, columns wrt. equality
    G, K,       # group to search, known subgroup
    w,          # vector to represent restrictions on basepoint images
    orbitG,     # orbitG[i] = Set(Orbit(G, i)) used for Blists

    # preprocessed information for backtrack-search
    blocksA,
    blocksA1,

    # local functions
    remappedMat_Partitions,
    youngGroup,
    chooseBasepointImage,
    init_chooseBasepointImage,
    elementInCoset,
    findSymmetry;

  # check argument; A may be a list of lists of anything, which
  # need not be a matrix. Thus IsMat() can not be used.
  if not ( 
    IsList(A) and 
    Length(A) >= 1 and
    Length(A[1]) >= 1 and
    ForAll(A, Ai -> Length(Ai) = Length(A[1]))
  ) then
    Error("usage: PermPermSym( <mat> )");
  fi;
  nr := Length(A);
  nc := Length(A[1]);
 
  # use J. Leon's program if desired and applicable
  if UseLeon then
    G := PermPermSymLeon(A);
    if G <> false then
      return PermPermSym_checkAndPolish(G, A);
    fi;
  fi;
  InfoPermSym1(
    "#I PermPermSym( <", nr, "x", nc, "-matrix> ) called\n"
  );

  # Preprocessing of A using row- and column-totals
  # -----------------------------------------------

  # First the values in A are remapped into [1..v] and the
  # row- and column-totals are used to partition the row- and
  # column-index sets repeatedly until the partitions stabilize.

  remappedMat_Partitions := function ( A0, nr, nc )
    local 
      A, A_r, A_c,          # the remapped A; a row, a column, a value
      A0_r, A0_rc,          # row of A0; a component
      Br, Bc, br, bc,       # partitions of row-, column index set; blocks
      Abr, Abc,             # submatrix A[br][*], A[*][bc]
      Length_Br, Length_Bc, # Length(Br), Length(Bc)
      set,                  # set, list of values in a matrix
      L,                    # lengths of blocks (for Info only)
      r, c, i, v,           # counter for row, column, block, value
      iteration,            # counter for iterations
      partitionIndex,       # local functions
      refineBlock;

    partitionIndex := function ( list ) # modifies argument
      local backperm, partition, block, k;

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
      return List(partition, b -> OnTuples(b, backperm));
    end;

    refineBlock := function ( partition, index, blockpartition )
      local block, i;

      block            := partition[index];
      partition[index] := Sublist(block, blockpartition[1]);
      for i in [2..Length(blockpartition)] do
        Add(partition, Sublist(block, blockpartition[i]));
      od;

      for i in [1..Length(partition)] do
        partition[i] := Set(partition[i]);
      od;
      Sort(partition);
    end;

    # initially rename A 
    InfoPermSym1("#I   remapping values into [1..\c");
    nr  := Length(A0);
    nc  := Length(A0[1]);
    set := Set(Union(A0));
    A := [];
    for A0_r in A0 do
      A_r := [];
      for A0_rc in A0_r do
        Add(A_r, Position(set, A0_rc));
      od;
      Add(A, A_r);
    od;
    InfoPermSym1(Length(set), "]\n");

    # iteratively refine partitions Br, Bc until both stabilize
    Br := [ Set([1..nr]) ];
    Bc := [ Set([1..nc]) ];
    iteration := 0;
    repeat

      # remember Length(Br), Length(Bc) to decide end of iteration
      Length_Br := Length(Br);
      Length_Bc := Length(Bc);

      # refine Br, Bc with frequencies of values in submatrices
      InfoPermSym1("#I   homogeneous partition is \c");
      for i in [1..Length(Br)] do
        Abr := List(Sublist(A, Br[i]), ShallowCopy);
        for A_r in Abr do 
          Sort(A_r); 
        od;
        refineBlock(Br, i, partitionIndex(Abr));
      od;
      for i in [1..Length(Bc)] do
        Abc := TransposedMat(List(A, A_r -> Sublist(A_r, Bc[i])));
        for A_c in Abc do 
          Sort(A_c);
        od;
        refineBlock(Bc, i, partitionIndex(Abc));
      od;
      if InfoPermSym1 <> Ignore then
        L := List(Br, Length); Sort(L);
        if Length(L) = 1 then
          InfoPermSym1(L[1]);
        else
          InfoPermSym1("(", L[1]);
          for i in [2..Length(L)] do InfoPermSym1("+", L[i]); od;
          InfoPermSym1(")");
        fi;
        InfoPermSym1("x");
        L := List(Bc, Length); Sort(L);
        if Length(L) = 1 then
          InfoPermSym1(L[1]);
        else
          InfoPermSym1("(", L[1]);
          for i in [2..Length(L)] do InfoPermSym1("+", L[i]); od;
          InfoPermSym1(")");
        fi;
        InfoPermSym1("\n");
      fi;

      # rename values in A to make blocks distinct
      if not ( Length(Br) = Length_Br and Length(Bc) = Length_Bc ) then

        InfoPermSym1("#I   remapping values into [1..\c");
        v := 0;
        for br in Br do
          for bc in Bc do
            set := [];
            for r in br do
              UniteSet(set, Sublist(A[r], bc));
            od;
            for r in br do
              for c in bc do
                A[r][c] := v + Position(set, A[r][c]);
              od;
            od;
            v := v + Length(set);
          od;
        od;
        InfoPermSym1(v, "]\n");

        iteration := iteration + 1;
      fi;

    until Length(Br) = Length_Br and Length(Bc) = Length_Bc;
    return [ A, Br, Bc ];
  end;

  # remap values in A; compute totals-partitions
  A  := remappedMat_Partitions(A, nr, nc);
  Br := A[2];
  Bc := A[3];
  A  := A[1];

  # compute equality partitions
  Er := PartitionIndex(A);
  Ec := PartitionIndex(TransposedMat(A));
  
  # catch a trivial case directly
  if Length(Br) = nr and Length(Bc) = nc then
    G               := Group( () );
    G.dimensionsMat := DimensionsMat(A);
    return G;
  fi;
  

  # Construction of group to search and of known subgroup
  # -----------------------------------------------------

  # First the column indices in [1..nc] are remapped into [nr+1..nr+nc]
  # to make them distinct from row indices. Then the Young-group K of the
  # partitions Er, Ec is set up as the known subgroup -- it contains all
  # permutations of equal rows and equal columns. A base with a strong
  # generating system is supplied. Finally the group G to search through is
  # constructed as the Young-group of the partitions Br, Bc. The base and
  # strong generating system for G is set up such that K is a stabilizer
  # of G and the Br/Bc-blocks containing the initial basepoints are short.

  # youngGroup( <orbits>, <basetail> )
  #   constructs the direct product of symmetric groups on the orbits
  #   and supplies a base and a strong generating system. The basetail
  #   is a list of basepoints to come last in the base. The orbits are
  #   sorted such that the orbits with the shortest non-basetail part
  #   come first in the base.

  youngGroup := function ( orbits0, basetail )
    local 
      group,  # the resulting group
      orbits, # list of non-trivial orbits in sorted order
      base,   # list of basepoints with basetail last
      gens,   # strong generating set
      b, b1,  # an orbit, a copy (order arranged)
      bh, bt, # head-part, tail-part of b1
      i;      # counter

    # sort a copy of orbits0 wrt. non-basetail length
    orbits := 
      List(
        Filtered(orbits0, b -> Length(b) > 1), 
        b -> 
          [ Length(b) - Length(IntersectionSet(b, basetail)),
            ShallowCopy(b)
          ]
      );
    Sort(orbits);
    orbits := List(orbits, nb -> nb[2]);

    # collect base and strong generating set (SGS)
    base := [];
    gens := [];
    for b in orbits do

      # split b into head and tail (preserve order of basetail)
      bh := Difference(b, IntersectionSet(b, basetail));
      bt := Filtered(basetail, p -> p in b);
      b1 := Concatenation(bh, bt);

      # add transpositions (b1[i], Last(b1)) to SGS
      for i in [1..Length(b1)-1] do
        Add(gens, (b1[i], b1[Length(b1)]) );
      od;

      # add the head-part to the base
      if Length(bt) = 0 then
        Append(base, Sublist(bh, [1..Length(bh)-1]));
      else
        Append(base, bh);
      fi;

    od;
    
    # finally add the basetail
    Append(base, basetail);

    # construct the group structure
    group := Group(gens, ());
    MakeStabChainStrongGenerators(group, base, gens);

    # report result
    if InfoPermSym1 <> Ignore then
      if 
        Length(orbits) = 0 or 
        Length(orbits) = 1 and Length(orbits[1]) = 1 
      then
	InfoPermSym1("trivial");
      elif Length(orbits) = 1 then
	InfoPermSym1("S", Length(orbits[1]));
      else
	InfoPermSym1("S", Length(orbits[1]));
	for i in [2..Length(orbits)] do
	  InfoPermSym1("xS", Length(orbits[i]));
	od;
      fi;
      InfoPermSym1("\n");
    fi;

    return group;
  end;

  # remap c -> nr + c; make sure Br, Bc, Er, Ec contain sets and are sets
  Br := Set(List(Br, Set));
  Bc := Set(List(Bc, b -> Set(nr + b)));
  Er := Set(List(Er, Set));
  Ec := Set(List(Ec, b -> Set(nr + b)));
  
  # construct K
  InfoPermSym1("#I   known subgroup is \c");
  K := youngGroup(Concatenation(Er, Ec), []);

  # catch a trivial case directly
  if Br = Er and Bc = Ec then
    K.dimensionsMat := DimensionsMat(A);
    return K;
  fi;

  # construct G
  InfoPermSym1("#I   group to search is \c");
  G := youngGroup(Concatenation(Br, Bc), K.operations.Base(K));


  # Backtrack-search for the symmetry group of A
  # --------------------------------------------

  # The backtrack-search basically runs through all elements in
  # all cosets with respect the the stabilizer chain given for G.
  # Namely the group K is expanded to contain all elements in the
  # symmetry group contained in the current stabilizer S of G.
  # Only the first permutation in a K-coset is considered which
  # is tested using the First-In-Orbit criterion. The restrictions
  # on the basepoint images imposed by the matrix A are traced by
  # keeping a set w[p] of remaining image points for every point p.
  # The sets w[p] are reduced when a basepoint image is chosen and
  # the contraints are propagated over w. The sets w[p] are stored
  # using GAP-Blists. Concerning the book of G. Butler on permutation
  # groups we consider the specific methods to improve the search:
  #   * First in orbit: done by considering K-orbits
  #   * Restriction of image points: done using w[p]-mechanism
  #   * Choosing apropriate base: done with short orbits first
  #   * Using a known subgroup: done with Young-group of equal rows, columns
  #   * Searching images of an initial base segment: done silently
  #   * More on cosets (left cosets): not done due to large overhead
  #   * Preprocessing: done; datastructures to propagate w-constraints

  # chooseBasepointImage(w, i, j)
  # init_chooseBasepointImage()
  #   the first function constructs a new vector for the restriction
  #   of the basepoint images with w[i] := [ j ] and all constraints being
  #   propagated. If this is inconsistent then false is returned. If
  #   the choice is uniquely determined then it is turned into a
  #   permutation to represent the choice. The second functions sets
  #   up the global variables orbitG and blocksA as a preprocessing and
  #   returns the initial vector w to be used in the search functions.

  init_chooseBasepointImage := function ()
    local
      tA,         # TransposedMat(A)
      B,          # list of orbits of G
      P,          # P[i] is list of sets of indices for constant A-value
      set,        # the set of values occuring in A[i]
      br, bc,     # a Br-block, Bc-block
      r, c, b, i; # counter for row, column, block, index

    # For i in [1..nr+nc] set orbitG[i] to be the G-orbit of i.
    # orbitG contains the ground sets to be used to resolve the
    # information stored in a Blist about a subset of G-orbits.
    B := 
      List(
        G.operations.Orbits(G, [1..nr+nc], OnPoints), 
        Set
      );
    orbitG := [];
    for b in B do
      for i in b do
        orbitG[i] := b;
      od;
    od;
 
    # Construct the list P such that P[i] is a list of blocks
    # b of indices and for all j1, j2 in b: A[i j1] = A[i j2].
    # The indices i, j1, j2 run in [1..nr+nc] (rows and columns)
    # and the blocks in P[i1], P[i2] correspond to each other
    # if i1, i2 are in the same Br- or Bc-block.
    P := [];
    for br in Br do
      set := Set(A[br[1]]);
      for r in br do
        P[r] := 
          List(
            set,
            x -> nr + Filtered([1..nc], c -> A[r][c] = x)
          );
      od;
    od;
    tA := TransposedMat(A);
    for bc in Bc do
      set := Set(tA[bc[1] - nr]);
      for c in bc do
        P[c] := 
          List(
            set,
            x -> Filtered([1..nr], r -> A[r][c - nr] = x)
          );
      od;
    od;

    # encode P in blocksA, blocksA1 as Blists:
    #   blocksA1[i] is a list [j_1, .., j_n] of indices such that orbitG[j_k]
    #               is the ground set of the set P[i][k]
    #   blocksA[i]  is a list of blists for P[i][k]
    blocksA := [];
    for i in [1..nr+nc] do
      blocksA[i] := 
        List( 
          P[i], 
          b -> BlistList(orbitG[b[1]], b)
        );
    od;
    blocksA1 := [];
    for br in Br do
      blocksA1[br[1]] := List(P[br[1]], b -> b[1]);
      for r in br do
        blocksA1[r] := blocksA1[br[1]];
      od;
    od;
    for bc in Bc do
      blocksA1[bc[1]] := List(P[bc[1]], b -> b[1]);
      for c in bc do
        blocksA1[c] := blocksA1[bc[1]];
      od;
    od;

    # return the initial w
    return 
      List(
        [1..nr+nc],
        i -> BlistList(orbitG[i], orbitG[i])
      );
  end;

  chooseBasepointImage := function (w0, i, j)
    local 
      w,             # resulting vector of Blists
      agenda,        # agenda of indices i still to be used
      w_i,           # w[i]
      SizeBlist_w_i, # SizeBlist(w[i])
      SizeBlist_w_j, # SizeBlist(w[j])
      p,             # position of true in w[i]
      C,             # list of Blists of the 'column' constraints
      k;             # counter for [1..Length(C)]

    # work with a new copy of w0
    w := List(w0, ShallowCopy);

    # start the agenda with the i -> j update
    w_i := BlistList(orbitG[i], [ j ]);
    if w[i] = w_i then
      return w;
    fi;
    w[i]   := w_i;
    agenda := [ i ];

    # iteratively update w until the agenda empties
    repeat
      
      # extract an entry i in agenda of minimal Size(w[i])
      i             := agenda[1];
      SizeBlist_w_i := SizeBlist(w[i]);
      for j in agenda do
        SizeBlist_w_j := SizeBlist(w[j]);
        if SizeBlist_w_j < SizeBlist_w_i then
          i             := j;
          SizeBlist_w_i := SizeBlist_w_j;
        fi;
      od;
      RemoveSet(agenda, i);

      # Constraint "Mutual exclusion": 
      #   If i is mapped to j then nothing else is mapped to j.
      if SizeBlist_w_i = 1 then
        p := Position(w[i], true);
        for j in orbitG[i] do
          if j <> i and w[j][p] then
            w[j][p] := false;
            if SizeBlist(w[j]) = 0 then
              return false;
            fi;
            AddSet(agenda, j);
          fi;
        od;
      fi;

      # Constraint "Row/column coupling":
      #   Let r be a row index being mapped into R1. Then there are
      #   constraints for all columns c. Namely c is mapped into
      #     { c1 | A[r1][c1] = A[r][c] for some r1 in R1 }.
      #   An analogous condition applies to column indices being mapped.

      # compute a list C of Blists such that C[k] is the union of 
      # all blocksA[j][k] for j in w[i] and k in [1..Length(blocksA1[i])]
      w_i := ListBlist(orbitG[i], w[i]);
      C   := List(blocksA[ w_i[1] ], ShallowCopy);
      for j in w_i do
        for k in [1..Length(C)] do
          UniteBlist(C[k], blocksA[j][k]);
        od;
      od;

      # update w[j] for j in the 'columns'
      for k in [1..Length(C)] do
        for j in ListBlist(orbitG[blocksA1[i][k]], blocksA[i][k]) do
          SizeBlist_w_j := SizeBlist(w[j]);
          IntersectBlist(w[j], C[k]);
          if SizeBlist(w[j]) < SizeBlist_w_j then
            if SizeBlist(w[j]) = 0 then 
              return false;
            fi;
            AddSet(agenda, j);
          fi;
        od;
      od;

    until Length(agenda) = 0;

    # recognize the unique case
    if ForAll(w, w_i -> SizeBlist(w_i) = 1) then
      return 
        PermList(
          List(
            [1..nr+nc],
            i -> ListBlist(orbitG[i], w[i])[1]
          )
        );
    fi;

    return w;
  end;


  # elementInCoset(level, S, t, K, w)
  #   searches for an element x in Sym(A) meet S*t. K is the intersection
  #   Sym(A) meet S. If there is no such x then false is returned. w is
  #   a list of Blists to restrict the basepoint images. The level is an
  #   indication of the nesting for Info2 only. The second function

  elementInCoset := function ( level, S, t, K, w )
    local 
      x,      # the element found; result
      points, # set of images of S.orbit[1]
      p,      # a point from points
      s,      # transversal permutation to take S.orbit[1]^s = p
      wp;     # new w-vector after choosing p

    InfoPermSym2(level, " \c");
    if S.generators = [] then

      # check the element t explicitly
      if 
        PermutedMat(
          RestrictedPerm(t, [1..nr]),
          A, 
          PermList(OnTuples([nr+1..nr+nc], t)-nr)^-1
        ) = A
      then
        return t;
      else
        return false;
      fi;

    fi;

    # the set images of S.orbit[1] to consider
    points := 
      IntersectionSet(
        OnTuples(S.orbit, t),
        ListBlist(orbitG[ S.orbit[1] ], w[ S.orbit[1] ])
      );

    # run through points
    while points <> [] do

      # choose one of the remaining points
      p := points[1];

      # find s in S with S.orbit[1]^s = p
      s := t;
      while S.orbit[1]^s <> p do
	s := S.transversal[p / s] mod s;
      od;

      # go down the recursion
      wp := chooseBasepointImage(w, S.orbit[1], p);
      if IsPerm(wp) then
        return wp; 
      elif wp <> false then
        x :=
          elementInCoset(
            level+1,
            S.stabilizer,
            s,
            Subgroup(G, Filtered(K.generators, x -> p^x = p)),
            wp
          );
        if x <> false then
          return x;
        fi;
      fi;

      # forget the entire K-orbit of p ('first in orbit'-method)
      points :=
        Difference(
          points,
          G.operations.Orbit(K, p, OnPoints)
        );
    od;
    return false;
  end;


  # findSymmetry(level, S, K, w)
  #   expands K to become Sym(A) meet S where S is a stabilizer of G.
  #   S is a stabilizer record (not a group record). w restricts the
  #   choice of basepoint images. The level is an integer for simplify
  #   debugging. The function returns nothing.

  findSymmetry := function ( level, S, K, w )
    local
      x,      # the element found; result
      points, # set of images of S.orbit[1]
      p,      # a point from points
      t,      # transversal permutation to take S.orbit[1]^t = p
      wp;     # new w-vector after choosing p

    if S.generators = [] or w = false or IsPerm(w) then
      return;
    fi;

    # the set of possible images of S.orbit[1]
    points := 
      IntersectionSet( 
        S.orbit, 
        ListBlist(orbitG[ S.orbit[1] ], w[ S.orbit[1] ])
      );

    # solve the problem for the stabilizer of S.orbit[1]
    wp := chooseBasepointImage(w, S.orbit[1], S.orbit[1]);
    findSymmetry(level+1, S.stabilizer, K.stabilizer, wp);
    for x in K.stabilizer.generators do
      if not x in K.generators then
        PermGroupOps.AddGensExtOrb( K, [ x ] );
      fi;
    od;
    points := 
      Difference(
        points, 
        G.operations.Orbit(K, S.orbit[1], OnPoints)
      );

    # run through points
    while points <> [] do

      # choose a remaining point
      p := points[1];

      # find t in S with S.orbit[1]^t = p
      t := S.identity;
      while S.orbit[1]^t <> p do
        t := S.transversal[p / t] mod t;
      od;

      # search for an element x in S.stabilizer*t meet Sym(A)
      wp := chooseBasepointImage(w, S.orbit[1], p);
      if IsPerm(wp) then
        x := wp;
      elif wp <> false then
        x := 
          elementInCoset(
            level,
            S.stabilizer, 
            t, 
            Subgroup(G, Filtered(K.generators, x -> p^x = p)),
            wp
          );
      else
        x := false;
      fi;

      # expand K
      if x <> false then
        PermGroupOps.AddGensExtOrb(K, [ x ]);
      fi;

      # forget the entire K-orbit of p ('first in orbit'-method)
      points := 
        Difference(
          points, 
          G.operations.Orbit(K, p, OnPoints)
        );
    od;

    if InfoPermSym2 <> Ignore then
      InfoPermSym2("\n");
    fi;
    InfoPermSym1("#I   completed stabilizer level ", level, "\n");
  end;

  # extend K to be the symmetry group in G (make stabchains identical first)
  InfoPermSym1("#I   starting backtrack search\n");
  K.parent := G;
  ExtendStabChain(K, G.operations.Base(G));
  w := init_chooseBasepointImage();

  findSymmetry(0, G, K, w);

  ReduceStabChain(K);
  Unbind(K.parent);
  InfoPermSym1("#I   backtrack search finished\n");
  K.dimensionsMat := DimensionsMat(A);
  return PermPermSym_checkAndPolish(K, A);
end;


# Using the implementation of J. Leon
# ===================================

# Perm-Perm-Symmetry(Mat) using J. Leon's package from GUAVA
#
# reference:
#   [1] J. Leon: Partition Backtrack Programs -- User's Manual. 
#       5/20/1992
#
# history of this part of the file:
#   SE, MP, 14.8.96, GAP v3.4
#   SE, 10.2.97 umgebaut auf "L M = M R" Konvention
#   MP, PermPermSymL/R eingebaut
#   SE, MP, 12.12.97 PermPermSymLeon in permperm.g integriert
#   SE, 11.3.98 AWK-Skript ersetzt durch stand-alone C-Programm

# PermPermSymLeon( <mat> )
#   computes the Perm-Perm-Symmetry of the given matrix.
#   The function computes the same as PermPermSym() but 
#   calls an external C-program 'desauto' written by J. Leon.
#   In fact, this program is most efficient but it has some
#   restrictions/problems at present:
#     * The number of distinct entries of the matrix must not
#       exceed 256.
#     * Passing the identity matrix of degree >= 64 causes 
#       desauto to abort with an error or to compute a wrong result.
#   The interface GAP/desauto uses two external C-programs to
#   convert the syntax to and from the form used in desauto.
#   If the function recognizes failure for any reason it is
#   free to return 'false' which indicates that PermPermSym 
#   should do the work with another implementation.
#

# Implementation notes:
#   * The flow of data through the files is as follows:
#
#       GAP ->( LeontmpPrintMat  )-> 'leontmp.raw'
#           ->( leonin           )-> 'leontmp'
#           ->( desauto -matrix  )-> 'leontmp.grp'
#           ->( leonout          )-> 'leontmp.in'
#           ->( LeontmpReadGroup )-> GAP
#
#     For the individual stages:
#       * 'leontmp.raw' is nearly CAYLEY-syntax suitable for 'desauto'
#         except for two stray characters '[' and ']'. These are
#         produced because we simply Print the list of matrix entries
#         to the file, which is orders of magnitude faster than 
#         Printing the entries one-by-one (even if the file is not
#         opened and closed for each entry).
#       * The trivial C-program 'leonin.c' removes the stray '[' 
#         and ']' by filtering stdin into stdout. The program is
#         used to produce the file 'leontmp' from 'leontmp.raw'.
#       * Note that the filenames for data in CAYLEY syntax is
#         restricted by the convention that name of the library must 
#         be the same as the name of the file containing it.
#       * J. Leon's *non-trivial* C-program 'desauto' is run to 
#         compute the Perm-Perm symmetry group of the matrix. The
#         base and strong generators are stored in CAYLEY-syntax
#         in the file 'leontmp.grp'.
#       * The option "-n:leontmp" is used in 'desauto' to rename
#         the output object in the CAYLEY library to 'leontmp'.
#         In fact, GAP 3.4.3 seems to crash on reading if this
#         is not done.
#       * If the global variable InfoPermSym1 is set to Print then
#         'desauto' is allowed to print runtime information to the
#         controling console. Otherwise we keep it silent with the
#         "-q" option (as described in [1]).
#       * Another trivial C-program 'leonout.c' does a few syntactic
#         changes on 'leontmp.grp' which are just enough to make the
#         result readable by the GAP reader. The program filters
#         stdin into stdout. It relies on several assumptions on
#         the form of the output! The program produces 'leontmp.in'.
#       * Finally, 'leontmp.in' is read into GAP. Normally, the file
#         executes a couple of assignments to the global variable
#         'leontmp'. Then LeontmpReadGroup analyses the data stored
#         in leontmp and converts it into a GAP group.
#       * A particular method to change the syntax is used on the
#         "seq"-construct of CAYLEY: We rename "seq" into "leontmp.seq"
#         and store the function "LeontmpSeq" into the global GAP-
#         variable "leontmp.seq". The effect is that the text
#         "seq(2,3,4)" produces the GAP list [2,3,4].

# global variable filled by reading output of 'desauto'
leontmp := rec( );
LEONTMP := leontmp; # an alias for case-insensitive file systems

# LeontmpPrintMat( <mat> )
#   prints the matrix <mat> in nearly CAYLEY-syntax into
#   the external file eventually used as input to 'desauto'.
#

LeontmpPrintMat := function ( M )
  local
    k, n, q,       # nr. of rows, columns, entries
    m,             # the entries as a list
    setM, Mi, Mij; # set of entries of M, M[i], M[i][j]

  # get size of M
  k := Length(M);
  n := Length(M[1]);

  # remap the entries of M into [0..q-1] and flatten the matrix
  setM := [ ];
  for Mi in M do
    UniteSet(setM, Mi);
  od;
  q := Length(setM);
  if q > 256 then
    Print(
      "#W Warning: >256 different entries in <M>,",
      " cannot use 'desauto' (by J. Leon)\n"
    );
    return false;
  fi;
  m := [ ];
  for Mi in M do
    for Mij in Mi do
      Add(m, Position(setM, Mij)-1);
    od;
  od;

  # write (k, n, q, m) to 'leontmp.raw' in (almost) CAYLEY-syntax
  InfoPermSym1("#I   Printing matrix to 'leontmp.raw'.\n");
  PrintTo("leontmp.raw",
    "LIBRARY leontmp;\n",
    "\" a (", k, "x", n, ")-matrix of ", q, " entries \"\n",
    "leontmp = seq(", q, ", ", k, ", ", n, ", seq(\n",
    m, "\n", # Produces superfluous '[' .. ']' (but fast)!
    "));\n",
    "FINISH;\n"
  );
  return true;
end;

# LeontmpRunDesauto( )
#   runs all the external programs.
#

LeontmpRunDesauto := function ( deg ) 
  local path, opts, cmd;

  # locate the ARep home directory;
  path := LOADED_PACKAGES.arep;

  # remove possibly left over leontmp.grp from previous run

  cmd :=
    ConcatenationString(
      "rm ",
      "leontmp.grp"
    );

  InfoPermSym1("#I   Exec(\"", cmd, "\");\n");
  Exec(cmd);

  # replace '[' -> ' ', ']' -> ' ' in leontmp.raw writing leontmp
  # (we use a little C-program, see below)

  cmd :=
    ConcatenationString(
      path, "bin/", "leonin ",
      "<leontmp.raw >leontmp"
    );

  InfoPermSym1("#I   Exec(\"", cmd, "\");\n");
  Exec(cmd);

  # run desauto (the program by J. Leon) on leontmp
  # to produce leontmp.grp (should contain the group)
  #
  # options:
  #   -mb:<int>   - increase the maximum nr. of base points; default 62
  #
  
  opts := "";
  if deg > 62 then
    opts := Concatenation(opts, " -mb:", String(deg + 10));
  fi;
  if InfoPermSym1 <> Print then
    opts := Concatenation(opts, " -q");
  fi;

  cmd :=
    ConcatenationString(
      path, "bin/desauto -matrix ",
      opts, " -n:leontmp leontmp leontmp.grp"
    );

  InfoPermSym1("#I   Exec(\"", cmd, "\");\n\n");
  Exec(cmd);
  InfoPermSym1("\n");

  # convert leontmp.grp to GAP-syntax writing leontmp.in

  cmd :=
    ConcatenationString(
      path, "bin/", "leonout ",
      "<leontmp.grp >leontmp.in"
    );

  InfoPermSym1("#I   Exec(\"", cmd, "\");\n");
  Exec(cmd);

  return true;
end;


# LeontmpSeq( <obj1>, .., <objN> )
#   returns the list [<obj1>, .., <objN>]. This function
#   is used to convert the CAYLEY syntax seq(..) into GAP
#   lists. Note that this function must be global because
#   the statements in the file being READ are executed in
#   the global binding environment.
#

LeontmpSeq := function ( arg ) 
  return arg;
end;


# LeontmpReadGroup( <nr>, <nc> )
#   reads the file 'leontmp.in' and decodes the information
#   in it to define the group for the (<nr> x <nc>)-matrix.
#

LeontmpReadGroup := function ( nr, nc )
  local G, sgs, g, retcode;

  # predefine global variable 'leontmp'
  leontmp     := rec( );
  leontmp.seq := LeontmpSeq; # make list
  LEONTMP     := leontmp;    # for case-insensitive file systems

  # read the information from 'leontmp.in'
  InfoPermSym1("#I   Reading permgroup from 'leontmp.in'.\n");
  if not READ("leontmp.in") then
    Print(
      "#W Warning: File 'leontmp.in' cannot be read.\n"
    );
    return false;
  fi;
  
  # check 'forder', 'base' and 'strong_generators'
  if not (
    IsBound(leontmp.forder) and
    IsInt(leontmp.forder) and 
    leontmp.forder >= 1 and

    IsBound(leontmp.base) and
    IsList(leontmp.base) and
    ForAll(leontmp.base, x -> IsInt(x) and x >= 1) and

    IsBound(leontmp.strong_generators) and
    IsList(leontmp.strong_generators) and
    ForAll(leontmp.strong_generators, IsPerm)
  ) then
    Print(
      "#W Warning: File 'leontmp.in' corrupted.\n"
    );
    return false;
  fi;
   
  # transform generators to L M = M R convention
  sgs := [ ];
  for g in leontmp.strong_generators do
    Add(
      sgs,
      RestrictedPerm(g, [1..nr])^-2 * g
    );
  od;

  # build the group
  InfoPermSym1("#I   Building the permgroup with base and SGS.\n");
  G      := Group(leontmp.strong_generators, ());
  G.size := leontmp.forder;
  MakeStabChainStrongGenerators(
    G, 
    leontmp.base, 
    leontmp.strong_generators
  );

  # store some additional information (used in AREP)
  G.theGenerators := leontmp.strong_generators;
  G.dimensionsMat := [nr, nc];
  return G;
end;

# PermPermSymLeon( <mat> )
#   the counter part of PermPermSym() using Leon's implementation.
#   This function is the driver function to prepare the external
#   files, run the external programs and read the answers.
#

PermPermSymLeon := function ( M )
  local G, state;

  # M is ok because we are called from PermPermSym()
  InfoPermSym1(
    "#I PermPermSymLeon( ", 
    "<", Length(M), "x", Length(M[1]), "-matrix> ",
    ") called\n"
  );

  # output the matrix
  state := LeontmpPrintMat(M);
  if state = false then
    return false;
  fi;

  # run Leon's program
  state := LeontmpRunDesauto(Length(M) + Length(M[1]));
  if state = false then
    return false;
  fi;

  # read, decode and check the answer
  return LeontmpReadGroup(Length(M), Length(M[1]));
end;

