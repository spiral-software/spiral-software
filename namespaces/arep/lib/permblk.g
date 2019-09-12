# -*- Mode: shell-script -*-
# Determination of Perm-Irred-Symmetry, 
# SE, 10.02.97 - , GAPv3.4

# SE, 10.2.97: erste Version GAP v3.4
# SE, MP, 21.2.97: PermBlockSymL/R
# SE, 15.4.97: b"osen Fehler in PermBlockSymByPermutations

#F HasseEdgesList( <list>, <less> )
#F   the set of pairs [i, j] of all Hasse edges of the elements
#F   in <list>, ordered with respect to func <less>.
#F   

HasseEdgesList := function ( list, less )
  local 
    edges, # the result
    lower, # i in lower[j] <=> list[i] less list[j] 
    max,   # set of maximal lower elements for j
    i, j;  # counter into list

  if not ( IsList(list) and IsFunc(less) ) then
    Error("<list> must be list, <less> an order func");
  fi;
  if Length(list) = 0 then
    return [ ];
  fi;

  lower := List([1..Length(list)], i -> [ ]);
  for i in [1..Length(list)] do
    for j in [1..Length(list)] do
      if 
        i <> j and list[i] <> list[j] and
        less(list[i], list[j])
      then
        Add(lower[j], i);
      fi;
    od;
  od;

  edges := [ ];
  for j in [1..Length(list)] do
    max := lower[j];
    for i in lower[j] do
      max := Difference(max, lower[i]);
    od;
    for i in max do
      Add(edges, [i, j]);
    od;
  od;
  Sort(edges);
  return edges;
end;


#F Partitions of sets
#F ==================
#F
#F A partition of a set S is a set of disjoint non-empty 
#F subsets of S which cover the entire set S. It is 
#F represented as a sorted list of the subsets.
#F

#F BlockOfPartition( p, x )    block in p containing x, or false
#F IsRefinedPartition( p, q )  p >= q, that is q is a refinement of p
#F MeetPartition( p, q )       greatest lower bound of p, q
#F JoinPartition( p, q )       least upper bound of p, q
#F   lattice operations for partitions of the same set.
#F   If p and q are not partitions or not of the same set
#F   then the result is unspecified.
#F

BlockOfPartition := function ( p, x )
  local b, x;

  for b in p do
    if x in b then
      return b;
    fi;
  od;
  return false;
end;

IsRefinedPartition := function ( p, q )
  return 
    ForAll(
      q, 
      function ( bq )
        local bp;

        bp := BlockOfPartition(p, bq[1]);
        if bp = false then
          Error("<p>, <q> must partition the same set");
        fi;
        return IsSubset(bp, bq);
      end
    );
end;

MeetPartition := function ( p, q )
  local r, bp, bq, br;

  r := [ ];
  for bp in p do
    for bq in q do
      br := Intersection(bp, bq);
      if Length(br) > 0 then
        Add(r, br);
      fi;
    od;
  od;
  Sort(r);
  return r;
end;

JoinPartition := function ( p, q )
  local r, br, r1, br1, bq, xq;

  r := List(p, ShallowCopy);
  for bq in q do
    br1 := [ ];
    for xq in bq do
      UniteSet(br1, BlockOfPartition(r, xq));
    od;
    
    r1 := [ br1 ];
    for br in r do
      if not br[1] in br1 then
        Add(r1, br);
      fi;
    od;
    r := r1;
  od;
  Sort(r);
  return r;
end;

#F PartitionIndex( <list> )
#F   partition [1..Length(list)] with respect to equality
#F   of the corresponding list elements.
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

#F kbs( <A> )
#F   given a square (n x n)-matrix A, compute the conjugated
#F   block structure as the partition
#F     kbs(A) = [1..n] / R*
#F   where R* is the reflexive symmetric transitive closure of
#F   the relation R defined by (i, j) in R <=> A[i][j] <> 0.
#F     The argument may also be a list of (n x n)-matrices, 
#F   a group of matrices, or a permutation group with the
#F   field A.degree bound to the degree considered. In each
#F   of these cases the join of the kbs(x) for all x in A 
#F   is computed and returned.
#F

kbs := function ( A )
  local n, bs, i, j, k;

  if IsMatGroup(A) then

    return kbs( Union(A.generators, [ A.identity ]) );

  elif IsPermGroup(A) then

    if IsBound(A.degree) then
      n := A.degree;
    elif IsTrivial(A) then
      n := 1;
    else
      n := PermGroupOps.LargestMovedPoint(A);
    fi;
    return Set( Orbits(A, [1..n]) );

  elif IsMat(A) then

    if not Length(A) = Length(A[1]) then
      Error("<A> must be square");
    fi;
    n  := Length(A);
    bs := List([1..n], i -> [ i ]);
    for i in [1..n] do
      for j in [1..n] do
    if A[i][j] <> 0*A[i][j] then

      # join i and j
      UniteSet(bs[i], bs[j]);
      for k in bs[i] do
        bs[k] := bs[i];
      od;
    fi;
      od;
    od;
    return Set(bs);

  elif IsList(A) then

    if not Length(A) > 0 then
      Error("<A> must contain an element");
    fi;
    bs := kbs(A[1]);
    for k in [2..Length(A)] do
      bs := JoinPartition(bs, kbs(A[k]));
    od;
    return bs;

  else
    Error("wrong arguments");
  fi;
end;

#F kbsM( <M^-1>, <permgrp>, <M> )
#F   the join of all kbs(M^-1 x M) for x in <permgroup>.
#F

kbsM := function ( InvM, G, M )
  local bs, x;

  bs := List([1..Length(M)], k -> [ k ]);
  for x in G.generators do
    bs := JoinPartition(bs, kbs(InvM * Permuted(M, x^-1)));
  od;
  return bs;
end;

#F Perm-Block symmetry
#F ===================
#F
#F Given an invertible (n x n)-matrix M define
#F
#F   PermBlock(M) := { G_M(kbs(M^-1 x M)) | x in SymmetricGroup(n) }.
#F
#F where G_M(p) := { x in SymmetricGroup(n) | kbs(M^-1 x M) <= p }
#F for all p. (The refinement relation is called IsRefinedPartition.)
#F
#F PermBlockSym<Method>( <mat> )
#F   applies <Method> to find the Perm-Block symmetry of
#F   the invertible square matrix <mat>. The functions return
#F   a list of subgroups G of the symmetric group. The group
#F   records are augmented by the following fields
#F     G.M    := M
#F     G.invM := M^-1
#F     G.kbsM := Join(kbs(M^-1 x M) : x in G)
#F   In addition, there are optional fields which might be present
#F     G.charactersM    :=
#F       the list of characters of (M^-1)_{b,*} * G * M_{*,b} for 
#F       the block b in G.kbsM
#F     G.isIrreducibleM := 
#F       whether G^M consists of irreducibles of G; this is the
#F       same as ForAll(G.charactersM, chi -> chi in Irr(G))
#F     G.intertwiningNr :=
#F       the intertwining number of G with itself
#F

if not IsBound(InfoPermBlock1) then
  InfoPermBlock1 := Ignore;
fi;
if not IsBound(InfoPermBlock2) then
  InfoPermBlock2 := Ignore;
fi;

#F LessPermBlockSym( <permgrp1>, <permgrp2> )
#F   decides <permgrp1> < <permgrp2> for the groups
#F   which result from PermBlockSym. The function compares
#F   the kbsM-field and then the groups.
#F

LessPermBlockSym := function ( G, H )
  if not ( IsPermGroup(G) and IsPermGroup(H) ) then
    return G < H;
  fi;
  if G = H then
    return false;
  fi;
  if IsBound(G.kbsM) and IsBound(H.kbsM) then
    if IsRefinedPartition(H.kbsM, G.kbsM) then return true;  fi;
    if IsRefinedPartition(G.kbsM, H.kbsM) then return false; fi;
  fi;
  if ForAll(G.generators, g -> g in H) then return true;  fi;
  if ForAll(H.generators, h -> h in G) then return false; fi;

  # not well defined; good luck
  if Size(G) < Size(H) then return true;  fi;
  if Size(H) < Size(G) then return false; fi;
  return G < H;
end;

#F CompletedPermBlockSym( <list-of-permgrp> )
#F   complete the information in the Perm-Block symmetry list
#F   passed. The list must be non-empty and at least one group
#F   record must contain the field .M for the matrix. Then
#F   everything else can be computed. 
#F     The function does not initiate computation of character 
#F   tables but it uses a character tables if it is present.
#F   Hence, to compute the irreducible blocks of the groups
#F   issue List(<list>, CharTable) and call CompletedPermBlockSym.
#F   Note that the <list> is modified and noting is returned.
#F

CompletedPermBlockSym := function ( L )
  local 
    M, InvM, n, # matrix, M^-1, Length(M)
    parent,     # the common parent group
    G, nr,      # a group in L, counter for G in L
    chi,        # the character of G (on [1..n]!)
    mults,      # mult[k] is the multiplicity of G.irr[k] in chi
    i,          # counter
    k;          # counter into G.irr

  # check the argument
  if not (
    IsList(L) and Length(L) >= 1 and 
    ForAll(L, IsPermGroup) and
    ForAny(L, G -> IsBound(G.M) and IsMat(G.M))
  ) then
    Error("<L> must be list of permgrp, <L>[1].M a matrix");
  fi;
  InfoPermBlock2("#I fetching M; \c");
  M  := false;
  nr := 1;
  while M = false and not IsBound(L[nr].M) do
    nr := nr + 1;
  od;
  M := L[nr].M;
  n := Length(M);
  if IsBound(L[nr].invM) then
    InvM := L[nr].invM;
  else
    InfoPermBlock2("computing M^-1; \c");
    InvM := M^-1;
  fi;

  # make sure all G in L have a common parent 
  if not ForAll(L, G -> Parent(G) = Parent(L[1])) then

    # get or build a parent
    InfoPermBlock2("building common parent group;\n#I ");
    parent := Parent(L[1]);
    for G in L do
      if Size(Parent(G)) > Size(parent) then
        parent := Parent(G);
      fi;
    od;
    if not ForAll(L, G -> ForAll(G.generators, x -> x in parent)) then
      parent      := SymmetricGroup(n);
      parent.name := ConcatenationString("S", String(n));
    fi;

    # make the group subgroups of parent
    for nr in [1..Length(L)] do
      if not Parent(L[nr]) = parent then
        G          := L[nr];
        L[nr]      := AsSubgroup(parent, L[nr]);
        L[nr].M    := M;
        L[nr].invM := InvM;
        if IsBound(G.kbsM) then
          L[nr].kbsM := G.kbsM;
        fi;
      fi;
    od;
  fi;
  
  # store M, invM, kbsM if not present
  InfoPermBlock2("storing M, invM, kbsM; \c");
  for G in L do
    if not IsBound(G.M) then
      G.M := M;
    fi;
    if not IsBound(G.invM) then
      G.invM := InvM;
    fi;
    if not IsBound(G.kbsM) then
      G.kbsM := kbsM(InvM, G, M);
    fi;
  od;

  # name and sort the groups
  InfoPermBlock2("naming/sorting; \c");
  for i in [1..Length(L)] do
    L[i].name := ConcatenationString("G", String(i));
  od;
  Sort(L, LessPermBlockSym);

  # complete information if charTable present
  InfoPermBlock2("using charTable; \c");
  for G in L do
    if IsBound(G.charTable) then

      # compute the characters of the block in kbsM
      if not IsBound(G.charactersM) then
	G.charactersM :=
	  List( 
	    G.kbsM, 
	    function ( b )
	      local Mb, InvMb, traceMb;

	      InvMb := Sublist(InvM, b);
	      Mb    := List(M, Mi -> Sublist(Mi, b));

	      # compute trace((M^-1)_{b,*} x M_{*,b})

	      traceMb := function ( x )
		local tr, k, i;

		tr := 0*M[1][1];
		for k in [1..Length(b)] do
		  for i in [1..n] do
		    tr := tr + InvMb[k][i] * Mb[i^x][k];
		  od;
		od;
		return tr;
	      end;

	      return
		Character( G,
		  List(
		    ConjugacyClasses(G), 
		    xG -> traceMb( Representative(xG) )
		  )
		); 
	    end
	  );
      fi;

      # test if M decomposes G into irreducibles
      if not IsBound(G.isIrreducibleM) then
        G.isIrreducibleM := 
          ForAll(G.charactersM, IsIrreducible);
      fi;

      # compute the intertwining number of G (with itself)
      if not IsBound(G.intertwiningNr) then
        
        # decompose the character of G
        mults := List([1..Length(Irr(G))], k -> 0);
        for chi in G.charactersM do
          if IsIrreducible(chi) then
            k := Position(G.irr, chi);
            mults[k] := mults[k] + 1;
          else
            for k in [1..Length(G.irr)] do
              mults[k] := mults[k] + ScalarProduct(chi, G.irr[k]);
            od;
          fi;
        od;
        G.intertwiningNr := 
          Sum( 
            List(
              [1..Length(G.irr)], 
              k -> mults[k]^2 * Degree(G.irr[k])^2
            )
          );
      fi;
    fi;
  od;

  InfoPermBlock2("\n");
end;

#F DisplayPermBlockSym( <list-of-permgrp> )
#F   display a Perm-Block symmetry list nicely. The argument
#F   is the same as for CompletedPermBlockSym. The function
#F   does not return anything.
#F

DisplayPermBlockSym := function ( L )
  local 
    S,               # converted information
    lenS,            # the lengths of the fields in S
    Sr, c,           # a row of S, counter for columns
    n, ceil_lg_n,    # size of the matrix, Ceiling(Log(n, 10))
    spaces,          # func to produce spaces
    stringPartition, # func to convert a partition
    hasseEdges, e,   # the order structure, an element of h.Es.
    nr;              # counting the groups in L

  # check argument and complete information
  if not (
    IsList(L) and Length(L) >= 1 and 
    ForAll(L, G -> IsPermGroup(G)) and
    ForAny(L, G -> IsBound(G.M) and IsMat(G.M))
  ) then
    Error("<L> must be a list of permgrp for a PermBlockSym");
  fi;
  CompletedPermBlockSym(L);
  n          := Length(L[1].M);
  ceil_lg_n  := LogInt(n, 10)+1;
  hasseEdges := 
    HasseEdgesList(
      L, 
      function ( G, H )
        return IsRefinedPartition(H.kbsM, G.kbsM);
      end
    );

  # formatting functions

  spaces := function ( n )
    local s, small;

    if n <= 0 then
      return "";
    fi;
    
    small := [" ", "  ", "   ", "    ", "     ", "      "];
    if n <= Length(small) then
      return small[n];
    fi;
    s := "";
    while n > 0 do
      s := ConcatenationString(s, " ");
      n := n - 1;
    od;
    return s;
  end;

  stringPartition := function ( p, irred )
    local s, ib, b, k, sk, braces;

    s := "";
    for ib in [1..Length(p)] do
       b := p[ib];

       # determine the braces to use
       braces := ["(", ")"];
       if irred <> false then
         if irred[ib] then
           braces := ["[", "]"];
         else
           braces := ["<", ">"];
         fi;
       fi;
  
       # print the block
       s := ConcatenationString(s, braces[1]);
       for k in b do
         sk := String(k);
         sk := 
           ConcatenationString(
             spaces(ceil_lg_n - LengthString(sk)),
             sk
           );
         s := ConcatenationString(s, sk);
         if k <> b[Length(b)] then
           s := ConcatenationString(s, "  ");
         fi;
       od;
       s := ConcatenationString(s, braces[2]);
    od;
   return s;
  end;

  # prepare information converted to strings
  S := [ ];
  for nr in [1..Length(L)] do

    # add nr.
    Add(S, [ String(nr) ]);

    # add kbsM(G)
    if IsBound(L[nr].charactersM) then
      Add( S[nr],
        stringPartition(
          L[nr].kbsM, 
          List(L[nr].charactersM, IsIrreducible)
        )
      );
    else
      Add( S[nr],
        stringPartition(L[nr].kbsM, false)
      );
    fi;    

    # add area covered by the blocks
    Add( S[nr],
      String( Sum(List(L[nr].kbsM, b -> Length(b)^2)) )
    );

    # add the name of the group
    if IsBound(L[nr].name) then
      Add(S[nr], L[nr].name);
    else
      Add( S[nr],
        ConcatenationString("Group", String(nr))
      );
    fi;

    # add intertwiningNr
    if IsBound(L[nr].intertwiningNr) then
      Add( S[nr],
        String( L[nr].intertwiningNr )
      );
    else
      Add(S[nr], "?");
    fi;

    # add |G|
    if Factorial(n)/Size(L[nr]) < Size(L[nr]) then
      Add( S[nr],
        ConcatenationString(
          String(n), "!/", String(Factorial(n)/Size(L[nr]))
        )
      );
    else
      Add( S[nr],
        String(Size(L[nr]))
      );
    fi;

    # add the maximal proper subgroups/refinements
    Add(S[nr], "");
    for e in hasseEdges do
      if e[2] = nr then
        if S[nr][Length(S[nr])] = "" then
          S[nr][Length(S[nr])] := String(e[1]);
        else
          S[nr][Length(S[nr])] := 
            ConcatenationString(
              S[nr][Length(S[nr])], " ",
              String(e[1])
            );
        fi;
      fi;
    od;
  od;

  # add headings
  S := 
    Concatenation(
      [ ["nr.", "p", "area(p) ", "G", "i(G)", "Size(G)", "maximals" ],
        ["",    "",  "",         "",  "",     "",        ""         ]
      ], 
      S
    );

  # align and print the table
  lenS := 
    List(
      [1..Length(S[1])], 
      c -> Maximum(List(S, Sr -> LengthString(Sr[c])))
    );

  Print("\n");
  for Sr in S do
    Print("  ");
    for c in [1..Length(Sr)] do
      Print(Sr[c], spaces(lenS[c] - LengthString(Sr[c])));
      if c < Length(Sr) then
        Print(" ");
      else
        Print("\n");
      fi;
    od;
  od;

  # print legend
  Print("\n ([] = irred., <> = not irred., () = unknown status)\n\n");
end;

#F PermBlockSymL( <sym>, <x> )
#F PermBlockSymR( <sym>, [ <block>, ] <x> )
#F   the projections of the element(s) <x> in the Perm-Block symmetry
#F   group <sym> (which must have .M, .invM, .kbsM bound) onto the
#F   left or right component. If <block> in <sym>.kbsM is given then
#F   the corresponding block of PermBlockSymR(<sym>, <x>) is extracted.
#F     The argument <x> may be either a permutation,  a list of
#F   permutations, or a group of permutations.
#F  

PermBlockSymL := function ( sym, x )
  if IsPerm(x) then
    return x;
  elif IsList(x) then
    return x;
  elif IsPermGroup(x) then
    if not ( IsPermGroup(sym) and IsBound(sym.M) ) then
      Error("<sym> must be permgroup with sym.M etc.");
    fi;
    x.M    := sym.M;
    x.invM := sym.invM;
    x.kbsM := sym.kbsM;
    return x;
  else
    Error("<x> must be perm, list-of-perm or permgroup");
  fi;
end;

PermBlockSymR := function ( arg )
  local
    sym, block, x, # arguments
    GR, id;        # the group, the identity in GR

  # decode argument list
  if Length(arg) = 2 then
    sym   := arg[1];
    block := false;
    x     := arg[2];
  elif Length(arg) = 3 then
    sym   := arg[1];
    block := arg[2];
    x     := arg[3];
  else
    Error("usage: PermBlockSymR( <sym>, [ <block>, ] <x> )");
  fi;

  # check arguments
  if not (
    IsPermGroup(sym) and 
    IsBound(sym.M) and IsBound(sym.invM) and IsBound(sym.kbsM)
  ) then
    Error("<sym> must be permgroup with .M, .invM, .kbsM bound");
  fi;
  if not (
    block = false or block in sym.kbsM
  ) then
    Error("<block> must be a block in <sym>.kbsM");
  fi;
  
  # dispatch on type of x
  if IsPerm(x) then

    if block = false then

      # compute M^-1 * x * M
      return sym.invM * Permuted(sym.M, x^-1);

    else

      # compute (M^-1)_{b,*} * x * M_{*,b}
      return 
        Sublist(sym.invM, block) *
        Permuted(
          List(sym.M, Mi -> Sublist(Mi, block)), 
          x^-1
        );

    fi;

  elif IsList(x) then

    return List(x, xk -> PermBlockSymR(sym, block, xk));

  elif IsPermGroup(x) then

    # construct a suitable identity matrix
    if block = false then
      id := 
        IdentityMat(
          Length(sym.M), 
          DefaultField([sym.M[1][1]])
        );
    else
      id := 
        IdentityMat(
          Length(block), 
          DefaultField([sym.M[1][1]])
        );
    fi;

    # construct the image of x under PermBlockSymR
    GR := 
      Group(
        List(
          x.generators, 
          xk -> PermBlockSymR(sym, block, xk)
        ),
        id
      );

    # copy the additional fields
    GR.M    := sym.M;
    GR.invM := sym.invM;
    GR.kbsM := sym.kbsM;
    return GR;

  fi;
end;


#F PermBlockSymByPermutations( <mat> )
#F   compute PermBlockSym( <mat> ) by enumerating the
#F   entire symmetric group acting on <mat>. This is
#F   the slowest but most simple approach.
#F

PermBlockSymByPermutations := function ( M )
  local
    L,         # the lattice, result
    n, InvM,   # Length(M), M^-1
    Sn,        # SymmetricGroup(n)
    x, xIndex, # element of Sn, running index
    kbsMx,     # kbs(M^-1 x M)
    kbsMxSeen, # flag if is kbsMx in L already
    nr,        # index into L
    tmp;

  # check argument
  if not ( IsMat(M) and Length(M) = Length(M[1]) ) then
    Error("<M> must be a square matrix");
  fi;
  n       := Length(M);
  InvM    := M^-1;
  Sn      := SymmetricGroup(n);
  Sn.name := ConcatenationString("S", String(n));

  # initialize L
  L         := [ Subgroup(Sn, [ ]) ];
  L[1].M    := M;
  L[1].invM := InvM;
  L[1].kbsM := List([1..n], k -> [ k ]);

  # run through SymmetricGroup(n) 
  xIndex := 0;
  InfoPermBlock1("#I running through ", Size(Sn), " elements:\n");
  for x in Elements(Sn) do
    xIndex := xIndex + 1;
    InfoPermBlock1(xIndex, " \c");

    # compute kbs(M^-1 x M)
    kbsMx := kbs(InvM * Permuted(M, x^-1)); 
    
    # insert x into the lattice
    kbsMxSeen := false;
    for nr in [1..Length(L)] do
      if L[nr].kbsM = kbsMx then
        kbsMxSeen := true;
      fi;

      if
        IsRefinedPartition(L[nr].kbsM, kbsMx) and
        not x in L[nr]
      then
        tmp        := [ L[nr].M, L[nr].invM, L[nr].kbsM ];
        L[nr]      := Closure(L[nr], x);
        L[nr].M    := tmp[1];
        L[nr].invM := tmp[2];
        L[nr].kbsM := tmp[3];
      fi;
    od;
    if not kbsMxSeen then
      tmp := [ x ];
      for nr in [1..Length(L)] do
        if IsRefinedPartition(kbsMx, L[nr].kbsM) then
          UniteSet(tmp, L[nr].generators);
        fi;
      od;
      Add(L, Subgroup(Sn, tmp));
      L[Length(L)].M    := M;
      L[Length(L)].invM := InvM;
      L[Length(L)].kbsM := kbsMx;
    fi;
  od;
  InfoPermBlock1("\n");

  CompletedPermBlockSym(L);
  return L;
end;

#F PermBlockSymBySubsets( <mat> [, <kset> ] )
#F   compute PermBlockSym( <mat> ) by enumerating the
#F   k-subsets b of [1..n] for all 2 k <= n and finding
#F   PermMatSym( <mat>_{*,b} ). This method does rely
#F   on PermMatSym but not on subtle lattice properties.
#F   If the parameter <kset> is supplied then only those
#F   k's appearing in <kset> are considered.
#F

PermBlockSymBySubsets := function ( arg )
  local
    M, kset, # the arguments
    L,       # the lattice, result
    n,       # Length(M), M^-1
    Sn,      # SymmetricGroup(n)
    k,       # the size of the subsets b to consider
    b,       # a k-subset of [1..n]
    G,       # PermMat( M_{*,b} )
    nr,      # index into L
    x;       # element of G

  # decode and check arg
  if Length(arg) = 1 and IsMat(arg[1]) then
    M    := arg[1];
    kset := [1..Length(M)];
  elif Length(arg) = 2 and IsMat(arg[1]) and IsList(arg[2]) then
    M    := arg[1];
    kset := arg[2];
  fi;
  if not Length(M) = Length(M[1]) then
    Error("<M> must be a square matrix");
  fi;

  n       := Length(M);
  Sn      := SymmetricGroup(n);
  Sn.name := ConcatenationString("S", String(n));
  Sn.M    := M;
  Sn.invM := M^-1;
  Sn.kbsM := kbsM(Sn.invM, Sn, Sn.M);  

  # run through the k-subsets
  L := [ Sn ];
  for k in [1..QuoInt(n, 2)] do
    if k in kset  or  (n - k) in kset then
      InfoPermBlock1(
        "#I considering ", Binomial(n, k), " many ", 
        k, "-subsets of [1..", n, "]\n"
      );
      for b in Combinations([1..n], k) do

        # compute G_M({b, {1..n}-b})
        G      := PermMatSym( List(M, Mi -> Sublist(Mi, b)) );
        G      := AsSubgroup(Sn, G);
        G.M    := M;
        G.invM := Sn.invM;
        G.kbsM := kbsM(Sn.invM, G, Sn.M);

        # insert G into L
        nr := PositionProperty(L, H -> H.kbsM = G.kbsM);
        if nr = false then          
          Add(L, G);
        else

          # join L[nr] and G
          for x in G.generators do
            if not x in L[nr] then
              L[nr] := Closure(L[nr], x);
            fi;
          od;
          L[nr].M    := G.M;
          L[nr].invM := G.invM;
          L[nr].kbsM := G.kbsM;
        fi;
      od;
    fi;
  od;

  CompletedPermBlockSym(L);
  return L;
end;
