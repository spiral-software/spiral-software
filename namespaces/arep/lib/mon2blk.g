# -*- Mode: shell-script -*-
# Mon2-Block-Symmetry
# ===================
#
# Sebastian Egner, 27. August 1999 in GAP v3.4.4
# This module needs 'mon2mat.g' and others from "arep".

# PermMatSym; # loads SelectBaseFromList()
# Read("mon2mat.g");

if not IsBound(InfoMon2Block1) then
  InfoMon2Block1 := Ignore;
fi;

if not IsBound(InfoMon2Block2) then
  InfoMon2Block2 := Ignore;
fi;

#F Displaying and Accessing Mon2-Block-Symmetry
#F ============================================
#F
#F In analogy to Perm-Block-Symmetry there is a function to
#F display the result of Mon2BlockSym in a suitable way. If
#F If you compute the CharTable for each group in the list 
#F the the program will also indicate whether blocks are
#F irreducible or can be split further.
#F

#F kbsMon2M( <M^-1>, <permgrp>, <M> )
#F   the join of all kbs(M^-1 x M) for x in <permgrp>, which
#F   is the encoded version of a monomial group with [-1, 1].
#F

kbsMon2M := function ( InvM, G, M )
  local bs, x;

  bs := List([1..Length(M)], k -> [ k ]);
  for x in G.generators do
    bs := JoinPartition(bs, kbs(InvM * MatMon(Mon2Decode(x, Length(M))) * M));
  od;
  return bs;
end;

#F LessMon2BlockSym( <permgrp1>, <permgrp2> )
#F   defines an order on results of Mon2BlockSym.
#F

LessMon2BlockSym := function ( G, H )
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


#F CompletedMon2BlockSym( <list-of-permgrp> )
#F   complete the information in the Mon2-Block symmetry list
#F   passed. The list must be non-empty and at least one group
#F   record must contain the field .M for the matrix. Then
#F   everything else can be computed. 
#F     The function does not initiate computation of character 
#F   tables but it uses a character tables if it is present.
#F   Hence, to compute the irreducible blocks of the groups
#F   issue List(<list>, CharTable) and call CompletedMon2BlockSym.
#F   Note that the <list> is modified and noting is returned.
#F

CompletedMon2BlockSym := function ( L )
  local 
    M, InvM, n, # matrix, M^-1, Length(M)
    parent,     # the common parent group
    G, nr,      # a group in L, counter for G in L
    chi,        # the character of G (on [1..n]!)
    mults,      # mult[k] is the multiplicity of G.irr[k] in chi
    k,          # counter into G.irr
    i;          # counter into [1..Length(L)]

  # check the argument
  if not (
    IsList(L) and Length(L) >= 1 and 
    ForAll(L, IsPermGroup) and
    ForAny(L, G -> IsBound(G.M) and IsMat(G.M))
  ) then
    Error("<L> must be list of permgrp, <L>[1].M a matrix");
  fi;
  InfoMon2Block2("#I fetching M; \c");
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
    InfoMon2Block2("computing M^-1; \c");
    InvM := M^-1;
  fi;

  # make sure all G in L have a common parent 
  if not ForAll(L, G -> Parent(G) = Parent(L[1])) then

    # get or build a parent
    InfoMon2Block2("building common parent group;\n#I ");
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
  InfoMon2Block2("storing M, invM, kbsM; \c");
  for G in L do
    if not IsBound(G.M) then
      G.M := M;
    fi;
    if not IsBound(G.invM) then
      G.invM := InvM;
    fi;
    if not IsBound(G.kbsM) then
      G.kbsM := kbsMon2M(InvM, G, M);
    fi;
  od;

  # name and sort the groups
  InfoMon2Block2("naming/sorting; \c");
  for i in [1..Length(L)] do
    if not IsBound(L[i].name) then
      L[i].name := ConcatenationString("G", String(i));
    fi;
  od;
  Sort(L, LessMon2BlockSym);

  # complete information if charTable present
  InfoMon2Block2("using charTable; \c");
  for G in L do
    if IsBound(G.charTable) then

      # compute the characters of the block in kbsM
      if not IsBound(G.charactersM) then
        G.charactersM :=
          List( 
            G.kbsM, 
            function ( b )
              local Mb, InvMb, traceMb, Gb; # ...

              InvMb := Sublist(InvM, b);
              Mb    := List(M, Mi -> Sublist(Mi, b));

              # compute trace((M^-1)_{b,*} x M_{*,b})

              traceMb := function ( x )
                local tr, k, i, xm;

                # decode x into perm * diag
                xm := Mon2Decode(x, Length(M));

                # compute trace
                tr := 0*M[1][1];
                for k in [1..Length(b)] do
                  for i in [1..n] do
                    tr := 
                      tr + 
                      InvMb[k][i] * 
                      xm.diag[i^xm.perm] * 
                      Mb[i^xm.perm][k];
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

  InfoMon2Block2("\n");
end;

#F DisplayMon2BlockSym( <list-of-permgrp> )
#F   display a Mon2-Block symmetry list nicely. The argument
#F   is the same as for CompletedMon2BlockSym. The function
#F   does not return anything.
#F

DisplayMon2BlockSym := function ( L )
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
    Error("<L> must be a list of permgrp for a Mon2BlockSym");
  fi;
  CompletedMon2BlockSym(L);
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
    Add( S[nr],
      String(Size(L[nr]))
    );

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

#F BlockMon2BlockSym( <permgrp>, <b> )
#F   given <permgrp> which has been constructed by Mon2BlockSym() and
#F   a set of integer <b> from <permgrp>.kbsM, the function constructs
#F   a permutation group as could have been returned by Mon2MatSym().
#F   In particular, this allows to obtain the left and right representation
#F   of <permgrp> for the given constituent block.
#F

BlockMon2BlockSym := function ( G, b )
  local Gb;

  if not ( IsPermGroup(G) and IsBound(G.M) and IsBound(G.kbsM) ) then
    Error("<G> must be a permgroup with <G>.M, <G>.kbsM bound");
  fi;
  if not b in G.kbsM then
    Error("<b> must be a block from <G>.kbsM");
  fi;

  # make a copy of G for block b
  Gb             := Copy(G);
  Gb.matrix      := List(G.M, Mi -> Sublist(Mi, b));
  Gb.baseIndices := SelectBaseFromList(Gb.matrix);
  Gb.baseMatrix  := Sublist(Gb.matrix, Gb.baseIndices)^-1;
  return Gb;
end;


#F Mon2-Block-Symmetry
#F ===================
#F
#F In the Mon2-Block-symmetry we are looking for monomial matrices L
#F with entries in [-1, 0, 1] such that M^-1*L*M is permuted 
#F block-diagonal. In the first place, we interested in which permuted
#F block-diagonal structures (kbs) are possible at all. In the second
#F place we single out those groups where the blocks are irreducible.
#F


#F Mon2BlockSymBySubsets( <mat> [, <kset> ] )
#F   compute Mon2BlockSym( <mat> ) by enumerating the
#F   k-subsets b of [1..n] for all 2 k <= n and finding
#F   Mon2MatSym( <mat>_{*,b} ). 
#F      If the parameter <kset> is supplied then only those
#F   k's appearing in <kset> are considered.
#F      The function returns a list of groups, each of which
#F   has a different kbs under conjugation with M. Use the
#F   function Mon2MatSymL/R to come to apply the left and right
#F   representation.
#F

Mon2BlockSymBySubsets := function ( arg )
  local
    M, kset, # the arguments
    L,       # the lattice, result
    n,       # Length(M), M^-1
    Sn,      # FullMon2Group(n)
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
  Sn      := FullMon2Group(n);
  Sn.name := ConcatenationString("MG", String(n));
  Sn.M    := M;
  Sn.invM := M^-1;
  Sn.kbsM := kbsMon2M(Sn.invM, Sn, Sn.M);  

  L := [ Sn ];
  for k in [1..QuoInt(n, 2)] do
    if k in kset  or  (n - k) in kset then
      InfoMon2Block1(
        "#I considering ", Binomial(n, k), " many ", 
        k, "-subsets of [1..", n, "]\n"
      );
      for b in Combinations([1..n], k) do

        # compute G_M({b, {1..n}-b})
        G      := Mon2MatSym( List(M, Mi -> Sublist(Mi, b)) );
        G      := AsSubgroup(Sn, G);
        G.M    := M;
        G.invM := Sn.invM;
        G.kbsM := kbsMon2M(Sn.invM, G, Sn.M);

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

  CompletedMon2BlockSym(L);  
  return L;
end;

# simple example:
#   L := Mon2BlockSymBySubsets( DCT_IV(8), [1..2] );
#   CharTable(L[2]);
#   DisplayMon2BlockSym(L);
