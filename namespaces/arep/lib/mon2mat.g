# -*- Mode: shell-script -*-
# Mon2-Mat-Symmetry
# =================
#
# Sebastian Egner, 5. August 1999 in GAP v3.4.4
# This module requires other files from "arep".

# RequirePackage("arep");

#F Encoding of [-1,1]-Monomial Matrices (Mon2) into Permutations
#F =============================================================
#F
#F We encode a monomial matrix with entries in [-1,0,1] of 
#F degree n in a permutation of degree 2*n. The encoding is
#F defined by choosing the base [e[1], .., e[n], -e[1], .., -e[n]]
#F of the underlying vector space and operating with the matrix
#F from the right. This conventions makes Mon2Encode an isomorphism
#F in the sense that Mon2Encode(m1*m2) = Mon2Encode(m1)*Mon2Encode(m2).
#F

#F Mon2Encode( <mon> )
#F   a permutation on [1..2*Length(mon.diag)] that encodes
#F   the Mon-object <mon> with entries in [-1, 1].
#F

Mon2Encode := function ( m )
  local n, p, i;

  if not ( IsMon(m) and ForAll(m.diag, x -> x in [-1, 1]) ) then
    Error("<m> must be a Mon with entries in [-1, 1]");
  fi;
  n := Length(m.diag);

  p := [ ];
  for i in [1..n] do
    if m.diag[i^m.perm] = 1 then
      p[i]   := i^m.perm;
      p[n+i] := i^m.perm + n;
    else
      p[i]   := i^m.perm + n;
      p[n+i] := i^m.perm;
    fi;
  od;
  return PermList(p);
end;

#F Mon2Decode( <perm>, <deg> )
#F   a Mon-object of degree deg with entries in [-1, 1] that is 
#F   represented by the permutation perm on [1..2*deg], or false
#F   the perm does not represent such a Mon-object.
#F

Mon2Decode := function ( perm, n )
  local p, d, i, im0, im1;

  if not ( IsPerm(perm) and IsInt(n) and n >= 1 ) then
    Error("<perm> must be a permutation on [1..2*<n>]");
  fi;

  d := [ ];
  p := [ ];
  for i in [1..n] do
    im0 := i^perm;
    im1 := (n + i)^perm;
    if im0 in [1..n] and im1 = n + im0 then
      p[i] := im0;
      d[i] := 1;
    elif im1 in [1..n] and im0 = n + im1 then
      p[i] := im1;
      d[i] := -1;
    else
      return false;
    fi;
  od;
  return Mon(d, PermList(p));
end;

#F FullMon2Group( <n> )
#F   a permutation group on [1..2*n] of all permutations that represent 
#F   a monomial matrix with entries in [-1, 0, 1] in the sense of the
#F   encoding Mon2Encode(). We call the group MG<n> for monomial group.
#F

FullMon2Group := function ( n )
  local G, base, sgs, i;

  # check argument
  if not ( IsInt(n) and n >= 1 ) then
    Error("<n> must be positive");
  fi;

  # construct BSGS
  base := [1..n];
  sgs  := [ ];
  for i in [1..n-1] do
    Add(sgs, (i,n)(n+i,2*n));
  od;
  for i in [1..n] do
    Add(sgs, (i,n+i));
  od;

  # construct the group with known BSGS
  G := Group(sgs, ());
  MakeStabChainStrongGenerators(G, base, sgs);
  G.name := Concatenation("MG", String(n));
  return G;
end;

#F Mon2-Mat-Symmetry
#F =================
#F

#F Mon2MatSym( <mat> )
#F Mon2MatSymL( <sym>, <x> )
#F Mon2MatSymR( <sym>, <x> )
#F   the Mon-Mat-symmetry of the (nr x nc)-matrix mat where nr >= nc 
#F   and mat does not contain linear dependent columns. The function
#F   only looks for monomial matrices with entries in [-1, 0, 1].
#F   The call Mon2MatSym(mat) returns a permgroup sym on [1..2*nr] 
#F   such that for all x in sym
#F
#F     MatMon(Mon2MatSymL(sym, x)) * mat = mat * Mon2MatSymR(sym, x).
#F
#F   The group record sym contains the additional fields
#F     sym.matrix      : identical to mat
#F     sym.baseIndices : indices of rows in mat forming a base
#F     sym.baseMatrix  : Sublist(mat, baseIndices)^-1
#F   This information is used in Mon2MatSymL/R. The <x> passed to 
#F   Mon2MatSymL/R may also be a list of perms or a permgroup.
#F

Mon2MatSymL := function ( sym, x )
  if IsPerm(x) then

    # <perm> 
    if not ( IsPermGroup(sym) and IsBound(sym.matrix) ) then
      Error("<sym> must be permgroup with sym.matrix etc.");
    fi;
    return Mon2Decode(x, Length(sym.matrix));

  elif IsList(x) then

    # <list-of-perm>
    return List(x, x1 -> Mon2MatSymL(sym, x1));

  elif IsPermGroup(x) then

    # <permgrp>
    if not ( IsPermGroup(sym) and IsBound(sym.matrix) ) then
      Error("<sym> must be permgroup with sym.matrix etc.");
    fi;
    return
      Group( 
        Mon2MatSymL(sym, x.generators), 
        Mon2MatSymL(sym, ()) 
      );

  else
    Error("<x> must be perm, list-of-perm or permgroup");
  fi;
end;

Mon2MatSymR := function ( sym, x )
  local G, L, R, n, i, j, k;

  if IsPerm(x) then

    # <perm>
    if not ( IsPermGroup(sym) and IsBound(sym.matrix) ) then
      Error("<sym> must be permgroup with sym.matrix etc.");
    fi;

    # compute R from L as
    #   sym.baseMatrix *
    #   Sublist(
    #     MatMon( Mon2Decode(x, Length(sym.matrix)) ) * sym.matrix,
    #     sym.baseIndices
    #   );

    L := Mon2Decode(x, Length(sym.matrix));

    n := Length(sym.baseIndices);
    R := NullMat(n, n);
    for i in [1..n] do
      for j in [1..n] do
        for k in [1..n] do
          R[i][j] := 
            R[i][j] + 
            sym.baseMatrix[i][k] *
            L.diag[sym.baseIndices[k]^L.perm] * 
            sym.matrix[sym.baseIndices[k]^L.perm][j];
        od;
      od;
    od;
if R <>  sym.baseMatrix * Sublist(
MatMon( Mon2Decode(x, Length(sym.matrix)) ) * sym.matrix,
sym.baseIndices) then
  Error("...");
fi;

    return R;

  elif IsList(x) then

    # <list-of-perm>
    return List(x, x1 -> Mon2MatSymR(sym, x1));

  elif IsPermGroup(x) then

    # <permgrp>
    G :=
      Group(
        Mon2MatSymR(sym, x.generators),
        sym.baseMatrix^0
      );
    return G;

  else
    Error("<x> must be perm, list-of-perm or permgroup");
  fi;
end;

Mon2MatSym := function ( M0 )
  local G, GPM, nr0, nc0;

  # decode and check arg
  if not IsMat(M0) then
    Error("<M0> must be matrix");
  fi;
  nr0 := Length(M0);
  nc0 := Length(M0[1]);
  if not nr0 >= nc0 then
    Error("<M0> must be matrix with more rows than columns");
  fi;

  # use Perm-Mat-Sym
  GPM := PermMatSym(Concatenation(M0, -M0));

  # filter out the monomial symmetry and reencode 
  G             := Intersection(GPM, FullMon2Group(nr0));
  G.baseIndices := SelectBaseFromList(M0);
  G.baseMatrix  := Sublist(M0, G.baseIndices)^-1;
  G.matrix      := M0;

  return G;
end;

# to do in in 'permmat.g':
#   * 'resul' -> result
#   * Orbit berechnungen anders
#   * error message 'with more rows' -> 'no less rows'
#   * eine Version die nur eine bestimmte Gruppe durchsucht
#     und eine bestimmte Gruppe schon weiss
#   * Intersection(GPM, FullMon2Group(nr0)) ist nicht noetig;
#     es wuerde reichen PermMat nur in FullMon2Group suchen
#     zu lassen und ihm das Zentrum auch gleich mitzugeben.

