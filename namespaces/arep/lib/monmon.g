# -*- Mode: shell-script -*-
# Determination of Mon-Mon-Symmetry,
# MP, SE, 01.11.97 - , GAPv3.4
# SE, 16.01.01: check for L M = M R added; better reporting

# AREP 1.3: 
# ---------
# 24.08.03: Little bug removed in MonMonCycOrderMat. The function
#   crashed when cycs with absolute value 1 but infinite order
#   occurred as quotients.
# 25.08.06: Little chnage to support matrices of floats

# Literature:
#   S. Egner   : PhD. Thesis, University of Karlsruhe, 1997
#   M. Pueschel : PhD. Thesis, University of Karlsruhe, 1998

if not IsBound(InfoMonSym1) then
  InfoMonSym1 := Ignore;  # switch if on if you like
fi;

# Coding and Decoding of the Matrix
# =================================

# MonMonCycOrderMat( <matrix> ) 
#   calculates the default cyclotomic order for the encoding 
#   of <matrix>. This is from now on called the 'key'.

MonMonCycOrderMat := function ( M )
  local degreeroot, isreal, M1, classes, keys, cl, c, i;

  # degreeroot ( <cyc> )
  #   for a cyclotomic number <cyc> of absolute value 1 (not checked)
  #   returns the smallest integer n such that <cyc>^n = 1.
  #   If n = infinity (as it is for example for 3/5 + 4/5*i)
  #   then 1 is returned.
  degreeroot := function ( x )
    local n;

    # first check whether we have a cyc with infinite order
    if x ^ (2 * NofCyc(x)) <> 1 then
      return 1;
    fi;

    n := NofCyc(x);
    if n mod 2 <> 0 and x ^ n = -1 then
      return 2 * n;
    fi;

    return n;
  end;

  isreal := function ( x )
    if not IsCyc(x) then
      Error("<x> must be a cyclotomic");
    fi;

    return x = GaloisCyc(x, -1);
  end;

  # main function starts here

  if not IsMat(M) then
    Error("usage: MonMonCycOrderMat( <matrix> )");
  fi;

  # catch case of a mat of floats
  if ForAll(M, r -> ForAll(r, IsDouble)) then
    return 2;
  fi;

  if not ForAll(M, r -> ForAll(r, IsCyc)) then
    Error("<M> must contain only cyclotomics or floats");
  fi;

  # partition the elements in M according
  # to their absolut values
  M1      := Concatenation(M);
  if ForAll(M1, isreal) then
    return 2;
  fi;
  classes := 
    Equivalenceclasses(
      M1,
      function(x, y)
        return x * GaloisCyc(x, -1) = y * GaloisCyc(y, -1);
       end
      ).classes;

  keys := [ ];
  for cl in classes do
    c := cl[1];
    if c <> 0 then
      for i in [2..Length(cl)] do
        Add(keys, degreeroot(cl[i]/c));
      od;
    fi;
  od;

  return Lcm(keys);
end;


# MonMonEncodeMat( <matrix>, <key> ) adds rows and columns for 
#   all powers of a <key>-th root of unity and returns 
#   the so obtained matrix. 
#   If M is a (r x c)- matrix then MonMonEncodeMat(M) is a 
#   (<key> * r x <key> * c)-matrix.

MonMonEncodeMat := function ( M , key )
  local dim, M1;

  if not ( IsMat(M) and IsInt(key) and key >= 0 ) then
    Error("usage: MonMonEncodeMat( <matrix>, <key> )");
  fi;

  dim := DimensionsMat(M);
 
  # blow matrix up according to the key
  M1 := 
    TransposedMat(
      Concatenation(
        List(
	  [1..dim[1]], 
	  i -> List([1..key], j -> E(key)^(j-1)*M[i])
        )
      )
    );
  M1 := 
    TransposedMat(
      Concatenation(
        List(
	  [1..dim[2]], 
	  i -> List([1..key], j -> E(key)^(j-1)*M1[i])
        )
      )
    );
  
  return M1;
end;


# IsMonMonDecodable( <perm>, <k>, <key> )
#   tests if <perm> is a valid encoding of a monomial matrix
#   which has <k> blocks, each of size (<key> x <key>) and
#   the blocks are powers of (1,..,<key>).

IsMonMonDecodable := function ( x, k, key )
  local a, b, i, j;

  for i in [0..k-1] do
    a := (i*key+1) ^ x;
    for j in [2..key] do
      b := (i*key+j) ^ x;
      if a mod key = 0 then
        if not b = a - key + 1 then
          return false;
        fi;
      else
        if not b = a + 1 then
          return false;
        fi;
      fi;
      a := b;
    od;      
  od;
  return true;
end;

# MonMonDecodables( <k>, <key> )
#   constructs the wreath product (Z_key wr S_k) as a subgroup
#   of the S_(k*key). This group is the group of all decodable
#   permutations of type (k, key).

MonMonDecodables := function ( k, key )
  local G, Skkey, gens, i, j, x;

  # the parent group containing the result
  Skkey      := SymmetricGroup(k*key);
  Skkey.name := ConcatenationString("S", String(k*key));

  # Z_key = (1, .., key)
  gens := 
    [ MappingPermListList(
        (k-1)*key + [1..key],
        (k-1)*key + Concatenation([2..key], [1])
      )
    ]; 

  # S_k = mapped(< (1,k), (2,k), .., (k-1,k) >)
  for i in [1..k-1] do

    # mapped((i,k)) = (i+
    x := ();
    for j in [1..key] do
      x := x * ((i-1)*key+j, (k-1)*key+j);
    od;
    Add(gens, x);
  od;

  G      := Subgroup(Skkey, gens);
  G.size := key^k * Factorial(k);
  G.name := 
    ConcatenationString(
      "(Z", String(key), " wr ",
      "S", String(k), ")"
    );

  MakeStabChainStrongGenerators(
    G,
    1 + key*[0 .. k-1],
    G.generators
  );
  return G;
end;


#F Monomial Symmetry 
#F  =================
#F

#F MonMonSym( <mat> [, <cycOrder> ] ) 
#F MonMonSymL( <sym>, <x> )
#F MonMonSymR( <sym>, <x> )
#F   calculates the mon-mon symmetry of <mat> in the cyclotomic
#F   field of order <cycOrder>. If no <cycOrder> is supplied then
#F   it is computed automatically from the quotients of the entries
#F   of the matrix. The function returns a permutation group with
#F   the field .cycOrder being bound to the value of <cycOrder> used
#F   and with the field .dimensionsMat bound to the dimensions of the
#F   matrix <mat>.
#F      The functions MonMonSymL and MonMonSymR construct the Mon-
#F   objects of the left and right representations for the element/
#F   list of elements/subgroup <x> of the group <sym>, which ought 
#F   to be the result of a call to the function MonMonSym. In fact,
#F
#F       MatMon(MonMonSymL(sym, x)) * mat
#F     = mat * MatMon(MonMonSymR(sym, x)).
#F

MonMonSymL := function ( sym, x )
  local key, k, blocks, p, diag, exp, i;

  if not ( 
    IsPermGroup(sym) and 
    IsBound(sym.cycOrder) and 
    IsBound(sym.dimensionsMat)
  ) then
    Error("<sym> must be a mon-mon symmetry group");
  fi;

  if IsPerm(x) then

    # assume x in sym
    x      := PermPermSymL(sym, x);
    key    := sym.cycOrder;
    k      := sym.dimensionsMat[1]/key; # nr. of blocks
    blocks := List([1..k], i -> [(i - 1)* key + 1..i * key]);
    p      := Permutation(x, blocks, OnSets);
    diag   := [ ];
    for i in [1..k] do
      exp     := ((i - 1) * key + 1) ^ x mod key;
      diag[i] := E(key) ^ (exp - 1);
    od;
    return Mon(diag, p);

  elif IsList(x) then
    return List(x, x1 -> MonMonSymL(sym, x1));
  elif IsPermGroup(x) then
    return 
      Group(
        MonMonSymL(sym, x.generators),
        Mon((), sym.dimensionsMat[1]/sym.cycOrder)
      );
  else
    Error("<x> must be a perm, list-of-perm or permgroup");
  fi;
end;

MonMonSymR := function ( sym, x )
  local key, k, blocks, p, diag, exp, i;

  if not ( 
    IsPermGroup(sym) and 
    IsBound(sym.cycOrder) and 
    IsBound(sym.dimensionsMat)
  ) then
    Error("<sym> must be a mon-mon symmetry group");
  fi;

  if IsPerm(x) then

    # assume x in sym
    x      := PermPermSymR(sym, x);
    key    := sym.cycOrder;
    k      := sym.dimensionsMat[2]/key;
    blocks := List([1..k], i -> [(i - 1)* key + 1..i * key]);
    p      := Permutation(x, blocks, OnSets);
    diag   := [ ];
    for i in [1..k] do
      exp     := ((i - 1) * key + 1) ^ x mod key;
      diag[i] := E(key) ^ (-(exp - 1));
    od;
    return Mon(diag, p);

  elif IsList(x) then
    return List(x, x1 -> MonMonSymR(sym, x1));
  elif IsPermGroup(x) then
    return 
      Group(
        MonMonSymR(sym, x.generators),
        Mon((), sym.dimensionsMat[2]/sym.cycOrder)
      );
  else
    Error("<x> must be a perm, list-of-perm or permgroup");
  fi;
end;

MonMonSym := function ( arg )
  local 
    M, cycOrder,
    Menc, 
    G, G1, H,
    state;

  # decode and check arguments
  if Length(arg) = 1 then
    M        := arg[1];
    cycOrder := "automatic";
  elif Length(arg) = 2 then
    M        := arg[1];
    cycOrder := arg[2];
  else
    Error("usage: MonMonSym( <mat> [, <cycOrder> ] )");
  fi;
  if not IsMat(M) then
    Error("<M> must be a matrix");
  fi; 
  if cycOrder = "automatic" then
    cycOrder := MonMonCycOrderMat(M);
  fi;
  if not IsInt(cycOrder) and cycOrder >= 1 then
    Error("<cycOrder> must be a positive integer");
  fi;
  InfoMonSym1(
    "#I MonMonSym( <",
    Length(M), "x", Length(M[1]), "-matrix>, ", 
    "<cycOrder=", cycOrder, "> ) called\n"
  );

  # encode 
  Menc := MonMonEncodeMat(M, cycOrder);

  # calculate perm-perm symmetry
  G          := PermPermSym(Menc);
  G.cycOrder := cycOrder;

  # reduce G to the decodable permutations
  if not 
    ForAll(
      G.generators, 
      x -> 
        IsMonMonDecodable(
          x, 
          Sum(G.dimensionsMat)/G.cycOrder, 
          G.cycOrder
        )
    )
  then

    # intersect G with wreath product (Z_key wr S_k)
    InfoMonSym1("#I   Making MonMon-decodables with base and SGS.\n");
    H := 
      MonMonDecodables(
        Sum(G.dimensionsMat) / G.cycOrder,
        G.cycOrder
      );

    InfoMonSym1("#I   Intersecting perms and MonMon-decodables.\n");
    G1               := AsSubgroup(Parent(H), G);
    G1               := Intersection(G1, H);
    G1.dimensionsMat := G.dimensionsMat;
    G1.cycOrder      := G.cycOrder;

    # make G a parent group (and save BSGS)
    InfoMonSym1("#I   Making group with base and SGS.\n");
    G               := Group(G1.generators, ());
    G.dimensionsMat := G1.dimensionsMat;
    G.cycOrder      := G1.cycOrder;
    MakeStabChainStrongGenerators(
      G,
      PermGroupOps.Base(G),
      PermGroupOps.StrongGenerators(G)
    );

  fi;

  InfoMonSym1("#I   Check L M = M R for (L, R) in the mongroup \c");
  state := 
    IsPermGroup(G) and
    ForAll(
      G.generators, 
      g -> MatMon(MonMonSymL(G, g)) * M = M * MatMon(MonMonSymR(G, g))
    );
  if state then
    InfoMonSym1("is ok.\n");
    return G;
  else
    InfoMonSym1("has failed.\n");
    return false;
  fi;
end;
