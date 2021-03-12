
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F Random SPLs
#F -----------
#F

#F RandomPermSPL ( <n> )
#F   returns a random spl of type "perm" or "symbol" representing
#F   a <n> x <n> permutation matrix.
#F

RandomPermSPL := function ( n )
  local type;

  type := Random(["perm", "J", "L"]);
  if type = "perm" then
    return Perm(Random(SymmetricGroup(n)), n);
  elif type = "J" then
    return SPLSymbol("J", n);
  elif type = "L" then
    return SPLSymbol("L", [n, Random(DivisorsInt(n))]);
  else
    Error("should not happen");
  fi;
end;


#F RandomSymbolSPL ( <n> )
#F   returns a random spl of type "symbol" representing
#F   a <n> x <n> matrix.
#F

RandomSymbolSPL := function ( n )
  local type, d;

  if n = 1 then
    type := Random(["I", "J"]);
  elif n = 2 then
    type := Random(["I", "J", "F", "Rot"]);
  elif IsPrime(n) then
    type := Random(["I", "J"]);
  else
    type := Random(["I", "J", "L", "T"]);
  fi;
  if type in ["I", "J", "F"] then
    return SPLSymbol(type, n);
  elif type = "Rot" then
    return SPLSymbol("Rot", Random(Rationals));
  elif type in ["L", "T"] then
    d := Random([2..Tau(n)-1]);
    return SPLSymbol(type, [n, DivisorsInt(n)[d]]);
  else
    Error("should not happen");
  fi;
end;


#F RandomTerminalSPL ( <n> )
#F   returns a random terminal spl representing a <n> x <n> matrix.
#F

RandomTerminalSPL := function ( n )
  local type, randscalar, L, i, P, p;

  # function to return a somewhat random scalar <> 0
  randscalar := function ( ) 
    local L, den;

    L := 
      Concatenation(
        [-5..-1], [1..5], List([1/3,2/5,3/7], CosPi), List([2..5], E)
      );
    return Random(L)/Random([1..5]);
  end;

  # choose type, mats only for n <= 3, give better odds to
  # composed types
  if n <= 3 then
    type := 
      Random( [
	"mat", "sparse", "diag", "perm", "symbol", 
        "compose", "directSum", "tensor", 
        "compose", "directSum", "tensor", 
        "scalarMultiple", "conjugate"
      ] );
  else
    type := 
      Random( [
	"sparse", "diag", "perm", "symbol", 
        "compose", "directSum", "tensor", 
        "scalarMultiple", "conjugate"
      ] );
  fi;  

  # dispatch on types
  if type = "mat" then

    return Mat(List([1..n], i -> List([1..n], j -> randscalar())));

  elif type = "sparse" then
    
    # choose n entries
    # make sure there are no doubles
    L := [ ];
    P := [ ];
    for i in [1..n] do
      repeat
        p := [Random([1..n]), Random([1..n])];
      until not p in P;
      AddSet(P, p); 
      Add(L, [p[1], p[2], randscalar()]);
    od;
    if ForAll(L, t -> t[1] <> n or t[2] <> n) then

      # add entry in lower right corner
      Add(L, [n, n, randscalar()]);
    fi;
    return Sparse(L);

  elif type = "diag" then

    return Diag(List([1..n], i -> randscalar()));

  elif type = "perm" then

    return RandomPermSPL(n);

  elif type = "symbol" then

    return RandomSymbolSPL(n);

  elif type = "compose" then

    # choose 2 factors
    i := Random([2]);
    return Compose(List([1..i], i -> RandomTerminalSPL(n)));

  elif type = "directSum" then

    if n <= 3 then
      return RandomTerminalSPL(n);
    fi;

    # choose partition into 2 summands
    i := Random([2]);
    return 
      DirectSum(List(Random(OrderedPartitions(n, i)), RandomTerminalSPL));

  elif type = "tensor" then

    if IsPrime(n) or n = 1 then
      return RandomTerminalSPL(n);
    else
      i := Random(DivisorsInt(n));
      return Tensor(RandomTerminalSPL(i), RandomTerminalSPL(n/i));
    fi;

  elif type = "scalarMultiple" then

    return Scale(randscalar(), RandomTerminalSPL(n));

  elif type = "conjugate" then

    return Conjugate(RandomTerminalSPL(n), RandomPermSPL(n));

  else
    Error("should not happen");
  fi;
end;
