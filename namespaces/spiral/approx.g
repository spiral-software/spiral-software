
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


if not IsBound(InfoApprox1) then InfoApprox1 := Ignore; fi;
if not IsBound(InfoApprox2) then InfoApprox2 := Ignore; fi;


# Universal tools
# ===============

# FloorRat( <rat> )
# CeilingRat( <rat> )
#   compute the largest integer less or equal x (FloorInt)
#   and the smallest integer greater or equal x (CeilingInt).
#

FloorRat := function ( x )
  if IsInt(x) then
    return x;
  elif x > 0 then
    return QuoInt(Numerator(x), Denominator(x));
  else # x < 0
    return -QuoInt(Numerator(-x), Denominator(-x)) - 1;
  fi;
end;

CeilingRat := function ( x )
  return -FloorRat(-x);
end;


# Isolation of real roots of rational polynomials
# ===============================================

#F SturmSequence( <f> )
#F   computes a Sturm sequence for the rational polynomial f.
#F

SturmSequence := function ( f )
  local S, g, h, i, contentRemoved;

  # contentRemoved( <f> )
  #   clears all denominators and removes the positive gcd of
  #   the remaining integer coefficients. Care is take not to
  #   alter the sign of the leading coefficient.

  contentRemoved := function ( f )
    local F, d, g, i;

    F := ShallowCopy(f.coefficients);

    d := 1;
    for i in [1..Length(F)] do
      if IsRat(F[i]) then
        d := LcmInt(d, Denominator(F[i]));
      fi;
    od;
    if d > 1 then
      for i in [1..Length(F)] do
        F[i] := d*F[i];
      od;
    fi;

    g := 0;
    i := 1;
    while i <= Length(F) and g <> 1 do
      if F[i] <> 0 then
        if g = 0 then
          g := AbsInt(F[i]); 
        else 
          g := GcdInt(g, AbsInt(F[i])); 
        fi;
      fi;
      i := i + 1;
    od;
    if g = 0 then
      g := 1;
    fi;

    return d/g * f;
  end;

  # handle trivial cases and make f a rational polynomial
  InfoApprox1("#I SturmSequence( <poly. of degree ", Degree(f), "> )\n");
  if Degree(f) = -1 then
    return [ ];
  fi;
  f := 
     Polynomial(
       Rationals, 
       Concatenation(List([1..f.valuation], i -> 0), f.coefficients)
     );

  # compute the Sturm sequence S for f; all denominators and gcds
  # of the coefficients are removed since we will evaluate the
  # sequence possibly a large number of times; subresultant methods
  # do not save time in this situation.

  S := [ contentRemoved(f) ];
  g := contentRemoved(Derivative(S[1]));
  while Degree(g) > 0 do
    Add(S, g);
    h := -contentRemoved(EuclideanRemainder(f, g));
    f := g;
    g := h;
  od;
  if Degree(g) <> -1 then
    Add(S, g);
  fi;
  # remove the gcd of f and f' from all polynomials in S;
  # this normalization makes S[1] squarefree and S[Length(S)] = 1.

  g := S[Length(S)];
  for i in [1..Length(S)] do
    S[i] := contentRemoved(EuclideanQuotient(S[i], g));
  od;
  return S;
end;


# SturmSequenceValue( <S>, <x> )
#   returns an integer which contains the relevant information
#   of evaluating the Sturm sequence S at the rational point x. 
#   Namely, the result is V = 2*v + w where v is the number of 
#   sign variations of S at x and w = 1-abs(sign(value(S[1], x))).
#   The point x may also be "infinity" or "-infinity".
#

SturmSequenceValue := function ( S, x )
  local V, s, si, sign, i;

  if Length(S) = 0 then
    return 0;
  fi;

  # sign(f) := SignInt(Value(f, x))
  if x = "-infinity" then
    sign := f -> (-1)^Degree(f) * SignInt(LeadingCoefficient(f));
  elif IsRat(x) then
    sign := f -> SignInt(Value(f, x));
  elif x = "infinity" then
    sign := f -> SignInt(LeadingCoefficient(f));
  fi;

  # count sign variations dropping zeros, and get sign(S[1])
  s := sign(S[1]);
  V := 1 - AbsInt(s);
  for i in [2..Length(S)] do
    si := sign(S[i]);
    if si <> 0 then
      if s <> 0 and si <> s then
        V := V + 2;
      fi;
      s := si;
    fi;
  od;
  return V;
end;


# SturmSequenceNrRealRoots( <a>, <Va>,  <b>, <Vb> )
#   returns the number of real roots of a rational polynomial 
#   f <> 0 in the *open* interval {x | a < x < b}, 
#   given the values Va, Vb of a Sturm sequence S_f at a, b 
#   as returned by SturmSequenceValue. The values a, b may be
#   rationals, "-infinity", or "infinity".
#

SturmSequenceNrRealRoots := function ( a, Va,  b, Vb )
  if 
    IsRat(a)        and IsRat(b) and a < b or
    a = "-infinity" and IsRat(b)           or
    IsRat(a)        and b = "infinity"     or
    a = "-infinity" and b = "infinity"
  then
    return QuoInt(Va, 2) - (QuoInt(Vb, 2)+(Vb mod 2));
  else
    return 0;
  fi;
end;


#F RealRootIsolationBinary(      <F> [, <min>, <max>] )
#F RealRootIsolationSternBrocot( <F> [, <min>, <max>] )
#F   return rational isolation intervals for all real roots of the
#F   rational polynomial f <> 0 in the closed interval [min..max].
#F   The argument F is either the polynomial f or a Sturm sequence
#F   for f. The bounds min, max are "-infinity", "infinity" or rational.
#F   The result is a list of rational pairs [a, b] which represent the 
#F   closed subinterval [a..b] of [min..max] such that there is exactly 
#F   one real x satisfying a <= x <= b and f(x) = 0. The two functions 
#F   differ in the way they generate the rational numbers a and b.
#F

RealRootIsolationBinary := function ( arg )
  local 
    S, min, max,                # completed argument list
    roots,                      # list of intervals [a, b] for the roots
    Vminf, Vm1, V0, Vp1, Vpinf, # sign variations at some points
    isolateSinglePoint,
    isolateOpenInterval;

  isolateSinglePoint := function ( a,Va )

    # stop if a is not in [min..max]
    if not (
      ( min = "-infinity" or min <= a ) and
      ( max =  "infinity" or max >= a )
    ) then
      return;
    fi;

    # simply test f(a) = 0
    InfoApprox2("[", a, "] \c");
    if Va mod 2 = 1 then
      Add(roots, [a, a]);
    fi;
  end;

  isolateOpenInterval := function ( a,Va, b,Vb )
    local c, Vc;

    # stop if (a..b) is disjoint to [min..max]
    if not (
      ( min = "-infinity" or b =  "infinity" or min < b ) and
      ( max =  "infinity" or a = "-infinity" or max > a )
    ) then
      return;
    fi;    

    # stop if (a..b) does not contain a root anymore
    InfoApprox2("(", a, "..", b, ") \c");
    if SturmSequenceNrRealRoots(a,Va, b,Vb) = 0 then
      return;
    fi;

    if IsRat(a) and IsRat(b) then

      # (a..b) finite: subdivide at (a+b)/2 until a root is isolated
      # and the bounds a, b both lie in [min..max]
      if 
        SturmSequenceNrRealRoots(a,Va, b,Vb) = 1 and
        ( min = "-infinity" or min <= a) and
        ( max =  "infinity" or max >= b)
      then
        Add(roots, [a, b]);
        return;
      fi;
      c := (a + b)/2;

    elif a = "-infinity" then

      # (-infinity..b): decrease b exponentially
      c := 2*b;

    elif b = "infinity" then
  
      # (a..infinity): increase a exponentially
      c := 2*a;

    fi;

    # split (a..b) as (a..c) union {c} union (c..b) 
    Vc := SturmSequenceValue(S, c);
    isolateOpenInterval(a,Va, c,Vc);
    isolateSinglePoint(c,Vc);
    isolateOpenInterval(c,Vc, b,Vb);
  end;

  # decode and check arg
  if Length(arg) = 1 then
    S   := arg[1];
    min := "-infinity";
    max :=  "infinity";
  elif Length(arg) = 3 then
    S   := arg[1];
    min := arg[2];
    max := arg[3];
  else
    Error("wrong number of arguments, 1 or 3 expected");
  fi;
  if not ( IsPolynomial(S) or IsList(S) and ForAll(S, IsPolynomial) ) then
    Error("<S> must be a rational polynomial or a Sturm sequence");
  fi;
  if not ( 
    ( IsRat(min) or min in ["-infinity", "infinity"] ) and
    ( IsRat(max) or max in ["-infinity", "infinity"] )
  ) then
    Error("<min> and <max> must be rational, \"infinity\", or \"-infinity\"");
  fi;
  if not IsList(S) then    
    S := SturmSequence(S);
  fi;
  InfoApprox1(
    "#I RealRootIsolationBinary( <poly. of degree ", 
     Degree(S[1]), 
    "> )\n"
  );

  # sample S at a number of standard points
  Vminf := SturmSequenceValue(S, "-infinity"); 
  Vm1   := SturmSequenceValue(S,          -1); 
  V0    := SturmSequenceValue(S,           0);
  Vp1   := SturmSequenceValue(S,           1); 
  Vpinf := SturmSequenceValue(S,  "infinity"); 

  InfoApprox2("#I   ");
  roots := [ ];
  isolateOpenInterval("-infinity", Vminf, -1, Vm1);
  isolateSinglePoint( -1, Vm1);
  isolateOpenInterval(-1, Vm1, 0, V0);
  isolateSinglePoint(  0, V0);
  isolateOpenInterval( 0, V0, 1, Vp1);
  isolateSinglePoint(  1, Vp1);
  isolateOpenInterval( 1, Vp1, "infinity", Vpinf);
  InfoApprox2("\n");

  return roots;
end;

RealRootIsolationSternBrocot := function ( arg )
  local 
    S, min, max,                # completed argument list
    roots,                      # list of intervals [a, b] for the roots
    Vminf, Vm1, V0, Vp1, Vpinf, # sign variations at some points
    isolateSinglePoint,
    isolateOpenInterval;

  isolateSinglePoint := function ( a,Va )
    if not (
      ( min = "-infinity" or min <= a ) and
      ( max =  "infinity" or max >= a )
    ) then
      return;
    fi;
    InfoApprox2("[", a, "] \c");
    if Va mod 2 = 1 then
      Add(roots, [a, a]);
    fi;
  end;

  isolateOpenInterval := function ( a,Va, b,Vb )
    local 
      c, Vc,             # the splitting point
      aN,aD, bN,bD,      # Numerator, Denominator of a, b
      splitPoint, valid, # functions to search faster
      kLo, kMi, kHi,     # positive integer indices to search faster
      cLo, cMi, cHi,     # points for kLo, kMi, kHi
      VLo, VMi, VHi;     # sign variations at cLo, cMi, cHi

    # stop if (a..b) is disjoint to [min..max]
    if not (
      ( min = "-infinity" or b =  "infinity" or min < b ) and
      ( max =  "infinity" or a = "-infinity" or max > a )
    ) then
      return;
    fi;    

    # stop if (a..b) does not contain a root anymore
    InfoApprox2("(", a, "..", b, ") \c");
    if SturmSequenceNrRealRoots(a,Va, b,Vb) = 0 then
      return;
    fi;

    if IsRat(a) and IsRat(b) then

      # (a..b) finite: subdivide until a root is isolated
      # and the bounds a, b both lie in [min..max]
      if 
        SturmSequenceNrRealRoots(a,Va, b,Vb) = 1 and
        ( min = "-infinity" or min <= a) and
        ( max =  "infinity" or max >= b)
      then
        Add(roots, [a, b]);
        return;
      fi;
      c := (Numerator(a)+Numerator(b))/(Denominator(a)+Denominator(b));

    elif a = "-infinity" then

      # (-infinity..b): decrease b by one 
      c := b - 1;

    elif b = "infinity" then
  
      # (a..infinity): increase a by one
      c := a + 1;

    fi;

    # speed up the search by exponentially increasing the stepsize
    # if (a..c) or (c..b) does not contain a root
    Vc := SturmSequenceValue(S, c);
    if 
      SturmSequenceNrRealRoots(a,Va, c,Vc) = 0 or 
      SturmSequenceNrRealRoots(c,Vc, b,Vb) = 0 
    then
      if a = "-infinity" then
        aN := -1;          
        aD :=  0;
      else
        aN := Numerator(a); 
        aD := Denominator(a);
      fi;
      if b = "infinity" then
        bN := 1; 
        bD := 0;
      else
        bN := Numerator(b);
        bD := Denominator(b);
      fi;
      if SturmSequenceNrRealRoots(a,Va, c,Vc) = 0 then
        splitPoint := k  -> (aN + k*bN)/(aD + k*bD);
        valid      := Vc -> Vc = Va;
      else
        splitPoint := k -> (k*aN + bN)/(k*aD + bD);
        valid      := Vc -> Vc = Vb;
      fi;

      # find the largest valid k by exponentially increasing k up to
      # an upper bound and binary search afterwards

      kLo := 1;
      cLo := c;
      VLo := Vc;
      kHi := 2;
      cHi := splitPoint(2);
      VHi := SturmSequenceValue(S, cHi);
      while valid(VHi) do
        kLo := kHi; 
        cLo := cHi; 
        VLo := VHi;
        kHi := 2*kHi;
	cHi := splitPoint(kHi);
	VHi := SturmSequenceValue(S, cHi);
      od;
      while kHi - kLo > 1 do
        kMi := QuoInt(kHi + kLo, 2);
        cMi := splitPoint(kMi);
        VMi := SturmSequenceValue(S, cMi);
        if valid(VMi) then
          kLo := kMi;
          cLo := cMi;
          VLo := VMi;
        else
          kHi := kMi;
          cHi := cMi;
          VHi := VMi;
        fi; 
      od;
      c  := cLo;
      Vc := VLo;
    fi;

    # split (a..b) into (a..c) union {c} union (c..b)
    isolateOpenInterval(a,Va, c,Vc);
    isolateSinglePoint(c,Vc);
    isolateOpenInterval(c,Vc, b,Vb);
  end;

  # decode and check arg
  if Length(arg) = 1 then
    S   := arg[1];
    min := "-infinity";
    max :=  "infinity";
  elif Length(arg) = 3 then
    S   := arg[1];
    min := arg[2];
    max := arg[3];
  else
    Error("wrong number of arguments, 1 or 3 expected");
  fi;
  if not ( IsPolynomial(S) or IsList(S) and ForAll(S, IsPolynomial) ) then
    Error("<S> must be a rational polynomial or a Sturm sequence");
  fi;
  if not ( 
    ( IsRat(min) or min in ["-infinity", "infinity"] ) and
    ( IsRat(max) or max in ["-infinity", "infinity"] )
  ) then
    Error("<min> and <max> must be rational, \"infinity\", or \"-infinity\"");
  fi;
  if not IsList(S) then
    S := SturmSequence(S);
  fi;
  InfoApprox1(
    "#I RealRootIsolationSternBrocot( <poly. of degree ", 
     Degree(S[1]), 
    "> )\n"
  );

  # precompute some standard values
  Vminf := SturmSequenceValue(S, "-infinity"); 
  Vm1   := SturmSequenceValue(S,          -1); 
  V0    := SturmSequenceValue(S,           0);
  Vp1   := SturmSequenceValue(S,           1); 
  Vpinf := SturmSequenceValue(S,  "infinity"); 

  InfoApprox2("#I   ");
  roots := [ ];
  isolateOpenInterval("-infinity", Vminf, -1, Vm1);
  isolateSinglePoint( -1, Vm1);
  isolateOpenInterval(-1, Vm1, 0, V0);
  isolateSinglePoint(  0, V0);
  isolateOpenInterval( 0, V0, 1, Vp1);
  isolateSinglePoint(  1, Vp1);
  isolateOpenInterval( 1, Vp1, "infinity", Vpinf);
  InfoApprox2("\n");

  return roots;
end;


# Iterative approximation of reals 
# ================================

# ApproximationInternal
#   fills the list path with information on the approximation until
#   it is accurate enough as specified by absError. The functions
#   lwb(a, b, k), upb(a, b, k) construct the k-th lower and upper
#   bound in (a..b). In particular lwb(a, b, 1) = upb(a, b, 1) and
#     a = lwb(a, b, 0) < lwb(a, b, 1) < ..  < b and
#     a <  .. < upb(a, b, 1) < upb(a, b, 0) = b.
#   The function returns false if sigma did not return a rational
#   and true otherwise.

ApproximationInternal := function ( sigma, absError, path, lwb, upb )
  local 
    a, x, b,       # points
    kLo, kMi, kHi, # integers to find transition point
    sLo, sMi, sHi, # sigma values
    s, i;          # sigma value, index into path

  # Initialize path[1], path[2] to contain a finite interval
  #
  if Length(path) < 2 then
    s := sigma(0); 
    if not IsRat(s) then return false; fi;
    if s = 0 then
      a := 0;
      b := 0;
    elif s > 0 then
      a := 0;
      b := 1;
      s := sigma(b); 
      if not IsRat(s) then return false; fi;
      while s > 0 do
        a := b;
        b := 2*b;
        s := sigma(b); 
        if not IsRat(s) then return false; fi;
      od;
      if s = 0 then
        a := b;
      else
        while b - a > 1 do
          x := QuoInt(a + b, 2);
          s := sigma(x); 
          if not IsRat(s) then return false; fi;
          if s = 0 then
            a := x;
            b := x;
          elif s > 0 then
            a := x;
          else
            b := x;
          fi;
        od;
      fi;
    else
      a := -1;
      b :=  0;
      s := sigma(a); 
      if not IsRat(s) then return false; fi;
      while s < 0 do
        b := a;
        a := 2*a;
        s := sigma(a); 
        if not IsRat(s) then return false; fi;
      od;
      if s = 0 then
        b := a;
      else
        while b - a > 1 do
          x := QuoInt(a + b, 2);
          s := sigma(x); 
          if not IsRat(s) then return false; fi;
          if s = 0 then
            a := x;
            b := x;
          elif s > 0 then
            a := x;
          else
            b := x;
          fi;
        od;
      fi;
    fi;
    path[1] := a;
    path[2] := b;
  fi;

  # Extend path to contain a sufficiently tight approximation.
  # There are two tightening operations for x in (a..b):
  #   A) a := x   and   B) b := x.
  # The list <path> encodes the sequence of intervals tightening
  # around c for sigma(c) = 0 in the form
  #      [path[1]..path[2]] 
  #   >= [path[3]..path[2]] 
  #   >= [path[3]..path[4]]
  #   >= ..
  # The method to extend the path is straight forward, except for
  # an additional improvement: We do not search by stepping single
  # A's and B's but we rather search exponentially/binary for the
  # next A/B-change.
  #
  i := Length(path);
  if i mod 2 = 0 then
    a := path[i-1];
    b := path[i];
  else
    a := path[i];
    b := path[i-1];
  fi;
  while not b - a < absError do
    if i mod 2 = 0 then

      # continue to raise lower bound
      kLo := 1;
      sLo := sigma(lwb(a, b, kLo));
      if not IsRat(sLo) then return false; fi;
      kHi := 2;
      sHi := sigma(lwb(a, b, kHi));
      if not IsRat(sHi) then return false; fi;
      while sHi > 0 do
        kLo := kHi;
        sLo := sHi;
        kHi := 2*kHi;
        sHi := sigma(lwb(a, b, kHi));
        if not IsRat(sHi) then return false; fi;
      od;
      if sHi = 0 then
        kLo := kHi;
        sLo := sHi;
      else
        while kHi - kLo > 1 do
          kMi := QuoInt(kHi + kLo, 2);
          sMi := sigma(lwb(a, b, kMi));
          if not IsRat(sMi) then return false; fi;
          if sMi = 0 then
            kLo := kMi;
            sLo := sMi;
            kHi := kMi;
            sHi := sMi;
          elif sMi > 0 then
            kLo := kMi;
            sLo := sMi;
          else
            kHi := kMi;
            sHi := sMi;
          fi;
        od;
      fi;
      x := lwb(a, b, kLo);
      s := sLo;

    else

      # continue to lower upper bound
      kLo := 1;
      sLo := sigma(upb(a, b, kLo));
      if not IsRat(sLo) then return false; fi;
      kHi := 2;
      sHi := sigma(upb(a, b, kHi));
      if not IsRat(sHi) then return false; fi;
      while sHi < 0 do
        kLo := kHi;
        sLo := sHi;
        kHi := 2*kHi;
        sHi := sigma(upb(a, b, kHi));
        if not IsRat(sHi) then return false; fi;
      od;
      if sHi = 0 then
        kLo := kHi;
        sLo := sHi;
      else
        while kHi - kLo > 1 do
          kMi := QuoInt(kHi + kLo, 2);
          sMi := sigma(upb(a, b, kMi));
          if not IsRat(sMi) then return false; fi;
          if sMi = 0 then
            kLo := kMi;
            sLo := sMi;
            kHi := kMi;
            sHi := sMi;
          elif sMi < 0 then
            kLo := kMi;
            sLo := sMi;
          else
            kHi := kMi;
            sHi := sMi;
          fi;
        od;
      fi;
      x := upb(a, b, kLo);
      s := sLo;

    fi;
    if s = 0 then
      a         := x;
      b         := x;
      path[i]   := x;
      path[i+1] := x;
      i         := i + 1;
    elif s > 0 then
      a := x;
      if i mod 2 = 0 then
        i := i + 1;
      fi;
      path[i] := x;
    else
      b := x;
      if i mod 2 <> 0 then
        i := i + 1;
      fi;
      path[i] := x;
    fi;
  od;
  return true;
end;


#F ApproximationBinary(      <sigma>, <absError> [, <path>] )
#F ApproximationSternBrocot( <sigma>, <absError> [, <path>] )
#F   returns a pair [a, b] such that 
#F     a <= c <= b   and   b - a < absError
#F   for a rational error bound absError > 0 and a real number c
#F   which is defined by the function sigma: Rationals -> Rationals. 
#F   The function sigma satisfies 
#F     sign sigma(x) = sign (c - x) for all rational x.
#F   If sigma does not return a Rational then the function is
#F   aborted immediately returning false. (This feature allows a 
#F   partially defined sigma to be used without messing up path.)
#F   The optional argument path is a list which is modified in order
#F   to contain the approximations computed so far. (In fact, it is
#F   some kind of run length encoding for the path of intervals 
#F   enclosing the number c.)
#F      The two variants of the approximation function differ in the 
#F   way they enumerate the rational numbers. Both start at (k..k+1) for
#F   integer k and split an interval (a..b) into (a..c), {c}, (c..b).
#F   The choice of the inner point is
#F     Binary        c = (a + b)/2, and
#F     Stern-Brocot  c = (N(a) + N(b))/(D(a) + D(b)) 
#F   where N: numerator, D: denominator. The methods return best binary
#F   fractions (Binary) and best rational approximations (Stern-Brocot)
#F   respectively.
#F

ApproximationBinary := function ( arg )
  local 
    sigma, absError, path,   # completed argument list
    a, b, x, a0, b0, am, bm, # points
    m, k,                    # integers denoting intermediate points
    lo, mi, hi,              # indices to find i in path
    s, i, K;                 # sigma value, index into path, intermediate

  # decode and type check argument list 
  if Length(arg) = 2 then
    sigma    := arg[1];
    absError := arg[2];
    path     := [ ];
  elif Length(arg) = 3 then
    sigma    := arg[1];
    absError := arg[2];
    path     := arg[3];
  else
    Error("wrong number of arguments, 2 or 3 expected");
  fi;
  if not ( 
    IsFunc(sigma) and 
    IsRat(absError) and absError > 0 and
    IsList(path)
  ) then
    Error("wrong arguments <sigma>, <absError>, <path>");
  fi;

  # path[1] contains the information stored for ApproximationBinary;
  # rename path[1] into path

  if not IsBound(path[1]) then
    path[1] := [ ];
  fi;
  path := path[1];

  # fill path sufficiently
  if 
    ApproximationInternal(
      sigma, absError, path, 
      function (a, b, k)
	return (a + (2^k - 1)*b)/2^k;
      end,
      function (a, b, k)
	return ((2^k - 1)*a + b)/2^k;
      end
    ) = false
  then
    return false;
  fi;

  # Get the easiest approximation better than absError from path.
  # We first find the least i where |path[i+1]-path[i]| < absError.
  # Then we expand the interval while it is not too large.

  lo := 1;
  hi := Length(path)-1;
  while hi - lo > 1 do
    mi := QuoInt(hi + lo, 2);
    if AbsInt(path[mi+1] - path[mi]) < absError then
      hi := mi;
    else
      lo := mi;
    fi;
  od;
  i := hi;
  if i = 1 then

    # just take the first interval
    a := path[i];
    b := path[i+1];

  elif i mod 2 = 0 then

    # decrease lower bound of (path[i+1]..path[i])
    # Consider a_0 < a_1 < .. < a_m < b for some m > 0
    # where a_k := 2^-k a_0 + (1 - 2^-k) b.
    # This implies
    #   m       = Log((a_0 - b)/(a_m - b), 2) and
    #   b - a_k = (b - a_0)*2^-k < absError <==> 2^k > (b - a_0)/absError.

    a0 := path[i-1];
    am := path[i+1];
    b  := path[i];

    if am = b then
      a := am; # a = b
    else
      m  := LogInt((a0 - b)/(am - b), 2);
      K  := (b - a0)/absError;
      if K <= 1 then
	k := 0;
      else
	k := LogInt(QuoInt(Numerator(K), Denominator(K)), 2);
	while not 2^k > K do
	  k := k + 1;
	od;
      fi;
      k := Minimum(k, m);
      a := (a0 + (2^k - 1)*b)/2^k;
    fi;
 
  else

    # increase upper bound of (path[i]..path[i+1])
    # Consider a < b_m < .. < b_1 < b_0 for some m > 0
    # where b_k := (1 - 2^-k) a + 2^-k b_0.
    # This implies
    #   m       = Log((a - b_0)/(a - b_m), 2) and
    #   b_k - a = (a - b_0)*2^-k < absError <==> 2^k > (a - b_0)/absError.

    a  := path[i];
    bm := path[i+1];
    b0 := path[i-1];

    if a = bm then
      b := bm; # a = b
    else
      m  := LogInt((a - b0)/(a - bm), 2);
      K  := (a - b0)/absError;
      if K <= 1 then
	k := 0;
      else
	k := LogInt(QuoInt(Numerator(K), Denominator(K)), 2);
	while not 2^k > K do
	  k := k + 1;
	od;
      fi;
      k := Minimum(k, m);
      b := ((2^k - 1)*a + b0)/2^k;
    fi;

  fi;
  return [a, b];
end;


ApproximationSternBrocot := function ( arg )
  local 
    sigma, absError, path,   # completed argument list
    a, b, x, a0, b0, am, bm, # points
    m, k,                    # integers denoting intermediate points
    lo, mi, hi,              # indices to find i in path
    s, i,                    # sigma value, index into path, intermediate
    N, D;                    # Numerator, Denominator for abbreviation

  # decode and type check argument list 
  if Length(arg) = 2 then
    sigma    := arg[1];
    absError := arg[2];
    path     := [ ];
  elif Length(arg) = 3 then
    sigma    := arg[1];
    absError := arg[2];
    path     := arg[3];
  else
    Error("wrong number of arguments, 2 or 3 expected");
  fi;
  if not ( 
    IsFunc(sigma) and 
    IsRat(absError) and absError > 0 and
    IsList(path)
  ) then
    Error("wrong arguments <sigma>, <absError>, <path>");
  fi;

  # path[2] contains the information stored for ApproximationSternBrocot;
  # rename path[2] into path

  if not IsBound(path[2]) then
    path[2] := [ ];
  fi;
  path := path[2];

  # fill path sufficiently
  if
    ApproximationInternal(
      sigma, absError, path, 
      function (a, b, k)
	return 
	  (   Numerator(a) + k*  Numerator(b) )/
	  ( Denominator(a) + k*Denominator(b) );
      end,
      function (a, b, k)
	return 
	  ( k*   Numerator(a) +   Numerator(b) )/
	  ( k*Denominator(a) + Denominator(b) );
      end
    ) = false
  then
    return false;
  fi;

  # Get the easiest approximation better than absError from path.
  # We first find the least i where |path[i+1]-path[i]| < absError.
  # Then we expand the interval while it is not too large.

  N := Numerator;
  D := Denominator;

  lo := 1;
  hi := Length(path)-1;
  while hi - lo > 1 do
    mi := QuoInt(hi + lo, 2);
    if AbsInt(path[mi+1] - path[mi]) < absError then
      hi := mi;
    else
      lo := mi;
    fi;
  od;
  i := hi;
  if i = 1 then

    # just take the first interval
    a := path[i];
    b := path[i+1];

  elif i mod 2 = 0 then

    # decrease lower bound of (path[i+1]..path[i])
    # Consider a_0 < a_1 < .. < a_m < b for some m > 0
    # where a_k := (N(a_0) + k N(b))/(D(a_0) + k D(b)).
    # This implies
    #   m = D(a_0)/D(b) * (a_m - a0)/(b - a_m)
    # and
    #   b - a_k = (b - a0)/(1 + k D(b)/D(a_0)) < absError
    #   <==> k > D(a_0)/D(b) * ((b - a0)/absError - 1).
 
    a0 := path[i-1];
    am := path[i+1];
    b  := path[i];
    if am = b then
      return [am, b]; # a = b
    else
      m  := D(a0)/D(b) * (am - a0)/(b - am);
      k  := CeilingRat(D(a0)/D(b) * ((b - a0)/absError - 1));
      k  := Minimum(Maximum(0, k), m);
      a  := (N(a0) + k*N(b))/(D(a0) + k*D(b));
    fi;
 
  else

    # increase upper bound of (path[i]..path[i+1])
    # Consider a < b_m < .. < b_1 < b_0 for some m > 0
    # where b_k := (k N(a) + N(b_0))/(k D(a) + D(b_0)).
    # This implies
    #   m = D(b_0)/D(a) * (b_0 - b_m)/(b_m - a)
    # and
    #   b_k - a = (b0 - a)/(1 + k D(a)/D(b_0)) < absError
    #   <==> k > D(b_0)/D(a) * ((b_0 - a)/absError - 1).
 
    a  := path[i];
    bm := path[i+1];
    b0 := path[i-1];
    if a = bm then
      return [a, bm]; # a = b
    else
      m  := D(b0)/D(a) * (b0 - bm)/(bm - a);
      k  := CeilingRat(D(b0)/D(a) * ((b0 - a)/absError - 1));
      k  := Minimum(Maximum(0, k), m);
      b  := (k*N(a) + N(b0))/(k*D(a) + D(b0));
    fi;

  fi;
  return [a, b];
end;


# Approximation of cyclotomics
# ============================

# ApproxCosPi( <x>, <absError> )
#   returns a pair [a, b] of rationals such that
#     a <= cos(pi x) <= b and b - a < absError
#   for rationals x and absError > 0. The approximation
#   is performed by ApproximationSternBrocot().
#

if not IsBound(ApproxCosPiTable) then
  ApproxCosPiTable := [ ];
fi;

ApproxCosPi := function ( x, absError )
  local 
    k, n,              # cos(pi x) = cos(2 pi k/n)
    f, ab, sign, path, # approximation data
    ik,                # Position(PrimeResidues(n), k)
    Sf,                # Sturm sequence for f
    Rf,                # isolation intervals [a, b] of real roots of f
    abs,               # Rf copied and arranged as for PrimeResidues
    signs,             # value of f at a for the [a, b] in abs
    paths;             # the lists which will contain approximations

  # check arguments
  if not ( IsRat(x) and IsRat(absError) and absError > 0 ) then
    Error("<x> must be a rational, <absError> positive rational");
  fi;

  # cos(pi x) = cos(2 pi k/n) where 0 <= k < n are relatively prime
  k := Numerator(x/2) mod Denominator(x/2); # 0 <= k < n
  n := Denominator(x/2);
 
  # provide ApproxCosPiTable[n] = [ F, abs, signs, paths ] where 
  #   F        is the coefficient list of the minimal polynomial 
  #            of the cos(2 pi k/n)
  #   abs[i]   = [a_i, b_i] such that a_i <= cos(2 pi R_i/n) <= b_i
  #            for the R = PrimeResidues(n)
  #   signs[i] = SignInt(Value(F, a_i))
  #   path[i]  is a list recording the approximation for cos(2 pi R_i/n)
  if not IsBound(ApproxCosPiTable[n]) then

    InfoApprox1("#I computing ApproxCosPiTable[", n, "]\n");
    if n = 1 then
      ApproxCosPiTable[n] := [ [-1, 1], [ [ 1, 1] ], [ 0 ], [ [] ] ];
    elif n = 2 then
      ApproxCosPiTable[n] := [ [ 1, 1], [ [-1,-1] ], [ 0 ], [ [] ] ];
    else
      f     := Polynomial(Rationals, MinPol((E(n) + E(n)^-1)/2));
      Sf    := SturmSequence(f);
      f     := Sf[1]; # (no denominators)
      Rf    := RealRootIsolationSternBrocot(Sf, -1, 1);
      abs   := Concatenation(Reversed(Rf), Rf);
      signs := List(abs, ab -> SignInt(Value(Sf[1], ab[1])));
      paths := List([1..Length(Rf)], i -> [ ]);
      paths := Concatenation(Reversed(paths), paths);
      if Length(abs) <> Phi(n) then
        Error("internal: |abs| <> Phi(n)");
      fi;
      ApproxCosPiTable[n] := 
        [ Concatenation(List([1..f.valuation], i -> 0), f.coefficients),
          abs,
          signs,
          paths
        ];
    fi;    
  fi;

  # get approximation data from ApproxCosPiTable[n]
  ik   := Position(PrimeResidues(n), k);
  f    := Polynomial(Rationals, ApproxCosPiTable[n][1]);
  ab   := ApproxCosPiTable[n][2][ik];
  sign := ApproxCosPiTable[n][3][ik];
  path := ApproxCosPiTable[n][4][ik];

  # compute the approximation
  return
    ApproximationSternBrocot(
      function ( x )
        if   x < ab[1] then return  1;
        elif x > ab[2] then return -1;
        elif sign >= 0 then return  Value(f, x); 
        else                return -Value(f, x);
        fi;
      end,
      absError,
      path
    );
end;


#F ApproxReCycSimple(     <z>, <absError>)
#F ApproxReCycBinary(     <z>, <absError>)
#F ApproxReCycSternBrocot(<z>, <absError>)
#F   return a pair [a, b] of rationals such that 
#F     a <= Re(z) <= b and b - a < absError
#F   for a cyclotomic element z and a positive rational
#F   error bound absError. The three functions differ in 
#F   the way they approximate.
#F     Simple      : binary fractions, not necessarly good ones
#F     Binary      : best binary fractions (n/2^k, k = min.)
#F     SternBrocot : best rational approximation (n/d, d = min.)
#F

ApproxReCycSimple := function ( z, absError )
  local
    x,             # the real part of z
    n,             # cyclotomic order of CF containing x
    c,             # coefficient list of x
    N,             # number of non-zero coefficients in c
    r, R,          # binary precision, 2^r > R
    ab0, ab1, AB2, # intermediate bounds
    ab,            # the result
    k,             # a prime residue for n
    err, err1;     # intermediate error bounds

  # check arguments
  if not ( IsCyc(z) and IsRat(absError) and absError > 0 ) then
    Error("<z> must be cyc, <absError> positive rational");
  fi;

  # catch the rational case
  x := (z + GaloisCyc(z, -1))/2;
  if IsRat(x) then
    return ApproximationBinary(y -> x - y, absError);
  fi;

  # Consider x = Sum(c_k cos(2 pi k/n) : k in PrimeResidues(n), c_k <> 0).
  # The method to compute the bounds a, b is
  #
  # 1. Find a0_k <= cos(2 pi k/n) <= b0_k by a Stern-Brocot approximation.
  #    This results in small numerators and denominators of a0_k, b0_k.
  #
  # 2. Set a1_k := c_k a0_k, b1_k := c_k b0_k for c_k > 0 and
  #        a1_k := c_k b0_k, b1_k := c_k a0_k for c_k < 0.
  #    Hence, a1_k <= c_k cos(2 pi k/n) <= b1_k.
  #    Since the numerators and denominators in a0_k, b0_k have been chosen
  #    minimal, the growth of numbers for a1_k, b1_k is relatively small.
  # 
  # 3. Set A2_k := floor(2^r a1_k), A2_k := ceiling(2^r b1_k).
  #    Hence, A2_k 2^-r <= a1_k, B2_k 2^-r >= b1_k.
  #    Clearly, A2_k and B2_k are integer. 
  #
  # 4. Form a := Sum(A2_k : k)/2^r, b := Sum(B2_k : k)/2^r.
  #    Hence, a <= cos(2 pi k/n) <= b as desired.
  #    The numbers do not grow exponentially since A2_, B2_k are integer.
  # 
  # Now, we compute the accuracy of the initial approximation in step 1. in
  # order to yield b - a < absError. Let N := |{k | c_k <> 0}|. We assume
  # that all summands introduce about the same error into the sum. Hence, let
  #   B2_k - A2_k < absError/N*2^r.
  # To choose r, consider the worst case bound
  #   B2_k - B2_k <= 2^r (b1_k + 1) - 2^r (a1_k - 1) = 2^r (b1_k - a1_k) + 2.
  # Hence, for some 2^r > 2 N/absError we find the chain of bounds
  #       b0_k - a0_k < (absError/N - 2^(1-r))/|c_k|
  #   ==> b1_k - a1_k < absError/N - 2^(1-r)
  #   ==> B2_k - B2_k < absError/N*2^r
  #   ==> b    - a    < absError.

  # represent x as Sum(c_k cos(2 pi k/n) : k) containing N summands
  n := NofCyc(x);
  c := CoeffsCyc(x, n);
  N := Number(c, ck -> ck <> 0);

  # choose r > 0 such that 2^r > 2 N/absError
  R := 2*N/absError;
  if R <= 1 then
    r := 1;
  else
    r := LogInt(QuoInt(Numerator(R), Denominator(R)), 2);
    while not 2^r > R do
      r := r + 1;
    od;
  fi;
  
  # compute a, b
  ab := [0, 0];
  for k in [0..n] do
    if IsBound(c[1+k]) and c[1+k] <> 0 then

      # lower the error bound by at most 1% to reduce the numbers involved
      err := (absError/N - 2/2^r)/AbsInt(c[1+k]);
      if err >= 1 then
        err1 := FloorRat(err);
      else
        err1 := 1/CeilingRat(1/err);
      fi;
      if (err - err1)/err > 1/100 then
        err1 := err;
      fi;

      # approximate cos(2 pi k/n)
      ab0 := ApproxCosPi(2*k/n, err1);

      # form the derived bound for the summand k
      if c[1+k] > 0 then
        ab1 := [ c[1+k]*ab0[1], c[1+k]*ab0[2] ];
      else
        ab1 := [ c[1+k]*ab0[2], c[1+k]*ab0[1] ];
      fi;
      AB2 := [ FloorRat(ab1[1]*2^r), CeilingRat(ab1[2]*2^r) ];
      ab  := [ ab[1] + AB2[1], ab[2] + AB2[2] ]; 
    fi;
  od;
  ab := [ ab[1]/2^r, ab[2]/2^r ];
  return ab;
end;


ApproxReCycBinary := function ( z, absError )
  local
    zRe,       # Re(z)
    ab,        # the result
    ab1,       # a tighter interval enclosing x
    absError1, # a lower error bound
    path,      # records ab approximation
    i;         # counter for the iterations

  # check arguments
  if not ( IsCyc(z) and IsRat(absError) and absError > 0 ) then
    Error("<z> must cyc, <absError> positive rational");
  fi;
  
  # catch the rational case
  zRe := (z + GaloisCyc(z, -1))/2;
  if IsRat(zRe) then
    return ApproximationBinary(x -> zRe - x, absError);
  fi;

  # The method we use to find the best binary approximation 
  # is to compute a more accurate approximation with ApproxReCycSimple
  # and to reapproximate it optimal binary again. This method seems to be
  # unelegant at first sight compared to approximating the roots
  # of the minimal polynomial. However, it avoids construction of
  # of the minimal polynomial at all!

  absError1 := absError/10;
  path      := [ ];
  for i in [1..100] do
    ab1 := ApproxReCycSimple(zRe, absError1);
    ab  :=
      ApproximationBinary(
	function (x)
	  if x < ab1[1] then
	    return 1;
	  elif x > ab1[2] then
	    return -1;
	  else
            return false; # aborts ApproximationBinary
	  fi;
	end,
	absError,
        path
      );
    if ab <> false then
      return ab;
    fi;
    absError1 := absError1/2;
  od;
  Error("internal: cannot find best binary approximation");
end;

ApproxReCycSternBrocot := function ( z, absError )
  local
    zRe,       # Re(z)
    ab,        # the result
    ab1,       # a tighter interval enclosing x
    absError1, # a lower error bound
    path,      # records ab approximation
    i;         # counter for the iterations

  # check arguments
  if not ( IsCyc(z) and IsRat(absError) and absError > 0 ) then
    Error("<z> must cyc, <absError> positive rational");
  fi;

  # catch the rational case
  zRe := (z + GaloisCyc(z, -1))/2;
  if IsRat(zRe) then
    return ApproximationSternBrocot(x -> zRe - x, absError);
  fi;

  # The method is the same as in ApproxReCycBinary.
  absError1 := absError/10;
  path      := [ ];
  for i in [1..100] do
    ab1 := ApproxReCycSimple(z, absError1);
    ab  :=
      ApproximationSternBrocot(
	function ( x )
	  if x < ab1[1] then
	    return 1;
	  elif x > ab1[2] then
	    return -1;
          else
	    return false; # aborts ApproximationSternBrocot
	  fi;
	end,
	absError,
        path
      );
    if ab <> false then
      return ab;
    fi;
    absError1 := absError1/2;
  od;
  Error("internal: cannot find best rational approximation");
end;


# Decimal fraction output
# =======================

#F StringDecimalRat( <x>, <mantissaDigits> )
#F   returns a string with a decimal expansion in scientific
#F   format for the rational x with mantissaDigits many digits
#F   after the decimal dot (no rounding). The syntax produced is
#F     [-]<digit>.<digits>e[-]<digits>
#F

StringDecimalRat := function ( x, mantissaDigits )
  local 
    out,    # the list of digits
    m, e,   # m*10^e = |x|
    mN, mD, # mN/mD = m
    i;      # index of the digit

  if not ( IsRat(x) and IsInt(mantissaDigits) ) then
    Error("<x> must be Rational and <mantissaDigits> integer");
  fi;
  if x = 0 then
    return "0.0";
  fi;

  # handle sign x
  if x >= 0 then
    out := "";
  else
    x   := -x;
    out := "-";
  fi;

  # x = m*10^e where 1 <= m < 10
  if x = 0 then
    e := 0;
  elif x < 1 then
    e := CeilingRat(1/(1+LogInt(QuoInt(Denominator(x), Numerator(x)), 10)));
  elif x = 1 then
    e := 0;
  else
    e := FloorRat(1+LogInt(QuoInt(Numerator(x), Denominator(x)), 10));
  fi;  
  m := x/10^e;
  while m < 1 do
    m := m*10;
    e := e-1;
  od;
  while m >= 10 do
    m := m/10;
    e := e+1;
  od;

  # convert mantissa m = mN/mD
  mN := Numerator(m);
  mD := Denominator(m);
  Append(out, String(QuoInt(mN, mD)));
  mN := mN mod mD; 
  Append(out, ".");
  for i in [1..Maximum(1, mantissaDigits)] do
    mN := 10*mN;
    Append(out, String(QuoInt(mN, mD)));
    mN := mN mod mD;
  od;

  # convert exponent
  Append(out, "e");
  Append(out, String(e));
  IsString(out);
  return out;  
end;

