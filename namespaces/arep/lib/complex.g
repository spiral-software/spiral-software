# -*- Mode: shell-script -*-
# Exact Complex Arithmetics
# SE, 17.01.96: initial version under GAP v3.4
# SE, 03.07.01: fixed 'Sqrt(12) < 0'-bug

# Literature:
#   [1] S. Lang: Algebra, 2nd ed.; Addison-Wesley 1984. 

# Cyclotomic fields embedded into complex numbers
# ===============================================

#F ImaginaryUnit()
#F Conjugate( <cyc-or-list> )
#F Re(        <cyc-or-list> )
#F Im(        <cyc-or-list> )
#F AbsSqr(    <cyc-or-list> )
#F   fundamental functions for complex numbers which are
#F   represented as cyclotomics. If a list is passed to
#F   these functions then the function is mapped over it.
#F

#F ExpIPi( <rational> ) ; exp(pi i n/d)
#F CosPi(  <rational> ) ; cos(pi n/d)
#F SinPi(  <rational> ) ; sin(pi n/d)
#F TanPi(  <rational> ) ; tan(pi n/d)
#F   functions to form certain complex values.
#F

# Imaginary unit i 

ImaginaryUnit := function () 
  return E(4); 
end;


# Complex conjugation for cyclotomic z

Conjugate := function ( z )
  if IsList(z) then
    return List(z, Conjugate);
  elif IsComplex(z) then
    return Complex(ReComplex(z), -ImComplex(z)); 
  elif IsDouble(z) then
    return z;
  else
    return GaloisCyc(z, -1);
  fi;
end;


# Re(z) for cyclotomic z

Re := function ( z )
  if IsList(z) then
    return List(z, Re);
  fi;
  return (z + Conjugate(z))/2;
end;


# Im(z) for cyclotomic z

Im := function ( z ) 
  if IsList(z) then
    return List(z, Im);
  fi;
  return (z - Conjugate(z))/(2 * ImaginaryUnit());
end;


# AbsSqr(z) for cyclotomic z

AbsSqr := function ( z )
  if IsList(z) then
    return List(z, AbsSqr);
  fi;
  return z * Conjugate(z);
end;


#F Sqrt( <rat> )
#F   computes the square root of a rational number x. The result is a
#F   cyclotomic field element a such that a^2 = x. Fixing the embedding 
#F   E(n) -> exp(2 pi i/n), we follow the convention a >= 0 for x >= 0
#F   and a = i*Sqrt(-x) for x < 0.
#F      The function breaks rationals down into positive integers and 
#F   these down into prime factors. For small integers a cache is maintained,
#F   for larger values and filling the cache a polynomial involving the
#F   Legendre symbol for the coefficients is evaluated.
#F

if not IsBound(SqrtTable) then
  SqrtTable          := [ 1, E(8) - E(8)^3 ]; 
  SqrtTableMaxLength := 200;
fi;

Sqrt := function (x)
  local s, pk, w, wk, nu;

  if IsDouble(x) then 
      return d_sqrt(x);
  elif not IsRat(x) then
      return Error("usage: Sqrt( <rational> )");
  elif not IsInt(x) then
      return Sqrt(Numerator(x)) / Sqrt(Denominator(x));
  fi;

  # x is an integer
  if x < -1 then
    return E(4) * Sqrt(-x);
  elif x = -1 then
    return E(4);
  elif x = 0 then
    return 0;
  elif x = 1 then
    return 1;
  elif x = 2 then
    return E(8) - E(8)^3;
  fi;

  # x > 2
  if IsBound(SqrtTable[x]) then
    return SqrtTable[x];
  fi;
  if not IsPrime(x) then
    s := 1;
    for pk in Collected( Factors( x ) ) do
      s := s * pk[1]^QuoInt(pk[2], 2);
      if pk[2] mod 2 = 1 then
        s := s * Sqrt(pk[1]);
      fi;
    od;
    if x <= SqrtTableMaxLength then
      SqrtTable[x] := s;
    fi;
    return s;
  fi;

  # x is an odd prime; use [1], Theorem 3.3
  s  := 0;
  w  := E(x);
  wk := 1;
  for nu in [1..x-1] do
    wk := w*wk;
    s  := s + Legendre(nu, x)*wk;
  od;
  if Legendre(-1, x) = -1 then
    s := s * E(4);
  fi;
  if x mod 4 = 3 then
    s := -s;
  fi;
  if x <= SqrtTableMaxLength then
    SqrtTable[x] := s;
  fi;
  return s;
end;


# Exp(pi i x) for rational x

ExpIPi := x -> Cond(
    IsDouble(x),   let(a := d_PI*x, Complex(d_cos(a), d_sin(a))),
    IsRat(x),      E( Denominator(x/2) )^Numerator(x/2),
    not IsRat(x),  Error("usage: ExpIPi( <rational> | <double> )")
);


# Cos(pi x) for rational x

CosPi := x -> Cond(
    IsDouble(x), d_cos(d_PI * x),
    Re( ExpIPi( x ) )
);

# Sin(pi x) for rational x

SinPi := x -> Cond(
    IsDouble(x), d_sin(d_PI * x),
    Im( ExpIPi( x ) )
);


# Tan(pi x) for rational x

TanPi := x -> Cond(
    IsDouble(x), d_tan(d_PI * x),
    SinPi(x) / CosPi(x)
);


# Useful things for better arithmetics
# ====================================


#F ReducingRatFactorCyc( <list-of../cyc> )
#F   let x be the argument then compute a non-zero 
#F   rational number b such that all denominators in 
#F   b*x are cleared and the remaining numerators do not 
#F   share a non-trivial factor. The sign of the first 
#F   non-zero coefficient of b*x is positive.
#F

ReducingRatFactor := function ( X )
  local d, n, denominator, numerator, sign;

  denominator := function ( X )
    local d, x;

    if IsRat(X) then
      if IsInt(X) then
        return 1;
      else
        return Denominator(X);
      fi; 
    elif IsList(X) then
      d := 1;
      for x in X do
        d := Lcm(d, denominator(x));
      od;
      return d;
    elif IsCyc(X) then
      d := 1;
      for x in CoeffsCyc(X, NofCyc(X)) do
        if x <> 0 then
          d := Lcm(d, Denominator(x));
        fi;
      od;
      return d;
    else
      Error("usage: ReducingFactorCyc( <list-of../cyc> )");
    fi;
  end;

  d := denominator( X );

  numerator := function ( X )
    local n, x;

    if IsRat(X) then
      return d*X;
    elif IsList(X) then
      n := 0;
      for x in X do
        n := Gcd(n, numerator(x));
      od;
      return n;
    else # IsCyc(X)
      n := 0;
      for x in CoeffsCyc(X, NofCyc(X)) do
        if x <> 0 then
          n := Gcd(n, d*x);
        fi;
      od;
      return n;
    fi;    
  end;

  n := numerator( X );

  sign := function ( X )
    local s, x;

    if IsRat(X) then
      if X > 0 then 
        return 1; 
      elif X = 0 then 
        return 0; 
      else 
        return -1; 
      fi;
    elif IsList(X) then
      for x in X do
        s := sign(x);
        if s <> 0 then
          return s;
        fi;
      od;
      return 0;
    else # IsCyc(X)
      for x in CoeffsCyc(X, NofCyc(X)) do
        if x <> 0 then
          return sign(x);
        fi;
      od;
      return 0;
    fi;
  end;

  if sign( X ) >= 0 then
    return d/n;
  else
    return -d/n;
  fi;
end;


#F ReducingCycFactor( <list-of../cyc> )
#F   let x be the argument, a list of lists of .. of cycs.
#F   Then this function computes a non-zero cyclotomic field 
#F   element b such that b*x is 'less complicated'.
#F

ReducingCycFactor := function ( X )
  local F, # field of the elements in X
        A, # A[i] = coeff. vector of F.base[i] * A1
        A1, # elements in X = first row of A
        tA; # columns of A, standardized

  # scalar case
  if IsCyc(X) then
    if X = 0*X then
      return 1;
    else
      return 1/X;
    fi;
  fi;

  A1 := Flat( X );
  F  := Field( A1 );
  A  := 
    List(
      F.base, 
      a -> Concatenation( List(A1, x -> Coefficients(F, a*x)) )
    );

  tA := 
    List(
      TransposedMat( A ),
      x -> x * ReducingRatFactor(x)
    );
  
  # ...weiter; 
  #    * Soll sparsifiziert werden, oder nur 
  #      standard Gauss-Elimination?
  #    * Fuer Gauss ist es gut, wenn immer der jeweils betragskleinste
  #      A[i][j] verwendet wird, um den Rest der Spalte zu plaetten.
  #    * Am Ende kann aus den Zeilen von A der beste gewaehlt werden.

  return A; # ...weiter
end;


