# -*- Mode: shell-script -*-
# Discrete Linear Signal Transforms
# SE, MP ab 27.02.96, GAP v3.4

# 27.02.96: Erste Version
# 02.04.96: Hartley-T.; DCT/DCT-IV
# 06.02.97: char > 0, ADFT, PADFT, RecognizeADFT
# 07.02.97: Korrektur: PADFT baut nicht unbedingt eine Basis
# 03.12.97: DCT_I
# 27.05.98: MP, Sinustransformationen

# AREP 1.2
# --------
# 19.10.99: Sinus trafos korrigiert gemaess dem Buch
#           Discrete Cosine Transform (Rao/Yip), Academic Press, 1990
# 13.03.00: Cos-DFT and Sin-DFT eingefuehrt.
# 28.03.00: Unskalierte trigonometrische Trafos eingefuehrt: 
#           DCT_IIIcunscaled, etc. DST_I korrigiert.

# AREP 1.3: 
# ---------
# 14.03.02: MDCT, z.B.: Britanak and Rao, An Efficient Implementation of
#           the Forward and Inverse MDCT in MPEG Audio Coding, 
#           IEEE SPL 8(2) 2001.
# 24.02.03: polynomial DTTs and skew DTTs, see MP's long DTT paper 
#           (in preparation)
# 27.06.03: RDFT (real DFT)

#F Discrete Signal Transforms
#F ==========================
#F

# Literature:
#   [1] S. Lang: Algebra, 2nd ed.; Addison-Wesley 1984. (HA Beth, D Lan).
#   [2] Elliott, Rao: Fast Transforms.
#   [3] Clausen, Baum: FFT.
#   [4] H. S. Malvar: Signal processing with lapped transforms.
#   [5] A. N"uckel, A. Klappenecker: On the Parametrization of
#       Algebraic Discrete Fourier Transforms. Extended Abstract
#       submitted to EUROCAST '97.
#   [6] A. Mertins: Signaltheorie

# Discrete Fourier Transform (DFT)
# ================================

#F DiscreteFourierTransform( <rootOfUnity> )
#F DiscreteFourierTransform( <N> [, <char>] )
#F InverseDiscreteFourierTransform( <rootOfUnity> )
#F InverseDiscreteFourierTransform( <N>, [, <char>] )
#F   constructs the Discrete Fourier Transform on N points
#F   or its inverse. (Elliott, Rao, 3.) If a prime int 
#F   <char> is supplied, then a DFT in that characteristic 
#F   is constructed if possible. If <rootOfUnity> is supplied
#F   then the transform of length Order(<rootOfUnity>) is
#F   constructed.
#F

DiscreteFourierTransform := function ( arg )
  local N, p, w, W, k;

  if 
    Length(arg) = 1 and IsInt(arg[1]) and arg[1] >= 1 or
    Length(arg) = 2 and IsInt(arg[1]) and arg[1] >= 1 and
    arg[2] = 0
  then
    N := arg[1];
    p := 0;
    w := ExpIPi(2/N);
  elif Length(arg) = 1 and arg[1] in FieldElements then
    w := arg[1];
    p := Characteristic(DefaultField(w));
    if p = 0 then
      N := OrderCyc(w);
    else
      N := OrderFFE(w);
    fi;
    if N = "infinity" then
      Error("<w> must be a root of unity");
    fi;
  elif 
    Length(arg) = 2 and 
    IsInt(arg[1]) and arg[1] >= 1 and
    IsInt(arg[2]) and arg[2] >= 2 and IsPrimeInt(arg[2])
  then
    N := arg[1];
    p := arg[2];
    k := 1;
    while not (p^k - 1) mod N = 0 do
      k := k + 1;
      if not p^k <= 2^16 then
        Error("cannot construct primitive <N>-th root of unity");
      fi;
    od;
    w := Z(p^k)^QuoInt(p^k - 1, N);
  else
    Error("wrong arguments");
  fi;

  W := [ w^0 ];
  for k in [1 .. N-1] do
    W[k+1] := W[k] * w;
  od;
  return 
    List([0..N-1], i -> List([0..N-1], j -> 
      W[((i*j) mod N)+1]
    ));
end;

InverseDiscreteFourierTransform := function ( arg )
  local F, N, NInv, k, Fk;

  if Length(arg) = 1 then
    F := DiscreteFourierTransform(arg[1]);
  elif Length(arg) = 2 then
    F := DiscreteFourierTransform(arg[1], arg[2]);
  else
    Error("wrong arguments");
  fi;

  N := Length(F);
  for k in [2..QuoInt(N+1, 2)] do
    Fk       := F[k];
    F[k]     := F[N-k+2];
    F[N-k+2] := Fk;
  od;

  NInv := (N * F[1][1]^0)^-1;
  for k in [1..N] do
    F[k] := NInv * F[k];
  od;
  return F;
end;

# Beispiel
# --------
#
# P23 := MatPerm((2,3), 4);
# I2  := IdentityMat(2);
# F2  := [[1,1], [E(4), -E(4)]]; 
# DFT(4) = 
#   P23 * TensorProductMat(I2, DFT(2)) * 
#   P23 * DirectSumMat(DFT(2), F2) * P23;


# Real Discrete Fourier Transform (RDFT)
# ======================================

#F RDFT ( <n> [, <k> ] )
#F   returns a real DFT of size <n>, where the defining root of unity
#F   for the DFT is given by E(n)^<k>.
#F

RDFT := function ( arg )
  local n, k;

  if Length(arg) = 1 then
    n := arg[1];
    k := 1;
  elif Length(arg) = 2 then
    n := arg[1];
    k := arg[2];
  else
    Error("usage: RDFT ( <n> [, <k> ]");
  fi;

  return 
    Concatenation(
      List(
        [0..Int(n/2)],
        r -> List([0..n-1], c -> CosPi(k*2*r*c/n))
      ),
      List(
        [Int(n/2)+1..n-1],
        r -> List([0..n-1], c -> -SinPi(k*2*r*c/n))
      )
    );
end;


# Discrete Hartley Transform (DHT)
# ================================

#F DiscreteHartleyTransform( <N> )
#F InverseDiscreteHartleyTransform( <N> )
#F   constructs the Discrete Hartley Transform on N points 
#F   or its inverse. (Malvar, 1.2.3)
#F

DiscreteHartleyTransform := function ( N )
  local H, i, j;

  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DiscreteHartleyTransform( <positive-int> )");
  fi;

  H := [];
  for i in [N-1, N-2 .. 0] do
    for j in [N-1, N-2 .. 0] do
      if not IsBound(H[1+i*j]) then
        H[1+i*j] := 1/Sqrt(N)*(CosPi(2*i*j/N) + SinPi(2*i*j/N));
      fi;
    od;
  od;
  return List([0..N-1], i -> List([0..N-1], j -> H[1+i*j]));
end;

InverseDiscreteHartleyTransform := 
  DiscreteHartleyTransform;


# Discrete Cosine/Sine Transforms 
# ===============================

# DCT, DST, type-I - type-IV, unscaled and scaled. The scaled versions
# are orthonormal. The unscaled versions are denoted by the suffix "unscaled".
# Note that type-III is the transposed of type-II.

# Unscaled DCTs
# -------------

#F DCT_Iunscaled( <N> )
#F   constructs an unscaled Discrete Cosine Transform (type-I) 
#F   on N points (the matrix contains pure cosines).
#F

DCT_Iunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DCT_Iunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> CosPi(i*j/(N-1))));
end;

#F DCT_IIunscaled( <N> )
#F   constructs an unscaled Discrete Cosine Transform (type-II) 
#F   on N points (the matrix contains pure cosines).
#F

DCT_IIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DCT_IIunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> CosPi(i*(j + 1/2)/N)));
end;

# for convenience
DCTunscaled := DCT_IIunscaled;


#F DCT_IIIunscaled( <N> )
#F   constructs an unscaled Discrete Cosine Transform (type-III) 
#F   on N points (the matrix contains pure cosines).
#F

DCT_IIIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DCT_IIIunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> CosPi((i + 1/2)*j/N)));
end;


#F DCT_IVunscaled( <N> )
#F   constructs an unscaled Discrete Cosine Transform (type-IV) 
#F   on N points (the matrix contains pure cosines).
#F

DCT_IVunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DCT_IVunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> CosPi((i + 1/2)*(j + 1/2)/N)));
end;


#F DCT_Vunscaled( <N> )
#F   constructs an unscaled Discrete Cosine Transform (type-V) 
#F   on N points (the matrix contains pure cosines).
#F

DCT_Vunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DCT_Vunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> CosPi(i*j/(N-1/2))));
end;


#F DCT_VIunscaled( <N> )
#F   constructs an unscaled Discrete Cosine Transform (type-VI) 
#F   on N points (the matrix contains pure cosines).
#F

DCT_VIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DCT_VIunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> CosPi(i*(j+1/2)/(N-1/2))));
end;


#F DCT_VIIunscaled( <N> )
#F   constructs an unscaled Discrete Cosine Transform (type-VII) 
#F   on N points (the matrix contains pure cosines).
#F

DCT_VIIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DCT_VIIunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> CosPi((i+1/2)*j/(N-1/2))));
end;


#F DCT_VIIIunscaled( <N> )
#F   constructs an unscaled Discrete Cosine Transform (type-VIII) 
#F   on N points (the matrix contains pure cosines).
#F

DCT_VIIIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DCT_VIIIunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> CosPi((i+1/2)*(j+1/2)/(N+1/2))));
end;



# Unscaled DSTs
# -------------

#F DST_Iunscaled( <N> )
#F   constructs an unscaled Discrete Sine Transform (type-I) 
#F   on N points (the matrix contains pure sines).
#F

DST_Iunscaled := function ( N )
  if not ( IsInt(N) and N > 1 ) then
    Error("usage: DST_Iunscaled( <int > 1> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> SinPi((i + 1)*(j + 1)/(N + 1))));
end;

#F DST_IIunscaled( <N> )
#F   constructs an unscaled Discrete Sine Transform (type-II) 
#F   on N points (the matrix contains pure sines).
#F

DST_IIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DST_IIunscaled( <positive-int> )");
  fi;
  return
    List([1..N], i -> List([1..N],  j -> SinPi(i*(j - 1/2)/N)));
end;

# for convenience
DSTunscaled := DST_IIunscaled;


#F DST_IIIunscaled( <N> )
#F   constructs an unscaled Discrete Sine Transform (type-III) 
#F   on N points (the matrix contains pure sines).
#F

DST_IIIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DST_IIIunscaled( <positive-int> )");
  fi;
  return
    List([1..N], i -> List([1..N],  j -> SinPi((i - 1/2)*j/N)));
end;


#F DST_IVunscaled( <N> )
#F   constructs an unscaled Discrete Sine Transform (type-IV) 
#F   on N points (the matrix contains pure sines).
#F

DST_IVunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DST_IVunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> SinPi((i + 1/2)*(j + 1/2)/N)));
end;


#F DST_Vunscaled( <N> )
#F   constructs an unscaled Discrete Sine Transform (type-V) 
#F   on N points (the matrix contains pure sines).
#F

DST_Vunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DST_Vunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> SinPi((i + 1)*(j + 1)/(N+1/2))));
end;


#F DST_VIunscaled( <N> )
#F   constructs an unscaled Discrete Sine Transform (type-VI) 
#F   on N points (the matrix contains pure sines).
#F

DST_VIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DST_VIunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> SinPi((i + 1)*(j + 1/2)/(N+1/2))));
end;


#F DST_VIIunscaled( <N> )
#F   constructs an unscaled Discrete Sine Transform (type-VII) 
#F   on N points (the matrix contains pure sines).
#F

DST_VIIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DST_VIIunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> SinPi((i + 1/2)*(j + 1)/(N+1/2))));
end;


#F DST_VIIIunscaled( <N> )
#F   constructs an unscaled Discrete Sine Transform (type-VIII) 
#F   on N points (the matrix contains pure sines).
#F

DST_VIIIunscaled := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DST_VIIIunscaled( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1],  j -> SinPi((i + 1/2)*(j + 1/2)/(N-1/2))));
end;


# Skew DTTs
# ---------

# exactly 4 of the 16 DTTs have as associated polynomial algebra C[x]/T_n, where
# T_n is the nth Chebyshev polynomial, namely DCT and DST of type III and IV.
# The associated "skew" DTT is associated to the algebra C[x]/(T_n - cos(r*pi))
# with the same basis. In particular, for r = 1/2 we get the ordinary DTT as special 
# case.

# normalizeCosine( <rat> )
#   normalizes the rational number <rat> to the unique rational number r
#   such that
#     r in [0, 1] and cos(r) = cos(<rat>)

normalizeCosine := function ( r )

  if not IsRat(r) then
    Error("<r> must be rational");
  fi;

  # move into the interval [0, 2)
  r := 2 * (r/2 - Int(r/2));
  if r < 0 then
    r := r + 2;
  fi;

  # flip into the interval [0, 1]
  if r > 1 then
    r := 2 - r;
  fi;

  return r;
end;

# zerosT( <n>, <r> )
#   returns (coded as rational number r_i) ascendingly list of the zeros of the polynomial 
#     T_n - cos(r*pi) = 2^(n-1) * prod_(0 <= i < n) (x - cos(r_i*pi))
#   where r_i = (r + 2i)/k is reduced to lie in the interval [0, 1].
#   Note that the final order depends on the sizes of r_i, not on i.
#   r has to be a rational number, T_n is a Chebyshev polynomial of the 
#   first kind. Setting r = 1/2 yields the zeros of T_n.

zerosT := function ( n, r )
  local L;

  # check args
  if not ( IsInt(n) and n > 0 ) then
    Error("<n> has to be a positive integer");
  fi;
  if not IsRat(r) then
    Error("<r> has to be a rational number");
  fi;

  # list of normalized r_i's
  L := List([0..n - 1], i -> normalizeCosine((r + 2*i)/n));

  # sort by size
  Sort(L);

  return L;
end;


#F SkewDCT_IIIunscaled( <n>, <r> )
#F   returns the skew unscaled DCT of type III for the polynomial algebra 
#F   C[x]/(T_n - cos(r*pi)) with basis (T_0, .., T_(n-1)).
#F   In particular, SkewDCT_IIIunscaled(n, 1/2) = DCT_IIIunscaled(n).
#F   This transform is a polynomial transform (since type III)
#F

SkewDCT_IIIunscaled := function ( n, r )
  local L;

  # check args
  if not ( IsInt(n) and n > 0 ) then
    Error("<n> has to be a positive integer");
  fi;
  if not IsRat(r) then
    Error("<r> has to be a rational number");
  fi;

  L := zerosT(n, r);
  return List(L, x -> List([0..n - 1], i -> CosPi(i * x)));
end;


#F SkewDCT_IVunscaled( <n>, <r> )
#F   returns the skew unscaled DCT of type IV for the polynomial algebra 
#F   C[x]/(T_n - cos(r*pi)) with basis (V_0, .., V_(n-1)).
#F   In particular, SkewDCT_IVunscaled(n, 1/2) = DCT_IVunscaled(n).
#F

SkewDCT_IVunscaled := function ( n, r )
  local L;

  # check args
  if not ( IsInt(n) and n > 0 ) then
    Error("<n> has to be a positive integer");
  fi;
  if not IsRat(r) then
    Error("<r> has to be a rational number");
  fi;

  L := zerosT(n, r);
  return List(L, x -> List([0..n - 1], i -> CosPi((i + 1/2) * x)));
end;


#F SkewDST_IIIunscaled( <n>, <r> )
#F   returns the skew unscaled DST of type III for the polynomial algebra 
#F   C[x]/(T_n - cos(r*pi)) with basis (T_0, .., T_(n-1)).
#F   In particular, SkewDST_IIIunscaled(n, 1/2) = DST_IIIunscaled(n).
#F

SkewDST_IIIunscaled := function ( n, r )
  local L;

  # check args
  if not ( IsInt(n) and n > 0 ) then
    Error("<n> has to be a positive integer");
  fi;
  if not IsRat(r) then
    Error("<r> has to be a rational number");
  fi;

  L := zerosT(n, r);
  return List(L, x -> List([0..n - 1], i -> SinPi((i + 1) * x)));
end;


#F SkewDST_IVunscaled( <n>, <r> )
#F   returns the skew unscaled DST of type IV for the polynomial algebra 
#F   C[x]/(T_n - cos(r*pi)) with basis (V_0, .., V_(n-1)).
#F   In particular, SkewDST_IVunscaled(n, 1/2) = DST_IVunscaled(n).
#F

SkewDST_IVunscaled := function ( n, r )
  local L;

  # check args
  if not ( IsInt(n) and n > 0 ) then
    Error("<n> has to be a positive integer");
  fi;
  if not IsRat(r) then
    Error("<r> has to be a rational number");
  fi;

  L := zerosT(n, r);
  return List(L, x -> List([0..n - 1], i -> SinPi((i + 1/2) * x)));
end;



# Polynomial DTTs
# ---------------

# polynomial DTTs are polynomial transforms for a suitable polynomial
# algebra C[x]/p and basis b; they arise from the unscaled DTTs above
# by scaling the rows such that the first entries are all equal to 1.

# For simplicity, we provide a function that converts an unscaled DTT (above)
# into a polynomial DTT

#F PolynomialDTT( <dtt> )
#F   scales the rows of <dtt> such that the first entries become 1.
#F   For example
#F     PolynomialDTT(DCT_IIunscaled(8))
#F   returns a polynomial DCT type II of size 8.
#F

PolynomialDTT := function ( dtt )
  local i;

  if not IsMat(dtt) then
    Error("<dtt> must be a matrix");
  fi;

  for i in [1..Length(dtt)] do
    dtt[i] := dtt[i]/dtt[i][1];
  od;

  return dtt;
end;


# Scaled DCTs
# -----------

#F DiscreteCosineTransformI( <N> )
#F InverseDiscreteCosineTransformI( <N> )
#F   constructs the Type-I Discrete Cosine Transform on N
#F   points or its inverse. (Mertins, 2.4.4)
#F

DiscreteCosineTransformI := function ( N )
  local M, i;

  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DiscreteCosineTransformI( <positive-int> )");
  fi;
  M := List([0..N-1], i -> List([0..N-1], j -> Sqrt(2/(N-1)) * CosPi(i*j/(N-1))));
  M[1] := 1/Sqrt(2) * M[1];
  M[N] := 1/Sqrt(2) * M[N];
  for i in [1..N] do
    M[i][1] := 1/Sqrt(2) * M[i][1];
    M[i][N] := 1/Sqrt(2) * M[i][N];
  od;

  return M;
end;

InverseDiscreteCosineTransformI := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: InverseDiscreteCosineTransformI( <positive-int> )");
  fi;
  return TransposedMat( DiscreteCosineTransformI( N ) );
end;


#F DiscreteCosineTransform( <N> )
#F InverseDiscreteCosineTransform( <N> )
#F   constructs the standard Discrete Cosine Transform (type-II) 
#F   on N points or its inverse. (Malvar, 1.2.5)
#F

DiscreteCosineTransform := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DiscreteCosineTransform( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1], function (j)
      if i = 0 then
        return 1/Sqrt(2) * Sqrt(2/N) * CosPi((j + 1/2)*i/N);
      else
        return 1         * Sqrt(2/N) * CosPi((j + 1/2)*i/N);
      fi;
    end ) ); 
end;

InverseDiscreteCosineTransform := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: InverseDiscreteCosineTransform( <positive-int> )");
  fi;
  return TransposedMat( DiscreteCosineTransform( N ) );
end;


#F DiscreteCosineTransformIV( <N> )
#F InverseDiscreteCosineTransformIV( <N> )
#F   constructs the Type-IV Discrete Cosine Transform on N points
#F   or its inverse. (Malvar, 1.2.6)
#F

DiscreteCosineTransformIV := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DiscreteCosineTransformIV( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1], j ->
      Sqrt(2/N) * CosPi((i + 1/2)*(j + 1/2)/N)
    ) );
end;

InverseDiscreteCosineTransformIV := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: InverseDiscreteCosineTransformIV( <positive-int> )");
  fi;
  return TransposedMat( DiscreteCosineTransformIV( N ) );
end;


# Scaled DSTs
# -----------

# scaled DTTs are orthonormal

#F DiscreteSineTransformI( <N> )
#F   constructs the Type-I Discrete Sine Transform on N points 
#F   which is equal to its inverse. 
#F

DiscreteSineTransformI := function ( N )
  local M, i;

  if not ( IsInt(N) and N > 1 ) then
    Error("usage: DiscreteSineTransformI( <int > 1> )");
  fi;
  return List([0..N-1], i -> List([0..N-1], j -> Sqrt(2/(N+1)) * SinPi((i+1)*(j+1)/(N+1))));
end;


#F DiscreteSineTransform( <N> )
#F   constructs the Discrete Sine Transform (type-II) 
#F   on N points. 
#F

DiscreteSineTransform := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DiscreteSineTransform( <positive-int> )");
  fi;
  return
    List([1..N], i -> List([1..N], function (j)
      if i = N then
        return 1/Sqrt(2) * Sqrt(2/N) * SinPi((j - 1/2)*i/N);
      else
        return 1         * Sqrt(2/N) * SinPi((j - 1/2)*i/N);
      fi;
    end ) ); 
end;

InverseDiscreteSineTransform := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: InverseDiscreteSineTransform( <positive-int> )");
  fi;
  return TransposedMat( DiscreteSineTransform( N ) );
end;

#F DiscreteSineTransformIV( <N> )
#F   constructs the Type-IV Discrete Sine Transform on N points
#F   or its inverse. (Malvar, 1.2.6)
#F

DiscreteSineTransformIV := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: DiscreteSineTransformIV( <positive-int> )");
  fi;
  return
    List([0..N-1], i -> List([0..N-1], j ->
      Sqrt(2/N) * SinPi((i + 1/2)*(j + 1/2)/N)
    ) );
end;


# Cos-DFT and Sin-DFT
# -------------------

#F CosDFT( <N> )
#F   returns the Cos-DFT of size <N>, which is the real part of the
#F   DFT of size <N>.
#F

CosDFT := function ( N )
  if not ( IsInt(N) and N > 0 ) then
    Error("<N> must be >= 1");
  fi;

  return
    List([0..N-1], i -> List([0..N-1], j -> CosPi(2*i*j/N)));
end;

#F SinDFT( <N> )
#F   returns the CosSinDFT of size <N>, which is the imaginary part of the
#F   DFT of size <N>.
#F

SinDFT := function ( N )
  if not ( IsInt(N) and N > 0 ) then
    Error("<N> must be >= 1");
  fi;

  return
    List([0..N-1], i -> List([0..N-1], j -> SinPi(2*i*j/N)));
end;

# Modified Cosine Transform (MDCT)
# ================================

#F IMDCTunscaled( <N> )
#F   returns the unscaled inverse MDCT of size 2N x N.
#F

IMDCTunscaled := function ( n )
  if not IsInt(n) then
    Error("<n> must be an integer");
  fi;

  return
    List(
      [0..2*n - 1],
      i ->
        List([0..n - 1], j -> CosPi((2*i + 1 + n)*(2*j + 1)/(4*n)))
    );
end;



#F ModifiedCosineTransform( <N> )
#F InverseModifiedCosineTransform( <N> )
#F   constructs the MDCT, which is an n x 2n matrix defined by
#F      [ cos( (2i+1)(2j+1+n)/(4n) ) | i = 0..n-1, j = 0..2n-1 ]
#F   or the inverse MDCT (which in fact is the transpose up to a scalar).
#F

ModifiedCosineTransform := function ( n )
  if not IsInt(n) then
    Error("<n> must be an integer");
  fi;

  return
    List(
      [0..n - 1],
      i ->
        List([0..2*n - 1], j -> CosPi((2*j + 1 + n)*(2*i + 1)/(4*n)))
    );
end;

InverseModifiedCosineTransform := function ( n )
  if not IsInt(n) then
    Error("<n> must be an integer");
  fi;

  return 1/n * IMDCTunscaled(n);
end;


# Walsh-Hadamard Transformation (WHT)
# ===================================

#F WalshHadamardTransform( <N> )
#F InverseWalshHadamardTransform( <N> )
#F   constructs the Walsh-Hadamard Transform on N points
#F   or its inverse. For N being a power of 2 this is
#F   DFT(2)^(tensor n). (Clausen, Baum, 1.7; Elliott, Rao, 8.3) 
#F   For general N the transform is defined to be a tensor product 
#F   of Fourier transforms of prime sizes (sorted by size).
#F

WalshHadamardTransform := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: WalshHadamardTransform( <positive-int> )");
  fi;
  if N = 1 then
    return [[1]];
  fi;
  return # [3], Kap. 1.7, S. 25., generalized
    TensorProductMat( 
      List(Factors(N), DiscreteFourierTransform) 
    );
end;

InverseWalshHadamardTransform := function ( N )
  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: InverseWalshHadamardTransform( <positive-int> )");
  fi;
  if N = 1 then
    return [[1]];
  fi;
  return 
    TensorProductMat( 
      List(Factors(N), InverseDiscreteFourierTransform) 
    );
end;


# Slant Transform (ST)
# ====================

#F SlantTransform( <N> )
#F InverseSlantTransform( <N> )
#F   constructs the Slant Transform of sidelength N or its 
#F   inverse. N must be a power of 2. (Elliott, Rao, 10.9)
#F

if not IsBound(SlantTransformTable) then
  SlantTransformTable := [

    # L = 1 from [2], (10.43)
    1/Sqrt(2) * [ 
      [ 1,  1 ],
      [ 1, -1 ]
    ],
    
    # L = 2 from [2], (10.45)
    1/2 * [       
      [ 1,  1,  1,  1 ],
      [ 3,  1, -1, -3 ] * 1/Sqrt(5),
      [ 1, -1, -1,  1 ],
      [ 1, -3,  3, -1 ] * 1/Sqrt(5)
    ]

  ];
fi;

SlantTransform := function ( N )
  local L,          # N = 2^L 
        I,          # identity matrix of size N/2-2
        aSqr, bSqr, # aSqr[k] = a_{2^k}^{2} as in [2]; b analogue
        aN, bN,     # a_N, b_N as in [2]
        T,          # composite factor 
        T11, T12,   # parts of T
        T21, T22,
        S,          # slant transform of sidelength 2^(L-1)
        k;          # counter

  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: SlantTransform( <positive-int> )");
  fi;
  L := LogInt(N, 2);

  # already computed before?
  if L = 0 then
    return [[1]];
  fi;
  if IsBound(SlantTransformTable[L]) then
    return List(SlantTransformTable[L], ShallowCopy);
  fi;

  # general case of sidelength N = 2^L >= 8, [2], (10.47)
  I := IdentityMat(N/2 - 2);

  aSqr := [ 1 ];
  bSqr := [ 1 ];
  for k in [2..L] do
    bSqr[k] := 1/(1 + 4*aSqr[k-1]);
    aSqr[k] := 4*bSqr[k]*aSqr[k-1];
  od;
  aN  := Sqrt( aSqr[L] );
  bN  := Sqrt( bSqr[L] );

  T11 := DirectSumMat([[ 1,  0 ], [  aN, bN ]],  I);
  T12 := DirectSumMat([[ 1,  0 ], [ -aN, bN ]],  I);
  T21 := DirectSumMat([[ 0,  1 ], [ -bN, aN ]],  I);
  T22 := DirectSumMat([[ 0, -1 ], [  bN, aN ]], -I);
  T   :=
    TensorProductMat([[1,0],[0,0]], T11) +
    TensorProductMat([[0,1],[0,0]], T12) +
    TensorProductMat([[0,0],[1,0]], T21) +
    TensorProductMat([[0,0],[0,1]], T22);

  S := SlantTransform(2^(L-1));

  SlantTransformTable[L] := 
    1/Sqrt(2) * T * DirectSumMat(S, S);

  return List(SlantTransformTable[L], ShallowCopy);
end;

InverseSlantTransform := function ( N )
  return TransposedMat( SlantTransform( N ) );
end;


# Haar Transform (HT)
# ===================

#F HaarTransform( <N> )
#F InverseHaarTransform( <N> )
#F   constructs the Haar Transform on N points or its 
#F   inverse. N must be a power of 2. (Elliott, Rao, 10.10)
#F

if true or not IsBound(HaarTransformTable) then
  HaarTransformTable := [

    # Ha(1) from [2], (10.53)
    [ [ 1,  1 ],
      [ 1, -1 ] 
    ],

    # Ha(2) from [2], (10.53)
    [ [ 1,  1,  1,  1 ],
      [ 1,  1, -1, -1 ],
      [ 1, -1,  0,  0 ] * Sqrt(2),
      [ 0,  0,  1, -1 ] * Sqrt(2) 
    ],

    # Ha(3) from [2], (10.53)
    [ [ 1,  1,  1,  1,  1,  1,  1,  1 ],
      [ 1,  1,  1,  1, -1, -1, -1, -1 ],
      [ 1,  1, -1, -1,  0,  0,  0,  0 ] * Sqrt(2),
      [ 0,  0,  0,  0,  1,  1, -1, -1 ] * Sqrt(2),
      [ 2, -2,  0,  0,  0,  0,  0,  0 ],
      [ 0,  0,  2, -2,  0,  0,  0,  0 ],
      [ 0,  0,  0,  0,  2, -2,  0,  0 ],
      [ 0,  0,  0,  0,  0,  0,  2, -2 ]
    ]

  ];
fi;

HaarTransform := function ( N )
  local L;

  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: HaarTransform( <positive-int> )");
  fi;
  L := LogInt(N, 2);
  if not 2^L = N then
    Error("<N> must be a power of 2");
  fi;
  if L = 0 then
    return [[1]];
  fi;
  if IsBound(HaarTransformTable[L]) then
    return 1/N * List(HaarTransformTable[L], ShallowCopy);
  fi;

  HaarTransformTable[L] := # [2], (10.54)
    Concatenation(
      TensorProductMat(N/2 * HaarTransform(N/2),     [[1,  1]]),
      TensorProductMat(Sqrt(N/2) * IdentityMat(N/2), [[1, -1]])
    );

  return 1/N * List(HaarTransformTable[L], ShallowCopy);
end;

InverseHaarTransform := function ( N )
  local L;

  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: InverseHaarTransform( <positive-int> )");
  fi;
  L := LogInt(N, 2);
  if not 2^L = N then
    Error("<N> must be a power of 2");
  fi;
  return N * TransposedMat( HaarTransform( N ) );
end;


# Rationalized Haar Transform (RHT)
# =================================

#F RationalizedHaarTransform( <N> )
#F InverseRationalizedHaarTransform( <N> )
#F   constructs the Tationalized Haar Transform on N points or its
#F   inverse. N must be a power of 2. (Elliott, Rao, 10.11).
#F

if true or not IsBound(RationalizedHaarTransformTable) then
  RationalizedHaarTransformTable := [

    # Rh(1) from [2], (10.58)
    [ [ 1,  1 ],
      [ 1, -1 ] 
    ]

  ];
fi;

RationalizedHaarTransform := function ( N )
  local L;

  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: RationalizedHaarTransform( <positive-int> )");
  fi;
  L := LogInt(N, 2);
  if not 2^L = N then
    Error("<N> must be a power of 2");
  fi;
  if L = 0 then
    return [[1]];
  fi;
  if IsBound(RationalizedHaarTransformTable[L]) then
    return List(RationalizedHaarTransformTable[L], ShallowCopy);
  fi;

  RationalizedHaarTransformTable[L] := # from [2] (induction)
    Concatenation(
      TensorProductMat(RationalizedHaarTransform(N/2), [[1,  1]]),
      TensorProductMat(IdentityMat(N/2),               [[1, -1]])
    );

  return List(RationalizedHaarTransformTable[L], ShallowCopy);
end;

InverseRationalizedHaarTransform := function ( N )
  local L, R, k;

  if not ( IsInt(N) and N >= 1 ) then
    Error("usage: InverseRationalizedHaarTransform( <positive-int> )");
  fi;
  L := LogInt(N, 2);
  if not 2^L = N then
    Error("<N> must be a power of 2");
  fi;

  R := RationalizedHaarTransform( N );
  for k in [1..N] do
    R[k] := R[k] / Sum(List(R[k], x -> x^2));
  od;
  return TransposedMat( R );
end;


# Abbreviations
# =============

#F Abbreviations
#F -------------
#F
#F   DFT       := DiscreteFourierTransform; 
#F   InvDFT    := InverseDiscreteFourierTransform;
#F   DHT       := DiscreteHartleyTransform;
#F   InvDHT    := InverseDiscreteHartleyTransform;
#F   DCT       := DiscreteCosineTransform;
#F   InvDCT    := InverseDiscreteCosineTransform;
#F   DCT_IV    := DiscreteCosineTransformIV;
#F   InvDCT_IV := InverseDiscreteCosineTransformIV;
#F   DCT_I     := DiscreteCosineTransformI;
#F   InvDCT_I  := InverseDiscreteCosineTransformI;
#F   WHT       := WalshHadamardTransform;
#F   InvWHT    := InverseWalshHadamardTransform;
#F   ST        := SlantTransform;
#F   InvST     := InverseSlantTransform;
#F   HT        := HaarTransform;
#F   InvHT     := InverseHaarTransform;
#F   RHT       := RationalizedHaarTransform;
#F   InvRHT    := InverseRationalizedHaarTransform;
#F

DFT       := DiscreteFourierTransform; 
InvDFT    := InverseDiscreteFourierTransform;
DHT       := DiscreteHartleyTransform;
InvDHT    := InverseDiscreteHartleyTransform;
DCT       := DiscreteCosineTransform;
DCT_II    := DiscreteCosineTransform;
DCT_III   := InverseDiscreteCosineTransform;
InvDCT    := InverseDiscreteCosineTransform;
DCT_IV    := DiscreteCosineTransformIV;
InvDCT_IV := InverseDiscreteCosineTransformIV;
DCT_I     := DiscreteCosineTransformI;
InvDCT_I  := InverseDiscreteCosineTransformI;
DST       := DiscreteSineTransform;
DST_II    := DiscreteSineTransform;
DST_III   := InverseDiscreteSineTransform;
DST_IV    := DiscreteSineTransformIV;
DST_I     := DiscreteSineTransformI;
MDCT      := ModifiedCosineTransform;
IMDCT     := InverseModifiedCosineTransform;
WHT       := WalshHadamardTransform;
InvWHT    := InverseWalshHadamardTransform;
ST        := SlantTransform;
InvST     := InverseSlantTransform;
HT        := HaarTransform;
InvHT     := InverseHaarTransform;
RHT       := RationalizedHaarTransform;
InvRHT    := InverseRationalizedHaarTransform;

# weitere:
#   andere Hadamard
#   Fermat
#   Mersenne
#   Rader
