/* Fast Fourier Transform (.h-file)
   ================================

   Sebastian Egner,
   in GNU-C v2.7.2
   2. April 1997

   The module 'fft.h/.cpp' implements a Fast Fourier Transform
   both for numerical input and over finite fields.
   The main features are
     * generic arithmetics over an arbitrary base field
     * efficient methods for non-power-of-two lengths
     * efficient methods even for large prime lengths
     * each of the implemented methods can be used individually
     * predefined composition methods make life easy
     * printer/parser for a human readable representation of the 
       method allow interactive experiments to improve speed

   Usage: Create an fft-object F with an instance creation function like
   fft_new_prettyGood(<length>) which is a pretty-good composition
   method depending on the length. Then apply F to a number of signal
   vectors x by calling fft_apply(). Finally, deallocate the fft-object
   F by calling fft_delete(). Note that there are no static variables in
   the code, so the module is reentrant. However, an fft-object F contains
   internal buffers which cannot be used concurrently. Hence, every thread
   must create its own fft-object to work with.

   The application functions fft_apply() and fft_apply_inverse() do not
   simply apply the fft-object F of length N to a signal of length N.
   In fact, they apply the tensor product (1_r (x) F (x) 1_s) to a signal
   x of length r*N*s. Put in simple words, this means they perform r*s many
   FFTs of length N simultaneously. Namely, for all ir in [0..r-1] and all
   is in [0..s-1] the vectors

     [ x[ir*N*s + 0*s + is], .., x[ir*N*s + (N-1)*s + is] ]

   of length N are transformed in place by F. This slight extension of the
   application functions has a number of positive effects on the efficiency
   of the generated code (mostly due to better memory locality and the
   possibility to rearrange loops for strength reduction).

   reference
     [1] H. J. Nussbaumer: 
         Fast Fourier Transform and Convolution Algorithms.
         Springer, Berlin, 1982.
     [2] W. H. Press, et. al.: Numerical Recipes in C. 2nd ed.
         Cambridge, New York, 1992.
     [3] M. Clausen, U. Baum: Fast Fourier Transforms.
         BI-Verlag, Mannheim, 1993.
     [4] Th. Beth, V. Hatz: Computer Algebra.
         Skriptum zur Vorlesung, Univ. Karlsruhe, 1988.
     [5] Th. Beth: Methoden der Schnellen Fourier-Transformation.
         Teubner, Stuttgart, 1984.
     [6] D. Elliott, K. R. Rao: Fast Transforms.
         Academic Press, Orlando, 1982.
     [7] S. Egner: Zur algorithmischen Zerlegungstheorie 
         linearer Systeme mit Symmetrie. Diss. Informatik,
         Univ. Karlsruhe, Juli 1997.
*/

#if !defined(FFT_H)
#define FFT_H

/* Base Field Arithmethics (user adjustable)
   =========================================

   The base field is entirely generic. All operations with
   scalars are done with the following defined symbols

     fft_char                       the characteristics of the base field
     fft_numerical                  is the arithmetics numerical?
     fft_setNumerical(x, re, im)    explicit assign; only for fft_numerical
     fft_value                      a type to be used in declarations
     fft_valuesPerScalar            how many (fft_value)-objects make a
                                    scalar; see comment below

     (int) fft_rootOfUnity(x, n, k) assigns x = w^k for a fixed primitive
                                    n-th root of unity w and returns if
                                    this has been successful
     (int) fft_rational(x, k, n)    assigns x = k/n and returns if this
                                    has been successful; for numerical ffts
                                            this k and n may be any reals

     fft_set(x, y)                  (*x) = (*y);
     fft_add(x, y, z)               (*x) = (*y) + (*z);
     fft_sub(x, y, z)               (*x) = (*y) - (*z);
     fft_mul(x, y, z)               (*x) = (*y) * (*z);

   where

     (int) n, k             : integers where 0 <= k < n
     (fft_value *) x, y, z  : pointer to scalars.
     (double) re, im        : explicit numerical values

   Comments
     * A scalar is declared by fft_value scalar[fft_valuesPerScalar],
       a vector of scalars by fft_value *vector. These definitions
       allow packed data representations down to one scalar/byte.
       In particular the calling convention of 'Numerical Recipes'
       is supported this way (value = float, values/scalar = 2).
     * The function fft_sub() is the same as fft_add() if the base
       field is of characteristic 2 (where 1+1 = 0).
     * If an FFT method is not applicable since the corresponding
       roots of unity or the scaling factors 1/N do not exist then
       an error message is issued.
     * The symmetric scaling convention can only be used in
       characteristic char = 0 where 1/Sqrt[N] exists.

   For convenience and testing purposes we supply three sample
   implementations of a suitable base field:
     fft_struct_scalars   Two double in one struct { .. }
     fft_array_scalars    Two floats in adjacent entries of an array
     fft_GF256_scalars    Finite field GF(2^8), logarithmic representation
*/

#define fft_array_scalars

/* Numerical FFT with a (struct ..)-type for complex numbers
   ---------------------------------------------------------

   fft_value is a C-struct which contains two fields for the
   real and imaginary part of a numerical complex number.
*/

#if defined(fft_struct_scalars)

#define fft_char 0

typedef struct { double re, im; } fft_value;
#define                           fft_valuesPerScalar 1

#define fft_numerical 1
#define fft_setNumerical(x, re, im) \
  ( (x).re = (double)(re), (x).im = (double)(im) )

#define fft_rootOfUnity(x, n, k) \
  ((x)->re = cos(2.0*Pi*k/n), (x)->im = sin(2.0*Pi*k/n), 1)

#define fft_rational(x, k, n) \
  ((x)->re = (double)(k) / (double)(n), (x)->im = 0.0, 1)

#define fft_set(x, y) \
  { (x)->re = (y)->re; (x)->im = (y)->im; }

#define fft_add(x, y, z) \
  { (x)->re = (y)->re + (z)->re; (x)->im = (y)->im + (z)->im; }

#define fft_sub(x, y, z) \
  { (x)->re = (y)->re - (z)->re; (x)->im = (y)->im - (z)->im; }

#define fft_mul(x, y, z) \
  { double _r = (y)->re * (z)->re - (y)->im * (z)->im, \
           _i = (y)->re * (z)->im + (y)->im * (z)->re; \
    (x)->re = _r; (x)->im = _i; }

#endif /* defined(fft_struct_scalars) */

/* Numerical FFT with two reals per complex number
   -----------------------------------------------

   fft_value is a numerical real, two adjacent fft_values
   are the real and imaginary part of a numerical complex
*/

#if defined(fft_array_scalars)

#define fft_char 0

typedef double fft_value; /* float->double by Yevgen S. Voronenko */
#define       fft_valuesPerScalar 2

#define fft_numerical 1
#define fft_setNumerical(x, re, im) \
  ( (x)[0] = (fft_value)(re), (x)[1] = (fft_value)(im))

#define fft_rootOfUnity(x, n, k) \
  ( (x)[0] = (fft_value)cos(2.0*Pi*k/n), \
    (x)[1] = (fft_value)sin(2.0*Pi*k/n), 1)

#define fft_rational(x, k, n) \
  ( (x)[0] = (fft_value)(k) / (fft_value)(n), \
    (x)[1] = (fft_value) 0.0, 1)

#define fft_set(x, y) \
  { (x)[0] = (y)[0]; (x)[1] = (y)[1]; }

#define fft_add(x, y, z) \
  { (x)[0] = (y)[0] + (z)[0]; (x)[1] = (y)[1] + (z)[1]; }

#define fft_sub(x, y, z) \
  { (x)[0] = (y)[0] - (z)[0]; (x)[1] = (y)[1] - (z)[1]; }

#define fft_mul(x, y, z) \
  { fft_value _r = (y)[0] * (z)[0] - (y)[1] * (z)[1], \
              _i = (y)[0] * (z)[1] + (y)[1] * (z)[0]; \
    (x)[0] = _r; (x)[1] = _i; }

#endif /* defined(fft_array_scalars) */

/* Small Finite Fields
   -------------------

   Consider the finite Field GF(q) = < Z(q) > union { 0*Z(q) } where
   q = p^e for a prime p and e >= 1. We encode the elements of GF(q)
   by their Z(q)-logarithm as

     f: [0..q-1] --> GF(q)
        k        +-> Z(q)^k  for k <= q-2
        q-1      +-> 0*Z(q).

   Then multiplication reads Z(q)^k1 * Z(q)^k2 = Z^(k1 + k2) for
   k1, k2 <= q-2. Addition can be computed by using the identity

     Z(q)^k1 + Z(q)^k2 = Z(q)^k1 * (1 + Z(q)^(k2 - k1))

   for k1, k2 <= q-2. Hence, it suffices to table for the
   operation add1: x +-> x+1.

   The DFT(N) exists over GF(q) iff N divides q-1. Then
   w = Z(q)^((q-1)/N) is a primitive N-th root of unity.
   Since we also need k/n for integer k, n, we tabulate
   the natural embedding int: Z -> GF(q), 1 +-> Z(q)^0.

   Both tables (add1 and int) can easily be computed with
   a little program written in GAP v3.4:
[GAP-begin]

f := function (q, k)
  if k = q-1 then return 0*Z(q); else return Z(q)^k; fi;
end;

fInverse := function (q, x)
  if x = 0*x then return q-1; else return LogFFE(x, Z(q)); fi;
end;

int := function (q)
  return List([0..SmallestRootInt(q)-1], k -> fInverse(q, k*Z(q)^0));
end;

add1 := function (q)
  return List([0..q-1], k -> fInverse(q, f(q, k) + Z(q)^0));
end;

[GAP-end]
*/

#if defined(fft_GFsmall_scalars)

#define fft_numerical 0

typedef int fft_value;
#define     fft_valuesPerScalar 1

#define fft_size   256

#define fft_char   2 /* [GAP] SmallestRootInt(fft_size)  */
#define fft_degree 8 /* [GAP] LogInt(fft_size, fft_char) */
#define fft_units  (fft_size-1)
#define fft_zero   (fft_size-1)
#define fft_one    0

#define fft_rootOfUnity(x, n, k) \
  ( (((n) > fft_units) || (fft_units % (n) != 0)) ? 0 : \
    (*(x) = (((k) % fft_units)*(fft_units/(n))) % fft_units, 1) )

#define fft_rational(x, k, n) \
  ( ((n) % fft_char == 0) ? 0 : ( *(x) = ( ((k) % fft_char == 0) ? fft_zero : \
    (fft_int[(k)%fft_char]+fft_units-fft_int[(n)%fft_char])%fft_units ), 1))

#define fft_set(x, y) \
  { *(x) = *(y); }

#define fft_add(x, y, z) \
  { if (*(y) == fft_zero) { *(x) = *(z); } else \
    if (*(z) == fft_zero) { *(x) = *(y); } else { \
    fft_mul(x, y, &fft_add1[(*(z) + fft_units - *(y)) % fft_units]); } }

#define fft_sub(x, y, z) \
  { if (fft_char == 2) { fft_add(x, y, z); } else { \
    fft_rational(x, fft_char-1, 1); fft_mul(x, x, z); fft_add(x, x, y); } }

#define fft_mul(x, y, z) \
  { if ((*(y) == fft_zero) || (*(z) == fft_zero)) { *(x) = fft_zero; } \
    else *(x) = (*(y) + *(z)) % fft_units; }

static fft_value fft_int[fft_char] = { /* [GAP] int(fft_size) */

#if fft_char == 2
  fft_zero, fft_one
#endif

#if fft_size == 125
  124, 0, 31, 93, 62
#endif

};

static fft_value fft_add1[fft_size] = { /* [GAP] add1(fft_size) */

#if fft_size == 256
  255, 25, 50, 223, 100, 138, 191, 112, 200, 120, 21, 245, 127, 99, 224, 33, 
  145, 68, 240, 92, 42, 10, 235, 196, 254, 1, 198, 104, 193, 181, 66, 45, 35, 
  15, 136, 32, 225, 179, 184, 106, 84, 157, 20, 121, 215, 31, 137, 101, 253, 
  197, 2, 238, 141, 147, 208, 63, 131, 83, 107, 82, 132, 186, 90, 55, 70, 
  162, 30, 216, 17, 130, 64, 109, 195, 236, 103, 199, 113, 228, 212, 174, 
  168, 160, 59, 57, 40, 170, 242, 167, 175, 203, 62, 209, 19, 158, 202, 176,
  251, 190, 139, 13, 4, 47, 221, 74, 27, 248, 39, 58, 161, 71, 126, 246, 7, 
  76, 166, 243, 214, 122, 164, 153, 9, 43, 117, 183, 180, 194, 110, 12, 140, 
  239, 69, 56, 60, 250, 177, 144, 34, 46, 5, 98, 128, 52, 218, 150, 135, 16,
  217, 53, 206, 188, 143, 178, 226, 119, 201, 159, 169, 41, 93, 155, 81, 108, 
  65, 182, 118, 227, 114, 87, 80, 156, 85, 211, 229, 232, 79, 88, 95, 134, 
  151, 37, 124, 29, 163, 123, 38, 249, 61, 204, 149, 219, 97, 6, 247, 28, 
  125, 72, 23, 49, 26, 75, 8, 154, 94, 89, 187, 207, 148, 205, 54, 91, 241, 
  171, 78, 233, 116, 44, 67, 146, 142, 189, 252, 102, 237, 3, 14, 36, 152, 
  165, 77, 172, 231, 230, 173, 213, 244, 22, 73, 222, 51, 129, 18, 210, 86, 
  115, 234, 11, 111, 192, 105, 185, 133, 96, 220, 48, 24, 0 
#endif

#if fft_size == 125
  31, 96, 71, 10, 86, 108, 15, 60, 99, 64, 107, 122, 89, 92, 34, 50, 24, 49,
  105, 5, 58, 22, 4, 11, 66, 44, 3, 74, 104, 54, 75, 93, 26, 61, 63, 52, 6, 
  70, 116, 36, 123, 57, 90, 80, 40, 72, 67, 115, 43, 51, 39, 12, 117, 109, 
  100, 114, 30, 14, 18, 23, 73, 102, 124, 41, 13, 88, 84, 81, 98, 59, 46, 56,
  65, 85, 113, 2, 119, 68, 21, 27, 120, 37, 48, 16, 83, 121, 78, 33, 94, 17, 
  29, 28, 118, 62, 45, 25, 76, 47, 101, 19, 42, 112, 106, 1, 38, 110, 87, 32, 
  8, 35, 20, 79, 77, 111, 97, 55, 91, 53, 9, 103, 82, 7, 69, 95, 0
#endif

};

#endif /* defined(fft_GFsmall_scalars) */

/* Compiler and Operating System Interfacing
   =========================================

   All error messages are produced through the function fft_error()
   which gets a single string with the detailed information.
*/

/* the function to call with the error message */
#define fft_error(msg) \
  { fprintf(stderr, "fatal: %s. Exiting.\n", (msg)); \
    /* Commented out by Yevgen S. Voronenko: *((char *)0) = 0;  crash */ \
    exit(1); }

/* If the compiler complains about too large macro expansion
   text in the macro Apply_perm() then define fft_noLargeMacros
   to be 0 and recompile. This replaces the large macro 
   Apply_perm() by a C-function.
*/
#define fft_hasLargeMacros 1

/* Some compilers complain about unused parameters of functions
   although they are necessary sometimes. To suppress a the 
   variable name of an unused parameter switch set this to 0:
*/
#define fft_NamesForUnusedParameters 1

/* Some compilers/operating systems provide alloca() for 
   allocating space on the runtime stack. This is very 
   efficient and should be used if possible. Otherwise
   malloc()/free() are can be used.
*/

/*#include <alloca.h>*/
#define fft_StackAllocWithAlloca 0

/* Limitations and Derived Constants
   =================================

   Since the module does not use any arbitrary precision arithmetics
   a number of arithmetic restrictions are present. The restrictions
   are based on the assumption of 32 bits signed (int)s.
*/

/* maximal length of the signal vector of an FFT 
   (should be < 2^28 = 268435456 for 32-bit (int)s)
*/
#define fft_max_length 268435456

/* maximal length of a permuation */
#define fft_max_perm fft_max_length

/* the type of indices for permutations */
#if (fft_max_perm <= 32766)
typedef short int fft_perm; /* assuming 16 bits signed */
#else
typedef int       fft_perm; /* assuming 32 bits signed */
#endif

/* maximal length of the straight-line encoded FFTs */
#define fft_max_small 9

/* maximum nr. of prime-power factors of an integer;
   Note that 2*3*5*7*11 * 13*17*19*23*29 > 2^32.
*/
#define fft_max_factors 10

/* Individual Methods
   ==================

   The following individual methods for the FFT are available:

                length               restriction/comment

   null         N                    `fake' implementation
   direct       N                    by matrix multiplication    
   small        N                    straight-line implementation
   rader        N                    N must be prime
   goodThomas   N = q[0]*..*q[r-1]   gcd(q[a], q[b]) = 1 for a != b
   cooleyTukey  N = n*m              2 <= n, m arbitrary
   qPower       N = q^e              2 <= q, e arbitrary      
   bluestein    N <= M/2             needs two FFTs of length M >= 2 N

   To every indiviual method there is an instance creation function
   fft_new_<method>(..args..). The creation function takes as arguments
   either integer parameters or the smaller (for bluestein: larger) 
   fft-objects. If the resulting fft-object does cannot be created
   an error message is issued.
*/

/* symbolic constants for (fft_t *)F->type */
#define fft_null        0
#define fft_direct      1
#define fft_small       2
#define fft_rader       3
#define fft_goodThomas  4
#define fft_cooleyTukey 5
#define fft_qPower      6
#define fft_bluestein   7

/* datatype for the precomputed information of an FFT */
typedef struct fft_s {
  int       type,                                    /* FFT_<method> */
            N;                                  /* length of the FFT */
  fft_value inverseN[fft_valuesPerScalar];                    /* 1/N */
#if (fft_char == 0)
  fft_value inverseSqrtN[fft_valuesPerScalar];          /* 1/Sqrt[N] */
#endif
  void    (*apply)(int r, struct fft_s *F, int s, fft_value *x);
  void    (*deleteObj)(struct fft_s *F);  /* delete is a C++ keyword */
  union tagPriv {
    struct tagDirect {
      fft_value    *rootOfUnity;          /* all N-th roots of unity */
    } direct;
    struct tagSmall {
      fft_value    *c;                                     /* consts */
    } small;
    struct tagRader {
      int           q;                      /* generator of (Z/pZ)^x */
      fft_perm     *L,                                /* output-perm */
                   *SR;               /* reversing perm * input-perm */
      struct fft_s *fft_pMinus1;                         /* DFT(p-1) */
      fft_value    *diag;                         /* diagonal matrix */
    } rader;
    struct tagGoodThomas {
      int           nq, q[fft_max_factors]; /* N = q[0], .., q[nq-1] */
      fft_perm     *L, *R;             /* output-, input-permutation */
      struct fft_s *fft_q[fft_max_factors];         /* the DFT(q[a]) */
    } goodThomas;
    struct tagCooleyTukey {
      int           n, m;                                 /* N = n m */
      fft_value    *rootOfUnity;          /* all N-th roots of unity */
      fft_perm     *L;                         /* output-permutation */
      struct fft_s *fft_n,                                 /* DFT(n) */
                   *fft_m;                                 /* DFT(m) */
    } cooleyTukey;
    struct {
      int           q, e;                                 /* N = q^e */
      fft_value    *rootOfUnity;          /* all N-th roots of unity */
      fft_perm     *R;                          /* input-permutation */
      struct fft_s *fft_q;                                 /* DFT(q) */
    } qPower;
    struct tagBluestein {
      int           M;                                   /* M >= 2*N */
      fft_value    *diagN,         /* [ w[2 N]^(k^2) : k in [0..N) ] */
                   *diagM;       /* 1/M DFT(M) [ v_k : k in [0..M) ] */
      struct fft_s *fft_M;                                 /* DFT(M) */
    } bluestein;
  } priv;                                /* private is a C++ keyword */
} fft_t;

/* creation functions for the specific methods */
extern fft_t *fft_new_null(int N);
extern fft_t *fft_new_direct(int N);
extern fft_t *fft_new_small(int N);
extern fft_t *fft_new_rader(fft_t *fft_pMinus1);
extern fft_t *fft_new_goodThomas(int nq, fft_t *fft_q[]);
extern fft_t *fft_new_cooleyTukey(fft_t *fft_n, fft_t *fft_m);
extern fft_t *fft_new_qPower(fft_t *fft_q, int e);
extern fft_t *fft_new_bluestein(int N, fft_t *fft_M);

/* Generic Interface
   =================

   The evaluation of the FFT is done in two steps:
     1. construct an FFT as an object.
     2. apply the FFT-object to a signal vector.

   The first step is done with a call to fft_new_<method>()
   either for an individual method or with a call to a
   generic method. The generic methods combine the individual
   methods by different strategies. Two such strategies are 
   available at the moment:

   [simple] 
   Uses qPower[small[q], e] for N = q^e if possible and embeds 
   with blustein[N, q^e] if N is no power of a small q. This 
   strategy uses few methods but is not very fast. In addition
   it cannot be used over finite fields. The function uses
   direct[N] if the base field is of positive characteristic.

   [prettyGood]
   Uses small[q], qPower[small[q], e] and rader[p] if possible.
   Otherwise N is decomposed into prime power factors with
   goodThomas[p[0]^e[0], .., p[r-1]^e[r-1]] and then qPower[p, e]
   is used. This strategy should produce sufficiently good 
   results in most cases. Bluestein and cooleyTukey are never 
   used. The function resorts to direct[N] if the base field is
   not rich enough to contain the necessary roots of unity.

   The second step is a call to fft_apply_<mode>() which applies
   the FFT-object to a signal vector in some way. In fact, the
   FFT is tensored with unit matrices left and right in the sense
   of the comment at the beginning of this file. In addition,
   the normalization is important. Two common conventions are
   supported at the moment:

   [forward-without-factor]
   The forward DFT is only evaluates the signal z-transform at
   the roots of unity, no normalization factor is applied. This
   means that the inverse DFT needs to multiply by 1/N. This
   convention is implemented by fft_apply() and fft_apply_inverse().
   
   [symmetric]
   Both the forward and inverse DFT multiply by 1/Sqrt[N]. This
   convention is implemented by fft_applyS() and
   fft_applyS_inverse(). Note that it is only defined if the
   base field does contain Sqrt[N] != 0.

   In addition there are two functions to multiply with a 
   scalar and to apply the reversing permutation (2,N)(3,N-1).. 
   which is important in the definition of some convention
   for the inverse.
*/

/* creation functions for composed methods */
extern fft_t *fft_new_simple(int N);
extern fft_t *fft_new_prettyGood(int N);

/* deletion of an fft-object (generic function for all fft-objects) */
#define fft_delete(fft) \
  (fft->deleteObj)(fft)

/* multiply x = a*x */
extern void fft_apply_scalar(
  int r, fft_t *F, int s, fft_value *x, 
  fft_value *a
);

/* apply reversing permutation to x */
extern void fft_apply_reverse(
  int r, fft_t *F, int s, fft_value *x
);

/* [forward-without-factor]-application */
#define fft_apply(r, fft, s, x) \
  (fft->apply)(r, (struct fft_s *)(fft), s, x)

#define fft_apply_inverse(r, fft, s, x) \
  ( (fft->apply)(r, (struct fft_s *)(fft), s, x), \
    fft_apply_scalar(r, fft, s, x, fft->inverseN), \
    fft_apply_reverse(r, fft, s, x) )

/* [symmetric]-application */
#if (fft_char == 0)

#define fft_applyS(r, fft, s, x) \
  ( (fft->apply)(r, (struct fft_s *)(fft), s, x), \
    fft_apply_scalar(r, fft, s, x, fft->inverseSqrtN) )

#define fft_applyS_inverse(r, fft, s, x) \
  ( (fft->apply)(r, (struct fft_s *)(fft), s, x), \
    fft_apply_reverse(r, fft, s, x), \
    fft_apply_scalar(r, fft, s, x, fft->inverseSqrtN) )

#endif


/* Supporting Functions
   ====================

   A number of supporting functions make life easier when the
   user wants to experiment with the individual methods:
   fft-objects can be printed, constructed from printed form
   and it can be checked if a specific method is applicable. 
*/

/* (char *) fft_print(indent, F)
     returns a pointer to a newly allocated string which contains
     the printed representation of the tree of methods used for
     the fft F. The parameter indent controls the indentation.
     If indent < 0 then everything is printed as compact as possible.
     If indent >= 0 then comments are added and indentation to
     column (2 indent + 1) takes place at every newline. No initial
     or trailing newline or indentation is produced.
     The the output is of the following form

       <fft> ::= 
       null[ <int> ]
     | direct[ <int> ]
     | small[ <int> ]
     | rader[ <fft> ]
     | goodThomas[ <fft>, .., <fft> ]
     | cooleyTukey[ <fft>, <fft> ]
     | qPower[ <fft>, <int> ]
     | bluestein[ <int>, <fft> ]
     | unrecognized[{type -> <int>, N -> <int>}]
     | simple[ <int> ]
     | prettyGood[ <int> ]

     Comments are printed as "(* .. *)", they cannot be nested.
*/
extern char *fft_print(int indent, fft_t *F);

/* (fft_t *) fft_parse(char **msg, char *in)
     returns an fft-object constructed from parsing the printed
     representation of (char *)in. The syntax accepted is the
     same as produced by fft_print, except for unrecognized[..].
     If a syntactical or semantical error occurs then (*msg) is 
     assigned a newly allocated string containing an error message. 
     The error message starts with "[%d]" where %d denotes one of
     the following error classes
       [1] syntax error
       [2] base field does not contain the appropriate roots of
           unity w(N) or the appropriate rational 1/N
       [3] length of a transform is out of range
       [4] length is not prime
       [5] lengths are not relative prime
       [6] embedding length is insufficient
       [7] wrong arguments to C-function
       [8] unrecognized fft-method
*/
extern fft_t *fft_parse(char **msg, char *in);

/* (char *) fft_applicable(int method, int N, int nq, int q[])
     tests if the given method (fft_null, etc.) on length N
     with additional parameters q[0], .., q[nq-1] is applicable
     and if not produce an error message. The result is a newly
     allocated string containing detailed information. The message
     starts with "[%d]" where %d identifies the error class as 
     in fft_parse(). The method is applicable NULL is returned.
*/
extern char *fft_applicable(int method, int N, int nq, int q[]);

/* (void) fft_factor(x, n, p, e)
     factors the integer x in [1..2^30] such that
       x = p[0]^e[0] * .. p[n-1]^e[n-1]
     where p[i] are prime.
*/
extern void fft_factor(int x, int *n, int p[], int e[]);

#endif /* !defined(FFT_H) */

