/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include <xmmintrin.h>
#include <emmintrin.h>
    //#include <math.h>
#include <malloc.h>
    //#include <complex.h>   // complex.h can't be found in my Cygwin

#ifdef __cplusplus
#include <cmath>
#include <cstdlib>
#include <complex>
#define CPX complex<float>
#define __I__ complex<float>(0,1)
#define RE_CPX(x) x.real()
#define IM_CPX(x) x.imag()
#define C_CPX(a, b) CPX(a, b)
#define C_CPXN(a, b) CPX(a, -b)
#define MUL_CPX_CPX(a, b) (a*b)
#define MUL_CPXN_CPX(a, b) (a*b)

using namespace std;
#else
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#define _Complex float
//#define __I__
#endif

#if defined(__ICL) || defined(_MSC_VER)
#define MALLOC(n) _mm_malloc(n, 16)
#else
#define MALLOC(n) memalign(16, n)
#endif

#define complex_t CPX
#define LIB_iface_elt CPX
#define FC1 __m64
#define FC2 __m128
#define FV2 __m128

#define FLT float
#define INT int

typedef __m128* PTR_FC2;
typedef __m64*  PTR_CPX;
typedef __m128* PTR_FLT;

#define v_lo2(a, b) (FC2)(_mm_shuffle_pd(a, b, _MM_SHUFFLE2(0,0)))
#define v_hi2(a, b) (FC2)(_mm_shuffle_pd(a, b, _MM_SHUFFLE2(1,1)))
#define v_rev2(a) (FC2)(_mm_shuffle_pd(a, a, _MM_SHUFFLE2(0,1)))

#define v_revhi2(a, b) _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,3,2,3))
#define v_neg03(a)     _mm_xor_ps(a, _mm_set_ps(-0.0, -0.0 , 0.0,  0.0))
#define v_neg23(a)     _mm_xor_ps(a, _mm_set_ps(-0.0, -0.0,  0.0,  0.0))
#define v_neg01(a)     _mm_xor_ps(a, _mm_set_ps( 0.0,  0.0, -0.0, -0.0))
#define v_mul1j(a)    _mm_xor_ps(_mm_shuffle_ps(a,a, _MM_SHUFFLE(2,3,1,0)), _mm_set_ps(0.0, 0.0, -0.0, 0.0))

#define _mm_storel_pd(ptr, val)  _mm_storel_pi((__m64*)ptr, val)
#define _mm_storeh_pd(ptr, val)  _mm_storeh_pi((__m64*)ptr, val)

#define vload1sd_2x64f(ptr) ((__m128)_mm_load_sd((double*)ptr))
#define vload_1h_2x64f _mm_loadh_pi
#define vload_1l_2x64f _mm_loadl_pi
#define vstore_1h_2x64f _mm_storeh_pi
#define vstore_1l_2x64f _mm_storel_pi

#define vunpackhi_2x64f(a, b) ((__m128)_mm_unpackhi_pd((__m128d)a, (__m128d)b))
#define vunpacklo_2x64f(a, b) ((__m128)_mm_unpacklo_pd((__m128d)a, (__m128d)b))

inline FC2 vpack(CPX a, CPX b) {
    return _mm_set_ps(IM_CPX(b),RE_CPX(b),IM_CPX(a),RE_CPX(a));
}

#define LOAD_CPX(v,ptr) (v) = *(ptr);
#define LOAD_FC2(v,ptr) (v) = *(ptr);
#define LOAD_FLT(v,ptr) (v) = *(ptr);
#define NTH_CPX(v,i) (v)[i]

#define SAVE_CPX(ptr,v) *(ptr) = (v);
#define SAVE_FC2(ptr,v) *(ptr) = (v);
#define SAVE_FLT(ptr,v) *(ptr) = (v);

#define C_INT(a) (a)
#define C_IM(a) (__I__*((float)a))
#define C_FLT(a) (float)a
#define C_FV2(a,b) _mm_set_ps(b,b,a,a)

/* -- */
//#define C_FC2(a,b) _mm_set_ps(1,1,1,1)
//#define MUL_FC2_FC2(a, b) _mm_mul_ps(a, b)
/* -- */
#define MUL_FLT_CPX(a, b) ((a)*(b))
#define MUL_IM_CPX(a, b) ((a)*(b))
#define MUL_FV2_FC2(a, b) _mm_mul_ps(a, b)
#define MUL_FV2_FV2(a, b) _mm_mul_ps(a, b)
#define MUL_FLT_FC2(a, b) _mm_mul_ps(_mm_set1_ps(a), b)

#define ADD_FC2_FC2 _mm_add_ps
#define ADD_FC2_FV2 _mm_add_ps
#define ADD_FV2_FV2 _mm_add_ps
#define SUB_FC2_FC2 _mm_sub_ps

#define NEG_FC2(a)   _mm_xor_ps(a, _mm_set_ps(-0.0, -0.0, -0.0, -0.0))
#define NTH_FC2(v,i) ((v)[i])
#define NTH_FV2(v,i) ((v)[i])

#define ADD_CPX_CPX(a,b) ((a)+(b))
#define SUB_CPX_CPX(a,b) ((a)-(b))
#define NEG_CPX(a) (-(a))

#define ADD_FLT_FLT(a,b) ((a)+(b))
#define SUB_FLT_FLT(a,b) ((a)-(b))
#define DIV_FLT_FLT(a,b) ((a)/(b))
#define NTH_FLT(v,i) ((v)[i])
#define MUL_FLT_FLT(a, b) ((a)*(b))
#define MUL_FLT_INT(a, b) ((a)*(b))
#define MUL_INT_FLT(a, b) ((a)*(b))
#define NEG_FLT(a) (-a)

#define MUL_FLT_UNK MUL_FLT_FLT
#define MUL_UNK_FLT MUL_FLT_FLT
#define MUL_INT_UNK MUL_INT_FLT
#define MUL_UNK_INT MUL_FLT_INT

#define ADD_INT_FLT(a,b) ADD_FLT_FLT(C_FLT(a), b)
#define ADD_INT_UNK(a,b) ADD_FLT_FLT(C_FLT(a), b)
#define ADD_FLT_INT(a,b) ADD_FLT_FLT(a, C_FLT(b))
#define ADD_UNK_INT(a,b) ADD_FLT_FLT(a, C_FLT(b))

#define FDIV_INT_INT(a,b) C_FLT(((double)a) / (b))
#define FDIV_FLT_INT(a,b) DIV_FLT_FLT(a, C_FLT(b))
#define FDIV_INT_FLT(a,b) DIV_FLT_FLT(C_FLT(a), b)
#define FDIV_FLT_FLT(a,b) DIV_FLT_FLT(a, b)

#define MUL_INT_INT(a,b) ((a) * (b))
#define ADD_INT_INT(a,b) ((a) + (b))
#define SUB_INT_INT(a,b) ((a) - (b))
#define AND_INT_INT(a,b) ((a) & (b))
#define XOR_INT_INT(a,b) ((a) ^ (b))
#define DIV_INT_INT(a,b) ((a) / (b))
#define IDIV_INT(a, b) ((a)/(b))
#define IMOD_INT(a, b) ((a)%(b))
#define NEG_INT(a) (-(a))
#define NTH_INT(v,i) ((v)[i])

#define NTH_SYM(v,i) (v)[i]

/* FIXME:
 *   MUL_I_CPX
 *   MUL_NI_CPX
 *   NTH_CPX    deref/load/store
 */

/* MUL_I_CPX(x) : multiply complex x by I */
inline CPX MUL_I_CPX(CPX x) { return __I__ * x; }
inline CPX MUL_NI_CPX(CPX x) { return (-__I__ * x); }

inline __m128 MUL_NI_FC2(__m128 x) {
     __m128 t = x;
     t = _mm_xor_ps(t, _mm_set_ps(0.0, -0.0, 0.0, -0.0));
     return _mm_shuffle_ps(t,t,_MM_SHUFFLE(2,3,0,1));
}

inline __m128 MUL_I_FC2(__m128 x) {
     __m128 t = x;
     t = _mm_xor_ps(t, _mm_set_ps(-0.0, 0.0, -0.0, 0.0));
     return _mm_shuffle_ps(t,t,_MM_SHUFFLE(2,3,0,1));
}

#define MUL_CPXN_FC2 MUL_CPX_FC2
#define MUL_FC2_FV2 MUL_FC2_FC2

inline __m128 MUL_CPX_FC2(CPX x, __m128 y) {
         __m128 yy = y;
         __m128 re, im;                                        /* yy = [yim yre | yim yre] */

         re = _mm_set1_ps(x.real());                          /* re = [xre xre | xre xre] */
         re = _mm_mul_ps(re, yy);                              /* re = [xre*yim xre*yre | --"--] */

         im = _mm_set1_ps(x.imag());                          /* im = [xim xim | xim xim] */
         yy = _mm_shuffle_ps(yy,yy,_MM_SHUFFLE(2,3,0,1));      /* yy = [yre yim | yre yim] */

         im = _mm_mul_ps(im, yy);                              /* im = [xim*yre  xim*yim | --"--] */
         im = _mm_xor_ps(im, _mm_set_ps(0.0,-0.0,0.0,-0.0));   /* im = [xim*yre -xim*yim | --"--] */
         return _mm_add_ps(re, im);  /* res = [ xre*yim+xim*yre xre*yre-xim*yim | --"-- ] */
}

inline __m128 MUL_FC2_FC2(__m128 x, __m128 y) {
         __m128 xx = x; __m128 yy = y;
         __m128 re, im;                                        /* yy = [yim yre | yim yre] */

         re = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2, 2, 0, 0)); /* re = [xre xre | xre xre] */
         re = _mm_mul_ps(re, yy);                              /* re = [xre*yim xre*yre | --"--] */

         im = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(3, 3, 1, 1)); /* im = [xim xim | xim xim] */
         yy = _mm_shuffle_ps(yy,yy,_MM_SHUFFLE(2,3,0,1));      /* yy = [yre yim | yre yim] */

         im = _mm_mul_ps(im, yy);                              /* im = [xim*yre  xim*yim | --"--] */
         im = _mm_xor_ps(im, _mm_set_ps(0.0,-0.0,0.0,-0.0));   /* im = [xim*yre -xim*yim | --"--] */
         return _mm_add_ps(re, im);  /* res = [ xre*yim+xim*yre xre*yre-xim*yim | --"-- ] */
}

#define PI    3.14159265358979323846

extern float cosf(float);
extern float sinf(float);

inline FLT cospi(FLT a) { return C_FLT(cosf(PI*a)); }

inline FLT sinpi(FLT a) {  return C_FLT(sinf(PI*a)); }

inline CPX sp_omega(int n, FLT r) {
    return cospi(2*(double)r / (double)n) + __I__ * sinpi(2*(double)r / (double)n);
}
