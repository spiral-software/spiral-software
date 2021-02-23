/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include <ippdefs.h>
#include "sys.h"

# define SPIRAL_FIXED_FRACBITS atoi(getenv("spiral_fp"))
# define SPIRAL_FIXED_ZEROBITS atoi(getenv("spiral_zb"))
# define spiral_fixed_tofixed32(x)  ((((Ipp32s) ((x) * (double)(0x1 << (SPIRAL_FIXED_FRACBITS)))) >> SPIRAL_FIXED_ZEROBITS) << SPIRAL_FIXED_ZEROBITS);
# define spiral_fixed_tofixed16(x)  ((((Ipp16s) ((x) * (double)(0x1 << (SPIRAL_FIXED_FRACBITS)))) >> SPIRAL_FIXED_ZEROBITS) << SPIRAL_FIXED_ZEROBITS);
# define spiral_fixed_todouble(x) ( (double)(x) / (double)(1<<(SPIRAL_FIXED_FRACBITS)) )
#  define spiral_fixed_mul(x, y)  \
    ({ signed int __hi;  \
       unsigned int __lo;  \
       signed int __result;  \
       asm ("smull	%0, %1, %3, %4\n\t"  \
	    "movs	%0, %0, lsr %5\n\t"  \
	    "adc	%2, %0, %1, lsl %6"  \
	    : "=&r" (__lo), "=&r" (__hi), "=r" (__result)  \
	    : "%r" (x), "r" (y),  \
	      "r" (SPIRAL_FIXED_SCALEBITS), "r" (32 - SPIRAL_FIXED_SCALEBITS)  \
	    : "cc");  \
       __result;  \
    })
#  define spiral_fixed_scale64(hi, lo)  \
    ({ signed int __result;  \
       asm ("movs	%0, %1, lsr %3\n\t"  \
	    "adc	%0, %0, %2, lsl %4"  \
	    : "=r" (__result)  \
	    : "r" (lo), "r" (hi),  \
	      "r" (SPIRAL_FIXED_SCALEBITS), "r" (32 - SPIRAL_FIXED_SCALEBITS)  \
	    : "cc");  \
       __result;  \
    })
#  define SPIRAL_FIXED_SCALEBITS  SPIRAL_FIXED_FRACBITS

void*  Ipp16_cplx_alloc (int N) {
    return xaligned_malloc(sizeof(Ipp16sc) * N); 
}
void   Ipp16_cplx_free  (void *vec) {
    xaligned_free(vec); 
}
void*  Ipp16_cplx_nth   (void *vec, int n) {
    return (Ipp16sc*)vec + n; 
}
void   Ipp16_cplx_one   (void *x) { 
    ((Ipp16sc*)x)->re = spiral_fixed_tofixed16(1); ((Ipp16sc*)x)->im = 0; 
}
void   Ipp16_cplx_zero  (void *x) {
    ((Ipp16sc*)x)->re = 0; ((Ipp16sc*)x)->im = 0;
}
void   Ipp16_cplx_set   (void *x, void *a) { 
    ((Ipp16sc*)x)->re = ((Ipp16sc*)a)->re; 
    ((Ipp16sc*)x)->im = ((Ipp16sc*)a)->im; 
} 
void   Ipp16_cplx_set_fix   (void *x, void *a) {
    ((Ipp16sc*)x)->re = spiral_fixed_tofixed16(*((double*)a)); 
    ((Ipp16sc*)x)->im = spiral_fixed_tofixed16(*((double*)a+1));
} 
void   Ipp16_cplx_set_double   (void *x, void *a) { 
    *((double*)x) = spiral_fixed_todouble(((Ipp16sc*)a)->re);
    *((double*)x+1) = spiral_fixed_todouble(((Ipp16sc*)a)->im);
} 
void   Ipp16_cplx_random(void *x) { 
    double tmp0, tmp1;
    tmp0 = -5 + 10*((double)rand() / RAND_MAX); 
    ((Ipp16sc*)x)->re = spiral_fixed_tofixed16(tmp0);
    tmp1 = -5 + 10*((double)rand() / RAND_MAX);
    ((Ipp16sc*)x)->im = spiral_fixed_tofixed16(tmp1);
}
void   Ipp16_cplx_add (void *x, void *a, void *b) {
    ((Ipp16sc*)x)->re = ((Ipp16sc*)a)->re + ((Ipp16sc*)b)->re;
    ((Ipp16sc*)x)->im = ((Ipp16sc*)a)->im + ((Ipp16sc*)b)->im;
}
void   Ipp16_cplx_sub (void *x, void *a, void *b) {
    ((Ipp16sc*)x)->re = ((Ipp16sc*)a)->re - ((Ipp16sc*)b)->re;
    ((Ipp16sc*)x)->im = ((Ipp16sc*)a)->im - ((Ipp16sc*)b)->im;
}
void   Ipp16_cplx_mul (void *x, void *a, void *b) { 
    ((Ipp16sc*)x)->re = (((Ipp16sc*)a)->re * ((Ipp16sc*)b)->re)-(((Ipp16sc*)a)->im * ((Ipp16sc*)b)->im);
    ((Ipp16sc*)x)->im = (((Ipp16sc*)a)->re * ((Ipp16sc*)b)->im)+(((Ipp16sc*)a)->im * ((Ipp16sc*)b)->re);
}
int    Ipp16_cplx_rational (void *x, int k, int n) {
    if(n!=0) { 
	((Ipp16sc*)x)->re = spiral_fixed_tofixed16((Ipp16s)k / (Ipp16s)n); 
	((Ipp16sc*)x)->im = 0; 
	return 1; 
    }
    else return 0;
} 
int    Ipp16_cplx_div (void *x, void *a, void *b) { 
    if(((Ipp16sc*)b)->re!=0 || ((Ipp16sc*)b)->im!=0) { 
	((Ipp16sc*)x)->re = (((Ipp16sc*)a)->re * ((Ipp16sc*)b)->re) + (((Ipp16sc*)a)->im * ((Ipp16sc*)b)->im); 
	((Ipp16sc*)x)->im = (((Ipp16sc*)a)->im * ((Ipp16sc*)b)->re) - (((Ipp16sc*)a)->re * ((Ipp16sc*)b)->im);
	((Ipp16sc*)x)->re = ((Ipp16sc*)x)->re / ((((Ipp16sc*)b)->re + ((Ipp16sc*)b)->re) + (((Ipp16sc*)b)->im * ((Ipp16sc*)b)->im)); 
	((Ipp16sc*)x)->im = ((Ipp16sc*)x)->im / ((((Ipp16sc*)b)->re + ((Ipp16sc*)b)->re) + (((Ipp16sc*)b)->im * ((Ipp16sc*)b)->im));
	return 1; 
    }
    else return 0;
}
void   Ipp16_cplx_abs  (void *x, void *a) { 
    ((Ipp16sc*)x)->re = (Ipp16s) sqrt(spiral_fixed_mul(((Ipp16sc*)a)->re, ((Ipp16sc*)a)->re) + spiral_fixed_mul(((Ipp16sc*)a)->im, ((Ipp16sc*)a)->im));
}
int    Ipp16_cplx_less (void *a, void *b) {
    void *abs_a = Ipp16_cplx_alloc(1);
    void *abs_b = Ipp16_cplx_alloc(1); 
    int result; 
    Ipp16_cplx_abs(abs_a, a); 
    Ipp16_cplx_abs(abs_b, b); 
    result = ((((Ipp16sc*)abs_a)->re < ((Ipp16sc*)abs_b)->re));
    free(abs_a); 
    free(abs_b); 
    return result; 
}
void   Ipp16_cplx_fprint (FILE* f, void *a) {
    fprintf(f, "(%.9g, %.9g)\n", spiral_fixed_todouble(((Ipp16sc*)a)->re), spiral_fixed_todouble(((Ipp16sc*)a)->im));
}
void   Ipp16_cplx_fprintgap (FILE* f, void *a) {
    fprintf(f, "Complex(FloatString(\"%.9g\"), FloatString(\"%.9g\"))\n", 
	    spiral_fixed_todouble(((Ipp16sc*)a)->re), spiral_fixed_todouble(((Ipp16sc*)a)->im)); 
}

SCALAR_TYPE_DECL(Ipp16_cplx, 
		 "Ipp16_cplx", 
		 sizeof(Ipp16sc),
		 "-C -xfixed",
		 "#include <ippSP.h>\n\ntypedef Ipp16sc Ipp16_cplx;",
		 Ipp16_cplx_record);

void*  Ipp32_cplx_alloc (int N) { return xaligned_malloc(sizeof(Ipp32sc) * N); }
void   Ipp32_cplx_free  (void *vec) { xaligned_free(vec); }
void*  Ipp32_cplx_nth   (void *vec, int n) { return (Ipp32sc*)vec + n; }

void   Ipp32_cplx_one   (void *x) { 
    ((Ipp32sc*)x)->re = spiral_fixed_tofixed32(1); ((Ipp32sc*)x)->im = 0;
}
void   Ipp32_cplx_zero  (void *x) {
    ((Ipp32sc*)x)->re = 0; ((Ipp32sc*)x)->im = 0;
}
void   Ipp32_cplx_set   (void *x, void *a) { 
    ((Ipp32sc*)x)->re = ((Ipp32sc*)a)->re; 
    ((Ipp32sc*)x)->im = ((Ipp32sc*)a)->im;
} 
void   Ipp32_cplx_set_fix   (void *x, void *a) { 
    ((Ipp32sc*)x)->re = spiral_fixed_tofixed32(*((double*)a));
    ((Ipp32sc*)x)->im = spiral_fixed_tofixed32(*((double*)a+1));
}
void   Ipp32_cplx_set_double   (void *x, void *a) { 
    *((double*)x) = spiral_fixed_todouble(((Ipp32sc*)a)->re);
    *((double*)x+1) = spiral_fixed_todouble(((Ipp32sc*)a)->im);
}
void   Ipp32_cplx_random(void *x) { 
    double tmp0, tmp1;
    tmp0 = -50 + 100*((double)rand() / RAND_MAX); 
    ((Ipp32sc*)x)->re = spiral_fixed_tofixed32(tmp0);
    tmp1 = -50 + 100*((double)rand() / RAND_MAX);
    ((Ipp32sc*)x)->im = spiral_fixed_tofixed32(tmp1);
}
void   Ipp32_cplx_add (void *x, void *a, void *b) {
    ((Ipp32sc*)x)->re = ((Ipp32sc*)a)->re + ((Ipp32sc*)b)->re;
    ((Ipp32sc*)x)->im = ((Ipp32sc*)a)->im + ((Ipp32sc*)b)->im; }

void   Ipp32_cplx_sub (void *x, void *a, void *b) {
    ((Ipp32sc*)x)->re = ((Ipp32sc*)a)->re - ((Ipp32sc*)b)->re;
    ((Ipp32sc*)x)->im = ((Ipp32sc*)a)->im - ((Ipp32sc*)b)->im; }

void   Ipp32_cplx_mul (void *x, void *a, void *b) { 
    ((Ipp32sc*)x)->re = (((Ipp32sc*)a)->re * ((Ipp32sc*)b)->re)-(((Ipp32sc*)a)->im * ((Ipp32sc*)b)->im);
    ((Ipp32sc*)x)->im = (((Ipp32sc*)a)->re * ((Ipp32sc*)b)->im)+(((Ipp32sc*)a)->im * ((Ipp32sc*)b)->re); }

int    Ipp32_cplx_rational (void *x, int k, int n) {
    if(n!=0) { 
	((Ipp32sc*)x)->re = spiral_fixed_tofixed32((Ipp32s)k / (Ipp32s)n); 
	((Ipp32sc*)x)->im = 0; 
	return 1; 
    }
    else return 0; } 

int    Ipp32_cplx_div (void *x, void *a, void *b) { 
    if(((Ipp32sc*)b)->re!=0 || ((Ipp32sc*)b)->im!=0) { 
	((Ipp32sc*)x)->re = (((Ipp32sc*)a)->re * ((Ipp32sc*)b)->re) + (((Ipp32sc*)a)->im * ((Ipp32sc*)b)->im); 
	((Ipp32sc*)x)->im = (((Ipp32sc*)a)->im * ((Ipp32sc*)b)->re) - (((Ipp32sc*)a)->re * ((Ipp32sc*)b)->im);
	((Ipp32sc*)x)->re = ((Ipp32sc*)x)->re / ((((Ipp32sc*)b)->re + ((Ipp32sc*)b)->re) + (((Ipp32sc*)b)->im * ((Ipp32sc*)b)->im)); 
	((Ipp32sc*)x)->im = ((Ipp32sc*)x)->im / ((((Ipp32sc*)b)->re + ((Ipp32sc*)b)->re) + (((Ipp32sc*)b)->im * ((Ipp32sc*)b)->im));
	return 1; 
    }
    else return 0; }

void   Ipp32_cplx_abs  (void *x, void *a) { 
    ((Ipp32sc*)x)->re = (Ipp32s) sqrt(spiral_fixed_mul(((Ipp32sc*)a)->re, ((Ipp32sc*)a)->re) + spiral_fixed_mul(((Ipp32sc*)a)->im, ((Ipp32sc*)a)->im)); }

int    Ipp32_cplx_less (void *a, void *b) {
    void *abs_a = Ipp32_cplx_alloc(1);
    void *abs_b = Ipp32_cplx_alloc(1); 
    int result; 
    Ipp32_cplx_abs(abs_a, a); 
    Ipp32_cplx_abs(abs_b, b); 
    result = ((((Ipp32sc*)abs_a)->re < ((Ipp32sc*)abs_b)->re));
    free(abs_a); 
    free(abs_b); 
    return result; 
}

void   Ipp32_cplx_fprint (FILE* f, void *a) {
    fprintf(f, "(%.9g, %.9g)", spiral_fixed_todouble(((Ipp32sc*)a)->re), spiral_fixed_todouble(((Ipp32sc*)a)->im));
}

void   Ipp32_cplx_fprintgap (FILE* f, void *a) {
    fprintf(f, "Complex(FloatString(\"%.9g\"), FloatString(\"%.9g\"))", 
	    spiral_fixed_todouble(((Ipp32sc*)a)->re), spiral_fixed_todouble(((Ipp32sc*)a)->im)); 
}

SCALAR_TYPE_DECL(Ipp32_cplx, 
		 "Ipp32_cplx", 
		 sizeof(Ipp32sc),
		 "-C -xfixed",
		 "#include <ippSP.h>\n\ntypedef Ipp32sc Ipp32_cplx;",
		 Ipp32_cplx_record);
