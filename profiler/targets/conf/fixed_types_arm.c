/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include "sys.h"

# define SPIRAL_FIXED_FRACBITS atoi(getenv("spiral_fp"))
# define SPIRAL_FIXED_ZEROBITS atoi(getenv("spiral_zb"))
# define spiral_fixed_tofixed(x)  ((((signed int) ((x) * (double)(0x1 << (SPIRAL_FIXED_FRACBITS)))) >> SPIRAL_FIXED_ZEROBITS) << SPIRAL_FIXED_ZEROBITS);
# define spiral_fixed_todouble(x) ((double)(x) / (double)(1<<(SPIRAL_FIXED_FRACBITS)))
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

void* int_cplx_alloc (int num) { return xaligned_malloc(sizeof(int) * num * 2); }
void  int_cplx_free  (void *vec) { xaligned_free(vec); }
void* int_cplx_nth   (void *vec, int n)  { return (int*)vec + 2*n; }
void  int_cplx_one   (void *x) { *((int*)x) = spiral_fixed_tofixed(1); *((int*)x+1) = 0;}
void  int_cplx_zero  (void *x) { *((int*)x) = 0; *((int*)x+1) = 0;}
void  int_cplx_set   (void *x, void *a) { *((int*)x) = *((int*)a); *((int*)x+1) = *((int*)a+1); }
void  int_cplx_set_fix   (void *x, void *a) { *((int*)x) = spiral_fixed_tofixed(*((double*)a)); *((int*)x+1) = spiral_fixed_tofixed(*((double*)a+1)); }
void  int_cplx_set_double   (void *x, void *a) { *((double*)x) = spiral_fixed_todouble(*((int*)a)); *((double*)x+1) = spiral_fixed_todouble(*((int*)a+1)); }
void  int_cplx_random(void *x) { 
    double tmp0, tmp1;
    tmp0 = -50 + 100*((double)rand() / RAND_MAX); 
    *((int*)x) = spiral_fixed_tofixed(tmp0);
    tmp1 = -50 + 100*((double)rand() / RAND_MAX);
    *((int*)x+1) = spiral_fixed_tofixed(tmp1);
}
void  int_cplx_add   (void *x, void *a, void *b)  {
    *((int*)x)   = *((int*)a) + *((int*)b);
    *((int*)x+1) = *((int*)a+1) + *((int*)b+1); 
}
void  int_cplx_sub   (void *x, void *a, void *b)  { 
    *((int*)x) = *((int*)a) - *((int*)b);
    *((int*)x+1) = *((int*)a+1) - *((int*)b+1); 
}
void  int_cplx_mul   (void *x, void *a, void *b)  { 
    *((int*)x) = *((int*)a) * *((int*)b) - *((int*)a+1) * *((int*)b+1);
    *((int*)x+1) = *((int*)a) * *((int*)b+1) + *((int*)a+1) * *((int*)b); 
}
int   int_cplx_div   (void *x, void *a, void *b)  { 
    *((int*)x) = *((int*)a) * *((int*)b) + *((int*)a+1) * *((int*)b+1);
    *((int*)x+1) = *((int*)a+1) * *((int*)b) - *((int*)a)   * *((int*)b+1);
    *((int*)x)   = *((int*)x)   / (*((int*)b) * *((int*)b) + *((int*)b+1) * *((int*)b+1));
    *((int*)x+1) = *((int*)x+1) / (*((int*)b) * *((int*)b) + *((int*)b+1) * *((int*)b+1));
    return 1; 
}
int   int_cplx_rational(void *x, int k, int n)    { 
    if(n!=0) { *((int*)x) = spiral_fixed_tofixed((int)k / (int)n); *((int*)x+1) = 0; return 1; }
    else return 0;
}
void  int_cplx_abs   (void *x, void *a)  { 
    *((int*)x) = (int) sqrt(spiral_fixed_mul(*((int*)a), *((int*)a)) + spiral_fixed_mul(*((int*)a+1), *((int*)a+1)));
    *((int*)x+1) = 0; 
}
int   int_cplx_less  (void *a, void *b)  { 
    void *abs_a = int_cplx_alloc(1);
    void *abs_b = int_cplx_alloc(1);
    int result;
    int_cplx_abs(abs_a, a); 
    int_cplx_abs(abs_b, b);
    result = *((int*)abs_a) < *((int*)abs_b);
    free(abs_a); 
    free(abs_b);
    return result;
}
void  int_cplx_fprint(FILE *f, void *a) { fprintf(f, "(%.9g, %.9g)\n", spiral_fixed_todouble(*((int*)a)), spiral_fixed_todouble(*((int*)a+1))); }
void  int_cplx_fprintgap(FILE *f, void *a) { fprintf(f, "Complex(FloatString(\"%.9g\"), FloatString(\"%.9g\"))\n", spiral_fixed_todouble(*((int*)a)), spiral_fixed_todouble(*((int*)a+1))); }

SCALAR_TYPE_DECL(int_cplx, 
		 "int_cplx", 
		 sizeof(int),
		 "-C -xfixed",
		 "typedef int int_cplx;\n",
		 int_cplx_record);

void* int_alloc (int num) { return xaligned_malloc(sizeof(int) * num); }
void  int_free  (void *vec) { xaligned_free(vec); }
void* int_nth   (void *vec, int n)  { return (int*)vec + n; }
void  int_one   (void *x) { *((int*)x) = spiral_fixed_tofixed(1); }
void  int_zero  (void *x) { *((int*)x) = 0; }
void  int_set   (void *x, void *a) { *((int*)x) = *((int*)a); }
void  int_set_fix   (void *x, void *a) { *((int*)x) = spiral_fixed_tofixed(*((double*)a)); }
void  int_set_double   (void *x, void *a) { *((double*)x) = spiral_fixed_todouble(*((int*)a)); }
void  int_random(void *x) { 
    double tmp0;
    tmp0 = -50 + 100*((double)rand() / RAND_MAX); 
    *((int*)x) = spiral_fixed_tofixed(tmp0);
}
void  int_add   (void *x, void *a, void *b)  {
    *((int*)x)   = *((int*)a) + *((int*)b);
}
void  int_sub   (void *x, void *a, void *b)  { 
    *((int*)x) = *((int*)a) - *((int*)b);
}
void  int_mul   (void *x, void *a, void *b)  { 
    *((int*)x) = *((int*)a) * *((int*)b);
}
int   int_div   (void *x, void *a, void *b)  {
    if(*((int*)b)!=0) {
	*((int*)x) = *((int*)a) / *((int*)b);
	return 1;
    }
    else return 0;
}
int   int_rational(void *x, int k, int n)    { 
    if(n!=0) { 
	*((int*)x) = spiral_fixed_tofixed((int)k / (int)n); 
	return 1;
    }
    else return 0;
}
void  int_abs   (void *x, void *a)  { 
    if(*((int*)a) < 0)
	*((int*)x) = - *((int*)a);
    else
	*((int*)x) = *((int*)a);
}
int   int_less  (void *a, void *b)  { 
    return *((int*)a) < *((int*)b);
}
void  int_fprint(FILE *f, void *a) { fprintf(f, "%.9g\n", spiral_fixed_todouble(*((int*)a))); }
void  int_fprintgap(FILE *f, void *a) { fprintf(f, "FloatString(\"%.9g\")\n", spiral_fixed_todouble(*((int*)a))); }

SCALAR_TYPE_DECL(int, 
		 "int", 
		 sizeof(int),
		 "-R -xfixed",
		 "",
		 int_record);
