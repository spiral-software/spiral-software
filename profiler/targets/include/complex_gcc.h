/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include <complex.h>
#include <math.h>

#ifndef _COMPLEX_GCC_H
#define _COMPLEX_GCC_H

#define CPX _Complex double
#define FLT double
#define INT int

#define C_I  I
#define C_NI (- C_I)
#define C_IM(a) (I * (a))
#define C_CPX(a,b) ((a) + C_I * (b))
#define C_CPXN(a,b) ((a) - C_I * (b))
#define C_INT(a) (a)
#define C_FLT(a) (a)

#define ADD_CPX(a,b) ((a) + (b))
#define SUB_CPX(a,b) ((a) - (b))
#define NEG_CPX(a)   (-(a))
#define NTH_CPX(v,i) ((v)[i])

#define ADD_FLT(a,b) ((a) + (b))
#define SUB_FLT(a,b) ((a) - (b))
#define NTH_FLT(v,i) ((v)[i])

#define ADD_INT(a,b) ((a) + (b))
#define SUB_INT(a,b) ((a) - (b))
#define NTH_INT(v,i) ((v)[i])
#define AND_INT(a,b) ((a) & (b))
#define XOR_INT(a,b) ((a) ^ (b))

#define MUL_I_CPX(a) (C_I * (a))
#define MUL_IM_CPX(a, b) ((a) * (b))
#define MUL_FLT_CPX(a, b) ((a) * (b))
#define MUL_CPX_CPX(a, b) ((a) * (b))
#define MUL_CPXN_CPX(a, b) ((a) * (b))

#define MUL_FLT_FLT(a, b) ((a) * (b))
#define MUL_INT_INT(a, b) ((a) * (b))

#define IDIV_INT(a, b) ((a)/(b))
#define IMOD_INT(a, b) ((a)%(b))

#define PI    3.14159265358979323846

#ifndef MALLOC
	#if defined(_WIN32) || defined(_WIN64)
	#define MALLOC(a) _mm_malloc(a, 16)
	#else
	#include <malloc.h>
	#define MALLOC(a) memalign(16, a)
	#endif
#endif // MALLOC


extern double cos(double); 
extern double sin(double); 

CPX omega(int N, int k) { 
   CPX w; 
   w = C_CPX(cos(2*PI*k/N), sin(2*PI*k/N)); 
   return w;
}

#endif // _COMPLEX_GCC_H
