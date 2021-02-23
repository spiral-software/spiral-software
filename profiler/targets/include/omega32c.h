/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#ifndef OMEGA_32C_H_INCLUDED
#include <math.h>
#include <stdlib.h>
#include <complex.h>

#define PI    3.14159265358979323846f

#ifdef __GNUC__
#define __I__ _Complex_I
#endif

static _Complex float omega(int N, int k) { return cos(2*PI*k/N) + __I__ * sin(2*PI*k/N); }

static float cospi(float a) { return cos(PI*a); }

static float sinpi(float a) { return sin(PI*a); }

static int powmod(int phi, int g, int exp, int N) {
    int retVal, i;
    retVal = phi % N;
    for (i=0; i<exp; i++) retVal = (retVal * g) % N;
    return(retVal);
}

#define RE_CPX(x) (creal(x))
#define IM_CPX(x) (cimag(x))

#if defined(_WIN32) || defined(_WIN64)
#define MALLOC(a) _mm_malloc(a, 16)
#else
#include <malloc.h>
#define MALLOC(a) memalign(16, a)
#endif

#define MAX_INT_INT(a,b) \
           ({ __typeof__ (a) _a = (a); \
              __typeof__ (b) _b = (b); \
              _a > _b ? _a : _b; })
#define MIN_INT_INT(a,b) \
           ({ __typeof__ (a) _a = (a); \
              __typeof__ (b) _b = (b); \
              _a < _b ? _a : _b; })

#define MAX_FLT_FLT   MAX_INT_INT
#define MIN_FLT_FLT   MIN_INT_INT

#endif

