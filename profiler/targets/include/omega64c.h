/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#ifndef OMEGA_64C_H_INCLUDED
#define OMEGA_64C_H_INCLUDED
#ifdef __cplusplus
#include <cmath>
#include <cstdlib>
#include <complex>
using namespace std;
#else
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#endif
#define PI    3.14159265358979323846

#ifdef __GNUC__
#ifdef __cplusplus
#define __I__ complex<double>(0,1)
#define complex_t complex<double>
#else
#define complex_t _Complex double
#define __I__ _Complex_I
#endif
#else
#define __I__ _Complex_I
#define complex_t _Complex double
#endif

static complex_t omega(int N, int k) { return cos(2*PI*k/N) + __I__ * sin(2*PI*k/N); }

static double cospi(double a) { return cos(PI*a); }

static double sinpi(double a) { return sin(PI*a); }

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

