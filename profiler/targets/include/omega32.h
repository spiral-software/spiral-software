/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#ifndef OMEGA_32_H_INCLUDED
#include <math.h>
#include <stdlib.h>

#define PI    3.14159265358979323846

typedef struct { float r,i; } complex_t;

static complex_t COMPLEX(float r, float i) { 
   complex_t cplx; 
   cplx.r = r; 
   cplx.i = i; 
   return cplx; 
}

static complex_t omega(int N, int k) { 
   complex_t w; 
   w.r = cos(2*PI*k/N); w.i = sin(2*PI*k/N); return w; 
}

static double cospi(double a) { return cos(PI*a); }

static double sinpi(double a) { return sin(PI*a); }

static int powmod(int phi, int g, int exp, int N) {
    int retVal, i;
    retVal = phi % N;
    for (i=0; i<exp; i++) retVal = (retVal * g) % N;
    return(retVal);
}

static double fr(int m, int i, double r) {
  if((i % 2)==0) 
    return (r + 2*((int)i/2)) / m;
  else 
    return (2 - r + 2*((int)i/2)) / m;
}

#define RE_CPX(x) ((x).r)
#define IM_CPX(x) ((x).i)
#define CPX complex_t

#define RE_FLT(a) (a)
#define IM_FLT(a) (0.0)
#define RE_INT(a) (a)
#define IM_INT(a) (0)
#define IMOD_INT(a,b)    ((a)%(b))
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

