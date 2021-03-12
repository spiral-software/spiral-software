/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#ifndef OMEGA_32I_H_INCLUDED
#define OMEGA_32I_H_INCLUDED

#include <math.h>
#include <stdlib.h>

#define PI    3.14159265358979323846f
#define SHIFT ((double)(2 << 30))

typedef struct { float r,i; } complex_t;

static complex_t omega(int N, int k) { 
   complex_t w; 
   w.r = cos(2*PI*k/N); w.i = sin(2*PI*k/N); return w; 
}

static short int cospi(float a) { return (short int)(cos(PI*a) * SHIFT); }

static short int sinpi(float a) { return (short int)(sin(PI*a) * SHIFT); }

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

#endif

