/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include <math.h>
#include <stdlib.h>

#ifndef COMPLEX_T
#define PI    3.14159265358979323846
#define COMPLEX_T complex_t

typedef struct { double r,i; } complex_t;

complex_t omega(int N, int k) { 
   complex_t w; 
   w.r = cos(2*PI*k/N); w.i = sin(2*PI*k/N); return w; 
}

#define MALLOC malloc

#endif

