/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#ifndef __MACRO_FV4_H__
#define __MACRO_FV4_H__

#include "macro_cmn.h"


typedef struct {
    FLT d[4];
} FV4;


/* three-op */
#define ADD_FV4_FV4_FV4(y,a,b)    y = ADD_FV4_FV4(a,b)
#define SUB_FV4_FV4_FV4(y,a,b)    y = SUB_FV4_FV4(a,b)
#define MUL_FV4_FV4_FV4(y,a,b)    y = MUL_FV4_FV4(a,b)
#define NEG_FV4_FV4(y,a)          y = NEG_FV4(a)

#define FMA_FV4_FV4_FV4_FV4(y,a,b,c)     y = FMA_FV4_FV4_FV4(a,b,c)
#define FMS_FV4_FV4_FV4_FV4(y,a,b,c)     y = FMS_FV4_FV4_FV4(a,b,c)
#define NFMA_FV4_FV4_FV4_FV4(y,a,b,c)    y = NFMA_FV4_FV4_FV4(a,b,c)


#define VUNPKLO_FV4_FV4_FV4(y,a,b)       y = VUNPKLO_FV4_FV4(a,b)
#define VUNPKHI_FV4_FV4_FV4(y,a,b)       y = VUNPKHI_FV4_FV4(a,b)
#define VUNPKLO2_FV4_FV4_FV4(y,a,b)      y = VUNPKLO2_FV4_FV4(a,b)
#define VUNPKHI2_FV4_FV4_FV4(y,a,b)      y = VUNPKHI2_FV4_FV4(a,b)
#define VPKLO_FV4_FV4_FV4(y,a,b)         y = VPKLO_FV4_FV4(a,b)
#define VPKHI_FV4_FV4_FV4(y,a,b)         y = VPKHI_FV4_FV4(a,b)
#define VPKLO2_FV4_FV4_FV4(y,a,b)        y = VPKLO2_FV4_FV4(a,b)
#define VPKHI2_FV4_FV4_FV4(y,a,b)        y = VPKHI2_FV4_FV4(a,b)


#define NTH_FV4(loc, idx)         (((FV4*)(loc))[idx])
#define VLOAD_FV4(var, loc, idx)  (var) = (((FV4*)(loc))[idx])
#define VSTORE_FV4(loc, idx, exp) (((FV4*)(loc))[idx]) = (exp)


#define VDECLC_FV4(c, a0, a1, a2, a3)     const FV4 c = {{(a0), (a1), (a2), (a3)}}
#define C_FV4(a, b, c, d)         VPK_FV4((a), (b), (c), (d))
#define VDUP_FV4(a)               VPK_FV4((a), (a), (a), (a))


INLINE FV4 NEG_FV4(const FV4 a) {
    FV4 r;  r.d[0] = -a.d[0];  r.d[1] = -a.d[1];  r.d[2] = -a.d[2];  r.d[3] = -a.d[3];  return r;
}

INLINE FV4 VPK_FV4(const FLT a, const FLT b, const FLT c, const FLT d) {
    FV4 r;
    r.d[0] = a;
    r.d[1] = b;
    r.d[2] = c;
    r.d[3] = d;
    return r;
}

INLINE FV4 VUNPKLO_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r;  r.d[0] = a.d[0];  r.d[1] = b.d[0];  r.d[2] = a.d[1];  r.d[3] = b.d[1];  return r;
}

INLINE FV4 VUNPKHI_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r;  r.d[0] = a.d[2];  r.d[1] = b.d[2];  r.d[2] = a.d[3];  r.d[3] = b.d[3];  return r;
}

INLINE FV4 VUNPKLO2_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r;  r.d[0] = a.d[0];  r.d[1] = a.d[1];  r.d[2] = b.d[0];  r.d[3] = b.d[1];  return r;
}

INLINE FV4 VUNPKHI2_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r;  r.d[0] = a.d[2];  r.d[1] = a.d[3];  r.d[2] = b.d[2];  r.d[3] = b.d[3];  return r;
}

INLINE FV4 VPKLO_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r;  r.d[0] = a.d[0];  r.d[1] = a.d[2];  r.d[2] = b.d[0];  r.d[3] = b.d[2];  return r;
}

INLINE FV4 VPKHI_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r;  r.d[0] = a.d[1];  r.d[1] = a.d[3];  r.d[2] = b.d[1];  r.d[3] = b.d[3];  return r;
}

INLINE FV4 VPKLO2_FV4_FV4(const FV4 a, const FV4 b) {
    return VUNPKLO2_FV4_FV4(a,b);
}

INLINE FV4 VPKHI2_FV4_FV4(const FV4 a, const FV4 b) {
    return VUNPKHI2_FV4_FV4(a,b);
}

INLINE FV4 ADD_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r; int i;
    for (i=0; i<4; i++)
        r.d[i] = a.d[i] + b.d[i];
    return r;
}

INLINE FV4 SUB_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r; int i;
    for (i=0; i<4; i++)
        r.d[i] = a.d[i] - b.d[i];
    return r;
}

INLINE FV4 MUL_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r; int i;
    for (i=0; i<4; i++)
        r.d[i] = a.d[i] * b.d[i];
    return r;
}

INLINE FV4 FMA_FV4_FV4_FV4(const FV4 a, const FV4 b, const FV4 c) {
    return ADD_FV4_FV4(a, MUL_FV4_FV4(b,c));
}

INLINE FV4 FMS_FV4_FV4_FV4(const FV4 a, const FV4 b, const FV4 c) {
    return SUB_FV4_FV4(a, MUL_FV4_FV4(b,c));
}

INLINE FV4 NFMA_FV4_FV4_FV4(const FV4 a, const FV4 b, const FV4 c) {
    return SUB_FV4_FV4(MUL_FV4_FV4(b,c), a);
}

INLINE FV4 VSWAPCX_FV4_FV4(const FV4 a) {
    FV4 r;
    r.d[0] = a.d[1];
    r.d[1] = a.d[0];
    r.d[2] = a.d[3];
    r.d[3] = a.d[2];
    return r;
}

INLINE FV4 VMULCX_FV4_FV4(const FV4 a, const FV4 b) {
    FV4 r;
    r.d[0] = a.d[0]*b.d[0] - a.d[1]*b.d[1];
    r.d[1] = a.d[0]*b.d[1] + a.d[1]*b.d[0];
    r.d[2] = a.d[2]*b.d[2] - a.d[3]*b.d[3];
    r.d[3] = a.d[2]*b.d[3] + a.d[3]*b.d[2];
    return r;
}

#endif /*  __MACRO_FV4_H__ */
