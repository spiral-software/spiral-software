/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#ifndef __MACRO_FV2_H__
#define __MACRO_FV2_H__

#include "macro_cmn.h"


typedef struct {
    FLT d[2];
} FV2;


/* three-op */
#define ADD_FV2_FV2_FV2(y,a,b)    y = ADD_FV2_FV2(a,b)
#define SUB_FV2_FV2_FV2(y,a,b)    y = SUB_FV2_FV2(a,b)
#define MUL_FV2_FV2_FV2(y,a,b)    y = MUL_FV2_FV2(a,b)
#define NEG_FV2_FV2(y,a)          y = NEG_FV2(a)

#define FMA_FV2_FV2_FV2_FV2(y,a,b,c)     y = FMA_FV2_FV2_FV2(a,b,c)
#define FMS_FV2_FV2_FV2_FV2(y,a,b,c)     y = FMS_FV2_FV2_FV2(a,b,c)
#define NFMA_FV2_FV2_FV2_FV2(y,a,b,c)    y = NFMA_FV2_FV2_FV2(a,b,c)

#define VUNPKLO_FV2_FV2_FV2(y,a,b)    y = VUNPKLO_FV2_FV2(a,b)
#define VUNPKHI_FV2_FV2_FV2(y,a,b)    y = VUNPKHI_FV2_FV2(a,b)
#define VPKLO_FV2_FV2_FV2(y,a,b)      y = VPKLO_FV2_FV2(a,b)
#define VPKHI_FV2_FV2_FV2(y,a,b)      y = VPKHI_FV2_FV2(a,b)


#define NTH_FV2(loc, idx)         (((FV2*)(loc))[idx])
#define VLOAD_FV2(var, loc, idx)  (var) = (((FV2*)(loc))[idx])
#define VSTORE_FV2(loc, idx, exp) (((FV2*)(loc))[idx]) = (exp)

#define VDECLC_FV2(c, a0, a1)     const FV2 c = {{(a0), (a1)}}
#define C_FV2(a, b)               VPK_FV2((a), (b))
#define VDUP_FV2(a)               VPK_FV2((a), (a))


INLINE FV2 NEG_FV2(const FV2 a) {
    FV2 r;
    r.d[0] = -a.d[0];
    r.d[1] = -a.d[1];
    return r;
}

INLINE FV2 VPK_FV2(const FLT a, const FLT b) {
    FV2 r;
    r.d[0] = a;
    r.d[1] = b;
    return r;
}

INLINE FV2 VUNPKLO_FV2_FV2(const FV2 a, const FV2 b) {
    FV2 r;
    r.d[0] = a.d[0];
    r.d[1] = b.d[0];
    return r;
}

INLINE FV2 VUNPKHI_FV2_FV2(const FV2 a, const FV2 b) {
    FV2 r;
    r.d[0] = a.d[1];
    r.d[1] = b.d[1];
    return r;
}

INLINE FV2 VPKLO_FV2_FV2(const FV2 a, const FV2 b) {
    return VUNPKLO_FV2_FV2(a,b);
}

INLINE FV2 VPKHI_FV2_FV2(const FV2 a, const FV2 b) {
    return VUNPKHI_FV2_FV2(a,b);
}

INLINE FV2 ADD_FV2_FV2(const FV2 a, const FV2 b) {
    FV2 r; int i;
    for (i=0; i<2; i++)
        r.d[i] = a.d[i] + b.d[i];
    return r;
}

INLINE FV2 SUB_FV2_FV2(const FV2 a, const FV2 b) {
    FV2 r; int i;
    for (i=0; i<2; i++)
        r.d[i] = a.d[i] - b.d[i];
    return r;
}

INLINE FV2 MUL_FV2_FV2(const FV2 a, const FV2 b) {
    FV2 r; int i;
    for (i=0; i<2; i++)
        r.d[i] = a.d[i] * b.d[i];
    return r;
}

INLINE FV2 FMA_FV2_FV2_FV2(const FV2 a, const FV2 b, const FV2 c) {
    return ADD_FV2_FV2(a, MUL_FV2_FV2(b,c));
}

INLINE FV2 FMS_FV2_FV2_FV2(const FV2 a, const FV2 b, const FV2 c) {
    return SUB_FV2_FV2(a, MUL_FV2_FV2(b,c));
}

INLINE FV2 NFMA_FV2_FV2_FV2(const FV2 a, const FV2 b, const FV2 c) {
    return SUB_FV2_FV2(MUL_FV2_FV2(b,c), a);
}

INLINE FV2 VSWAPCX_FV2_FV2(const FV2 a) {
    FV2 r;
    r.d[0] = a.d[1];
    r.d[1] = a.d[0];
    return r;
}

INLINE FV2 VMULCX_FV2_FV2(const FV2 a, const FV2 b) {
    FV2 r;
    r.d[0] = a.d[0]*b.d[0] - a.d[1]*b.d[1];
    r.d[1] = a.d[0]*b.d[1] + a.d[1]*b.d[0];
    return r;
}


#endif /*  __MACRO_FV2_H__ */
