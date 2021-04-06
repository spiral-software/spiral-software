/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#ifndef __MACRO_FV8_H__
#define __MACRO_FV8_H__

#include "macro_cmn.h"

typedef struct {
    FLT d[8];
} FV8;


/* three-op */
#define ADD_FV8_FV8_FV8(y,a,b)    y = ADD_FV8_FV8(a,b)
#define SUB_FV8_FV8_FV8(y,a,b)    y = SUB_FV8_FV8(a,b)
#define MUL_FV8_FV8_FV8(y,a,b)    y = MUL_FV8_FV8(a,b)
#define NEG_FV8_FV8(y,a)          y = NEG_FV8(a)

#define FMA_FV8_FV8_FV8_FV8(y,a,b,c)     y = FMA_FV8_FV8_FV8(a,b,c)
#define FMS_FV8_FV8_FV8_FV8(y,a,b,c)     y = FMS_FV8_FV8_FV8(a,b,c)
#define NFMA_FV8_FV8_FV8_FV8(y,a,b,c)    y = NFMA_FV8_FV8_FV8(a,b,c)

#define VUNPKLO_FV8_FV8_FV8(y,a,b)        y = VUNPKLO_FV8_FV8(a,b) 
#define VUNPKHI_FV8_FV8_FV8(y,a,b)        y = VUNPKHI_FV8_FV8(a,b)
#define VUNPKLO2_FV8_FV8_FV8(y,a,b)       y = VUNPKLO2_FV8_FV8(a,b)
#define VUNPKHI2_FV8_FV8_FV8(y,a,b)       y = VUNPKHI2_FV8_FV8(a,b)
#define VPKLO_FV8_FV8_FV8(y,a,b)          y = VPKLO_FV8_FV8(a,b)
#define VPKHI_FV8_FV8_FV8(y,a,b)          y = VPKHI_FV8_FV8(a,b)
#define VPKLO2_FV8_FV8_FV8(y,a,b)         y = VPKLO2_FV8_FV8(a,b)
#define VPKHI2_FV8_FV8_FV8(y,a,b)         y = VPKHI2_FV8_FV8(a,b)


#define NTH_FV8(loc, idx)          (((FV8*)(loc))[idx])
#define VLOAD_FV8(var, loc, idx)   (var) = (((FV8*)(loc))[idx])
#define VSTORE_FV8(loc, idx, exp)  (((FV8*)(loc))[idx]) = (exp)

#define VDECLC_FV8(c, a0, a1, a2, a3, a4, a5, a6, a7)    const FV8 c = {{(a0), (a1), (a2), (a3), (a4), (a5), (a6), (a7)}}
#define C_FV8(a0, a1, a2, a3, a4, a5, a6, a7)     VPK_FV8((a0), (a1), (a2), (a3), (a4), (a5), (a6), (a7))
#define VDUP_FV8(a)                               VPK_FV8( (a),  (a),  (a),  (a),  (a),  (a),  (a),  (a))


INLINE FV8 NEG_FV8(const FV8 a) {
    FV8 r;
    r.d[0] = -a.d[0];  r.d[1] = -a.d[1];  r.d[2] = -a.d[2];  r.d[3] = -a.d[3];
    r.d[4] = -a.d[4];  r.d[5] = -a.d[5];  r.d[6] = -a.d[6];  r.d[7] = -a.d[7];
    return r;
}

INLINE FV8 VPK_FV8(const FLT a0, const FLT a1, const FLT a2, const FLT a3,
                   const FLT a4, const FLT a5, const FLT a6, const FLT a7) 
{
    FV8 r;
    r.d[0] = a0;  r.d[1] = a1;  r.d[2] = a2;  r.d[3] = a3;
    r.d[4] = a4;  r.d[5] = a5;  r.d[6] = a6;  r.d[7] = a7;
    return r;
}

INLINE FV8 VUNPKLO_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r;
    r.d[0] = a.d[0];  r.d[1] = b.d[0];  r.d[2] = a.d[1];  r.d[3] = b.d[1];
    r.d[4] = a.d[2];  r.d[5] = b.d[2];  r.d[6] = a.d[3];  r.d[7] = b.d[3];
    return r;
}

INLINE FV8 VUNPKHI_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r;
    r.d[0] = a.d[4];  r.d[1] = b.d[4];  r.d[2] = a.d[5];  r.d[3] = b.d[5];
    r.d[4] = a.d[6];  r.d[5] = b.d[6];  r.d[6] = a.d[7];  r.d[7] = b.d[7];
    return r;
}

INLINE FV8 VUNPKLO2_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r;
    r.d[0] = a.d[0];  r.d[1] = a.d[1];  r.d[2] = b.d[0];  r.d[3] = b.d[1];
    r.d[4] = a.d[2];  r.d[5] = a.d[3];  r.d[6] = b.d[2];  r.d[7] = b.d[3];
    return r;
}

INLINE FV8 VUNPKHI2_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r;
    r.d[0] = a.d[4];  r.d[1] = a.d[5];  r.d[2] = b.d[4];  r.d[3] = b.d[5];
    r.d[4] = a.d[6];  r.d[5] = a.d[7];  r.d[6] = b.d[6];  r.d[7] = b.d[7];
    return r;
}

INLINE FV8 VPKLO_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r;
    r.d[0] = a.d[0];  r.d[1] = a.d[2];  r.d[2] = a.d[4];  r.d[3] = a.d[6];
    r.d[4] = b.d[0];  r.d[5] = b.d[2];  r.d[6] = b.d[4];  r.d[7] = b.d[6];
    return r;
}

INLINE FV8 VPKHI_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r;
    r.d[0] = a.d[1];  r.d[1] = a.d[3];  r.d[2] = a.d[5];  r.d[3] = a.d[7];
    r.d[4] = b.d[1];  r.d[5] = b.d[3];  r.d[6] = b.d[5];  r.d[7] = b.d[7];
    return r;
}

INLINE FV8 VPKLO2_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r;
    r.d[0] = a.d[0];  r.d[1] = a.d[1];  r.d[2] = a.d[4];  r.d[3] = a.d[5];
    r.d[4] = b.d[0];  r.d[5] = b.d[1];  r.d[6] = b.d[4];  r.d[7] = b.d[5];
    return r;
}

INLINE FV8 VPKHI2_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r;
    r.d[0] = a.d[2];  r.d[1] = a.d[3];  r.d[2] = a.d[6];  r.d[3] = a.d[7];
    r.d[4] = b.d[2];  r.d[5] = b.d[3];  r.d[6] = b.d[6];  r.d[7] = b.d[7];
    return r;
}

INLINE FV8 ADD_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r; int i;
    for (i=0; i<8; i++)
        r.d[i] = a.d[i] + b.d[i];
    return r;
}

INLINE FV8 SUB_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r; int i;
    for (i=0; i<8; i++)
        r.d[i] = a.d[i] - b.d[i];
    return r;
}

INLINE FV8 MUL_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r; int i;
    for (i=0; i<8; i++)
        r.d[i] = a.d[i] * b.d[i];
    return r;
}

INLINE FV8 FMA_FV8_FV8_FV8(const FV8 a, const FV8 b, const FV8 c) {
    return ADD_FV8_FV8(a, MUL_FV8_FV8(b,c));
}

INLINE FV8 FMS_FV8_FV8_FV8(const FV8 a, const FV8 b, const FV8 c) {
    return SUB_FV8_FV8(a, MUL_FV8_FV8(b,c));
}

INLINE FV8 NFMA_FV8_FV8_FV8(const FV8 a, const FV8 b, const FV8 c) {
    return SUB_FV8_FV8(MUL_FV8_FV8(b,c), a);
}

INLINE FV8 VSWAPCX_FV8_FV8(const FV8 a) {
    FV8 r;
    r.d[0] = a.d[1]; r.d[1] = a.d[0];
    r.d[2] = a.d[3]; r.d[3] = a.d[2];
    r.d[4] = a.d[5]; r.d[5] = a.d[4];
    r.d[6] = a.d[7]; r.d[7] = a.d[6];
    return r;
}

INLINE FV8 VMULCX_FV8_FV8(const FV8 a, const FV8 b) {
    FV8 r; int i;
    for (i=0; i<4; i++) {
        r.d[2*i  ] = a.d[2*i]*b.d[2*i  ] - a.d[2*i+1]*b.d[2*i+1];
        r.d[2*i+1] = a.d[2*i]*b.d[2*i+1] + a.d[2*i+1]*b.d[2*i  ];
    }
    return r;
}

#endif /*  __MACRO_FV8_H__ */
