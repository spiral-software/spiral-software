/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#ifndef __MACRO_CMN__
#define __MACRO_CMN__


#if defined(__ICL) || defined(_MSC_VER)
#define INLINE	__inline
#else
#define INLINE	__inline__
#endif

#include <include/omega64.h>

#define FLT    double
#define INT    int

#define C_FLT(a)    ((FLT)(a))
#define CD_FLT(a)   ((FLT)(a))

#define ADD_FLT_FLT(a,b) ((a)+(b))
#define SUB_FLT_FLT(a,b) ((a)-(b))
#define DIV_FLT_FLT(a,b) ((a)/(b))
#define NTH_FLT(v,i) ((v)[i])
#define MUL_FLT_FLT(a, b) ((a)*(b))
#define MUL_FLT_INT(a, b) ((a)*(b))
#define MUL_INT_FLT(a, b) ((a)*(b))
#define NEG_FLT(a) (-a)

#define MUL_FLT_UNK MUL_FLT_FLT
#define MUL_UNK_FLT MUL_FLT_FLT
#define MUL_INT_UNK MUL_INT_FLT
#define MUL_UNK_INT MUL_FLT_INT

#define ADD_INT_FLT(a,b) ADD_FLT_FLT(C_FLT(a), b)
#define ADD_INT_UNK(a,b) ADD_FLT_FLT(C_FLT(a), b)
#define ADD_FLT_INT(a,b) ADD_FLT_FLT(a, C_FLT(b))
#define ADD_UNK_INT(a,b) ADD_FLT_FLT(a, C_FLT(b))

#define FDIV_INT_INT(a,b) C_FLT(((double)(a)) / (b))
#define FDIV_FLT_INT(a,b) DIV_FLT_FLT(a, C_FLT(b))
#define FDIV_INT_FLT(a,b) DIV_FLT_FLT(C_FLT(a), b)
#define FDIV_FLT_FLT(a,b) DIV_FLT_FLT(a, b)

#define MUL_INT_INT(a,b) ((a) * (b))
#define ADD_INT_INT(a,b) ((a) + (b))
#define SUB_INT_INT(a,b) ((a) - (b))
#define AND_INT_INT(a,b) ((a) & (b))
#define XOR_INT_INT(a,b) ((a) ^ (b))
#define DIV_INT_INT(a,b) ((a) / (b))
#define IDIV_INT(a, b) ((a)/(b))
#define IMOD_INT(a, b) ((a)%(b))
#define NEG_INT(a) (-(a))
#define NTH_INT(v,i) ((v)[i])

#define NTH_SYM(v,i) (v)[i]


#define ASSIGN_INT(loc, val)   (loc) = (val)


#endif /* __MACRO_CMN__ */

