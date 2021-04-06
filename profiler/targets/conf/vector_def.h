/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#ifndef VECTOR_DEF_INCLUDED
#define VECTOR_DEF_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "xmalloc.h"
#include "sys.h"

/* for fixed point */

#define FRACBITS(t)    (t->fracbits)
#define ZEROBITS(t)    (t->zerobits)
#define TOFIXED(t, x)  ((((signed int) ((x) * (double)(0x1 << (FRACBITS(t))))) >> ZEROBITS(t)) << ZEROBITS(t));
#define TODOUBLE(t, x) ((double)(x) / (double)(1<<(FRACBITS(t))))
#define FPMUL(t, x, y) (((x)*(y))>>(FRACBITS(t)))

/* container structure that knows everthing relevant to a particular data type. */
typedef struct scalar_type {
    char *name;           /*!< name of this type to be used in textual output */
    int size;             /*!< sizeof() this type to be used in array calculations */

    void*  (*alloc)         (struct scalar_type *t, int N);                   /*!< allocate a vector of N scalars */
    void   (*free)          (struct scalar_type *t, void *vec);               /*!< deallocate storage allocated with alloc */
    void*  (*nth)           (struct scalar_type *t, void *vec, int n);        /*!< return pointer to n-th scalar in a vector vec */
    void   (*one)           (struct scalar_type *t, void *x);                 /*!< (*x) = 1 */
    void   (*zero)          (struct scalar_type *t, void *x);                 /*!< (*x) = 0 */
    void   (*set)           (struct scalar_type *t, void *x, void *a);        /*!< (*x) = (*a) */
    void   (*set_fix)       (struct scalar_type *t, void *x, void *a);        /*!< (*x) = (*a) */
    void   (*set_double)    (struct scalar_type *t, void *x, void *a);        /*!< (*x) = (*a) */
    void   (*random)        (struct scalar_type *t, void *x);                 /*!< (*x) = random element */

    void   (*add)           (struct scalar_type *t, void *x, void *a, void *b); /*!< (*x) = (*a) + (*b) */
    void   (*sub)           (struct scalar_type *t, void *x, void *a, void *b); /*!< (*x) = (*a) - (*b) */
    void   (*mul)           (struct scalar_type *t, void *x, void *a, void *b); /*!< (*x) = (*a) * (*b) */
    int    (*rational)      (struct scalar_type *t, void *x, int k, int n);/*!< (*x) = k/n and return 1 on success */
    int    (*div)           (struct scalar_type *t, void *x, void *a, void *b); /*!< (*x) = k/n and returns 1 on success */

    int    (*less)          (struct scalar_type *t, void *a, void *b);         /*!< returns 1 if (*a) < (*b) */
    void   (*abs)           (struct scalar_type *t, void *x, void *a);         /*!< (*x) = absolute_value_of(*a) */

    /* int    (*rootOfUnity) (void *x, int n, int k);   (*x) = w^k for a fixed primitive n-th root of unity w and returns 1 on success */

    void   (*fprint)        (struct scalar_type *t, FILE* f, void *a);       /*!< print (*a) to a file f */
    void   (*fprint_gap)    (struct scalar_type *t, FILE* f, void *a);   /*!< print (*a) to a file f as GAP expression */

    /* additional information for particular types is stored beyond this point */
    /* there is no inheritence structure in this code, so additional fields used by only */
    /* certain datatypes will all just be place here. too bad. */

    	int fracbits;
	int zerobits;
} scalar_type_t;


typedef void* scalar_t;


scalar_t ALLOC      (scalar_type_t *t, int N); 
#define  ALLOC(T, N)          T->alloc(T, N)
void     FREE       (scalar_type_t *t, scalar_t vec);
#define  FREE(T, X)           T->free(T, X)
scalar_t IDX        (scalar_type_t *t, scalar_t vec, int n);
#define  IDX(T, X, N)         T->nth(T, X, N)
void     ONE        (scalar_type_t *t, scalar_t x);         
#define  ONE(T, X)            T->one(T, X)
void     ZERO       (scalar_type_t *t, scalar_t x);         
#define  ZERO(T, X)           T->zero(T, X)
void     SET        (scalar_type_t *t, scalar_t x, scalar_t a);
#define  SET(T, X, A)         T->set(T, X, A)
void     SET_FIX    (scalar_type_t *t, scalar_t x, scalar_t a);
#define  SET_FIX(T, X, A)     T->set_fix(T, X, A)
void     SET_DOUBLE (scalar_type_t *t, scalar_t x, scalar_t a);
#define  SET_DOUBLE(T, X, A)  T->set_double(T, X, A)
void     RANDOM     (scalar_type_t *t, scalar_t x);                
#define  RANDOM(T, X)         T->random(T, X)
void     ADD        (scalar_type_t *t, scalar_t x, scalar_t a, scalar_t b);
#define  ADD(T, X, A, B)      T->add(T, X, A, B)
void     SUB        (scalar_type_t *t, scalar_t x, scalar_t a, scalar_t b);
#define  SUB(T, X, A, B)      T->sub(T, X, A, B)
void     MUL        (scalar_type_t *t, scalar_t x, scalar_t a, scalar_t b);
#define  MUL(T, X, A, B)      T->mul(T, X, A, B)
int      RATIONAL   (scalar_type_t *t, scalar_t x, int k, int n);
#define  RATIONAL(T, X, K, N) T->rational(T,X,K,N)
int      DIV        (scalar_type_t *t, scalar_t x, scalar_t a, scalar_t b);
#define  DIV(T, X, A, B)      T->div(T, X, A, B)
int      LESS       (scalar_type_t *t, scalar_t a, scalar_t b);
#define  LESS(T, A, B)        T->less(T,A,B)
void     ABS        (scalar_type_t *t, scalar_t x, scalar_t a);
#define  ABS(T, X, A)         T->abs(T,X,A)
void     PRINT      (scalar_type_t *t, FILE *f, scalar_t a);
#define  PRINT(T, f, X)          T->fprint(T,f, X)
void     PRINT_GAP  (scalar_type_t *t, FILE *f, scalar_t a);
#define  PRINT_GAP(T, f, X)      T->fprint_gap(T,f, X)


/**
 * vector_t
 *
 **/
typedef struct vector {
    void *data;
    int size;
    scalar_type_t * type;
} vector_t;


/** 
 * SCALAR_TYPE_DECL( ) 
 *
 **/
#define SCALAR_TYPE_DECL(PREFIX, TYP_STRING, SIZEOF_T, TYP_REC) \
scalar_type_t __##TYP_REC = { \
	TYP_STRING, \
	SIZEOF_T, \
	PREFIX##_alloc, \
	PREFIX##_free, \
        PREFIX##_nth, \
	PREFIX##_one, \
	PREFIX##_zero, \
        PREFIX##_set, \
        PREFIX##_set_fix, \
        PREFIX##_set_double, \
        PREFIX##_random, \
	PREFIX##_add, \
	PREFIX##_sub, \
	PREFIX##_mul, \
	PREFIX##_rational, \
	PREFIX##_div, \
	PREFIX##_less, \
	PREFIX##_abs, \
	/* PREFIX##_rootOfUnity, */ \
	PREFIX##_fprint, \
	PREFIX##_fprintgap \
}; \
scalar_type_t * TYP_REC = &__##TYP_REC;

/** 
 * PRIMITIVE_REAL_PACKAGE( ) 
 *
 * this is a C implementation of a C++ style template
 **/
#define PRIMITIVE_REAL_PACKAGE(T, PREFIX, FMT, GAP_FMT, RAND) \
\
	void* PREFIX##_alloc     (scalar_type_t *t, int num)   { return xaligned_malloc(sizeof(T) * num); } \
	void  PREFIX##_free      (scalar_type_t *t, void *vec) { xaligned_free(vec); } \
	void* PREFIX##_nth       (scalar_type_t *t, void *vec, int n)   { return (T*)vec + n; } \
	void  PREFIX##_one       (scalar_type_t *t, void *x) { *((T*)x) = 1; } \
	void  PREFIX##_zero      (scalar_type_t *t, void *x) { *((T*)x) = 0; } \
	void  PREFIX##_set       (scalar_type_t *t, void *x, void *a) { *((T*)x) = *((T*)a); } \
	void  PREFIX##_set_fix   (scalar_type_t *t, void *x, void *a) { *((T*)x) = *((T*)a); } \
	void  PREFIX##_set_double(scalar_type_t *t, void *x, void *a) { *((T*)x) = *((T*)a); } \
	void  PREFIX##_random    (scalar_type_t *t, void *x) { *((T*)x) = RAND; } \
	void  PREFIX##_add       (scalar_type_t *t, void *x, void *a, void *b)  { *((T*)x) = *((T*)a) + *((T*)b); } \
	void  PREFIX##_sub       (scalar_type_t *t, void *x, void *a, void *b)  { *((T*)x) = *((T*)a) - *((T*)b); } \
	void  PREFIX##_mul       (scalar_type_t *t, void *x, void *a, void *b)  { *((T*)x) = *((T*)a) * *((T*)b); } \
	int   PREFIX##_div       (scalar_type_t *t, void *x, void *a, void *b)  { if((T*)b!=0) { *((T*)x) = *((T*)a) / *((T*)b); return 1; } \
                                                    else return 0; } \
	int   PREFIX##_rational  (scalar_type_t *t, void *x, int k, int n)    { if(n!=0) { *((T*)x) = (T)k / (T)n; return 1; } \
                                                    else return 0; } \
	int   PREFIX##_less      (scalar_type_t *t, void *a, void *b)  { return *((T*)a) < *((T*)b); } \
	void  PREFIX##_abs       (scalar_type_t *t, void *x, void *a)  { if(*((T*)a) < 0) *((T*)x) = - *((T*)a); else *((T*)x) = *((T*)a); } \
	void  PREFIX##_fprint    (scalar_type_t *t, FILE *f, void *a) { fprintf(f, FMT, *((T*)a)); } \
	void  PREFIX##_fprintgap (scalar_type_t *t, FILE *f, void *a) { fprintf(f, GAP_FMT, *((T*)a)); } \
\
EXPORT SCALAR_TYPE_DECL(PREFIX, #PREFIX, sizeof(T), PREFIX##_record);

/** 
 * PRIMITIVE_CPLX_PACKAGE( ) 
 *
 * this is a C implementation of a C++ style template
 **/
#define PRIMITIVE_CPLX_PACKAGE(T, PREFIX, FMT, GAP_FMT, RAND, SQRT) \
\
	void* PREFIX##_cplx_alloc       (scalar_type_t *t, int num)           { return xaligned_malloc(sizeof(T) * num * 2); } \
	void  PREFIX##_cplx_free        (scalar_type_t *t, void *vec)         { xaligned_free(vec); } \
	void* PREFIX##_cplx_nth         (scalar_type_t *t, void *vec, int n)  { return (T*)vec + 2*n; } \
	void  PREFIX##_cplx_one         (scalar_type_t *t, void *x)           { *((T*)x) = 1; *((T*)x+1) = 0;} \
	void  PREFIX##_cplx_zero        (scalar_type_t *t, void *x)           { *((T*)x) = 0; *((T*)x+1) = 0;} \
	void  PREFIX##_cplx_set         (scalar_type_t *t, void *x, void *a)  { *((T*)x) = *((T*)a); *((T*)x+1) = *((T*)a+1); } \
	void  PREFIX##_cplx_set_fix     (scalar_type_t *t, void *x, void *a)  { *((T*)x) = *((T*)a); *((T*)x+1) = *((T*)a+1); } \
	void  PREFIX##_cplx_set_double  (scalar_type_t *t, void *x, void *a)  { *((T*)x) = *((T*)a); *((T*)x+1) = *((T*)a+1); } \
	void  PREFIX##_cplx_random      (scalar_type_t *t, void *x)           { *((T*)x) = (T) RAND; *((T*)x+1) = (T) RAND; } \
	\
	void  PREFIX##_cplx_add         (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		*((T*)x)   = *((T*)a)   + *((T*)b); \
		*((T*)x+1) = *((T*)a+1) + *((T*)b+1); \
	} \
	\
	void  PREFIX##_cplx_sub         (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		*((T*)x)   = *((T*)a)   - *((T*)b); \
		*((T*)x+1) = *((T*)a+1) - *((T*)b+1); \
	} \
	\
	void  PREFIX##_cplx_mul         (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		*((T*)x)   = *((T*)a) * *((T*)b)   - *((T*)a+1) * *((T*)b+1); \
		*((T*)x+1) = *((T*)a) * *((T*)b+1) + *((T*)a+1) * *((T*)b); \
	} \
	\
	int   PREFIX##_cplx_div         (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		*((T*)x)   = *((T*)a)   * *((T*)b) + *((T*)a+1) * *((T*)b+1); \
		*((T*)x+1) = *((T*)a+1) * *((T*)b) - *((T*)a)   * *((T*)b+1); \
		*((T*)x)   = *((T*)x)   / (*((T*)b) * *((T*)b) + *((T*)b+1) * *((T*)b+1)); \
		*((T*)x+1) = *((T*)x+1) / (*((T*)b) * *((T*)b) + *((T*)b+1) * *((T*)b+1)); \
		 \
		return 1; \
	} \
	 \
	int   PREFIX##_cplx_rational    (scalar_type_t *t, void *x, int k, int n) \
	{ \
		if(n!=0) \
		{ \
			*((T*)x) = (T)k / (T)n; \
			*((T*)x+1) = (T) 0; \
			return 1; \
		} \
		 \
		return 0; \
	} \
	\
	void  PREFIX##_cplx_abs         (scalar_type_t *t, void *x, void *a) \
	{ \
		*((T*)x) = (T) SQRT(*((T*)a) * *((T*)a) + *((T*)a+1) * *((T*)a+1)); \
		*((T*)x+1) = (T) 0; \
	} \
	\
	int   PREFIX##_cplx_less        (scalar_type_t *t, void *a, void *b) \
	{ \
		void *abs_a = PREFIX##_cplx_alloc(t, 1); \
		void *abs_b = PREFIX##_cplx_alloc(t, 1); \
		int result; \
		 \
		PREFIX##_cplx_abs(t, abs_a, a); \
		PREFIX##_cplx_abs(t, abs_b, b); \
		 \
		result = *((T*)abs_a) < *((T*)abs_b); \
		 \
		free(abs_a); \
		free(abs_b); \
		 \
		return result; \
	} \
	\
	void  PREFIX##_cplx_fprint      (scalar_type_t *t, FILE *f, void *a) { fprintf(f, FMT, *((T*)a), *((T*)a+1)); } \
	void  PREFIX##_cplx_fprintgap   (scalar_type_t *t, FILE *f, void *a) { fprintf(f, GAP_FMT, *((T*)a), *((T*)a+1)); } \
\
SCALAR_TYPE_DECL(PREFIX##_cplx, #PREFIX "_cplx", sizeof(T) * 2, PREFIX##_cplx_record);


/*
 * fixed point real
 *
 */

#define PRIMITIVE_FPREAL_PACKAGE(T, PREFIX, FMT, GAP_FMT, RAND) \
	void* PREFIX##_alloc (scalar_type_t *t, int num) { return xaligned_malloc(sizeof(T) * num); } \
	void  PREFIX##_free  (scalar_type_t *t, void *vec) { xaligned_free(vec); } \
	void* PREFIX##_nth   (scalar_type_t *t, void *vec, int n)  { return (T*)vec + n; } \
	void  PREFIX##_one   (scalar_type_t *t, void *x) { *((T*)x) = TOFIXED(t,1); } \
	void  PREFIX##_zero  (scalar_type_t *t, void *x) { *((T*)x) = 0; } \
	void  PREFIX##_set   (scalar_type_t *t, void *x, void *a) { *((T*)x) = *((T*)a); } \
	void  PREFIX##_set_fix   (scalar_type_t *t, void *x, void *a) { *((T*)x) = TOFIXED(t, *((double*)a)); } \
	void  PREFIX##_set_double   (scalar_type_t *t, void *x, void *a) { *((double*)x) = TODOUBLE(t,*((T*)a)); } \
	\
	void  PREFIX##_random(scalar_type_t *t, void *x) \
	{ \
		double tmp0; \
		\
		tmp0 = RAND; \
		*((T*)x) = TOFIXED(t, tmp0); \
	} \
	\
	void  PREFIX##_add   (scalar_type_t *t, void *x, void *a, void *b) {	*((T*)x)   = *((T*)a) + *((T*)b); } \
	void  PREFIX##_sub   (scalar_type_t *t, void *x, void *a, void *b) { *((T*)x) = *((T*)a) - *((T*)b); } \
	void  PREFIX##_mul   (scalar_type_t *t, void *x, void *a, void *b) { *((T*)x) = *((T*)a) * *((T*)b); } \
	\
	int   PREFIX##_div   (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		if(*((T*)b)!=0) \
		{ \
			*((T*)x) = *((T*)a) / *((T*)b); \
			return 1; \
		} \
		else \
			return 0; \
	} \
	\
	int   PREFIX##_rational(scalar_type_t *t, void *x, int k, int n) \
	{ \
		if(n!=0) \
		{ \
			*((T*)x) = TOFIXED(t,(int)k / (int)n); \
			return 1; \
		} \
		\
		return 0; \
	} \
	\
	void  PREFIX##_abs   (scalar_type_t *t, void *x, void *a) \
	{ \
		if(*((T*)a) < 0) \
			*((T*)x) = - *((T*)a); \
		else \
			*((T*)x) = *((T*)a); \
	} \
	\
	int   PREFIX##_less  (scalar_type_t *t, void *a, void *b) { return *((T*)a) < *((T*)b); } \
	\
	void  PREFIX##_fprint(scalar_type_t *t, FILE *f, void *a) { fprintf(f, FMT, TODOUBLE(t,*((T*)a))); } \
	void  PREFIX##_fprintgap(scalar_type_t *t, FILE *f, void *a) { fprintf(f, GAP_FMT, TODOUBLE(t,*((T*)a))); } \
\
SCALAR_TYPE_DECL(PREFIX, #PREFIX "_fp", sizeof(T), PREFIX##_record);

/* 
 * fixed point complex
 *
 */

#define PRIMITIVE_FPCPLX_PACKAGE(T, PREFIX, FMT, GAP_FMT, RAND, SQRT) \
	void* PREFIX##_cplx_alloc      (scalar_type_t *t, int num)          { return xaligned_malloc(sizeof(T) * num * 2); } \
	void  PREFIX##_cplx_free       (scalar_type_t *t, void *vec)        { xaligned_free(vec); } \
	void* PREFIX##_cplx_nth        (scalar_type_t *t, void *vec, int n) { return (T*)vec + 2*n; } \
	void  PREFIX##_cplx_one        (scalar_type_t *t, void *x)          { *((T*)x) = TOFIXED(t,1); *((T*)x+1) = 0;} \
	void  PREFIX##_cplx_zero       (scalar_type_t *t, void *x)          { *((T*)x) = 0; *((T*)x+1) = 0;} \
	void  PREFIX##_cplx_set        (scalar_type_t *t, void *x, void *a) { *((T*)x) = *((T*)a); *((T*)x+1) = *((T*)a+1); } \
	void  PREFIX##_cplx_set_fix    (scalar_type_t *t, void *x, void *a) { *((T*)x) = TOFIXED(t,*((double*)a)); *((T*)x+1) = TOFIXED(t,*((double*)a+1)); } \
	void  PREFIX##_cplx_set_double (scalar_type_t *t, void *x, void *a) { *((double*)x) = TODOUBLE(t,*((T*)a)); *((double*)x+1) = TODOUBLE(t,*((T*)a+1)); } \
	\
	void  PREFIX##_cplx_random(scalar_type_t *t, void *x) \
	{ \
		double tmp0, tmp1; \
		tmp0 = RAND; \
		*((T*)x) = TOFIXED(t, tmp0); \
		tmp1 = RAND; \
		*((T*)x+1) = TOFIXED(t, tmp1); \
	} \
	\
	void  PREFIX##_cplx_add   (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		*((T*)x)   = *((T*)a) + *((T*)b); \
		*((T*)x+1) = *((T*)a+1) + *((T*)b+1); \
	} \
	\
	void  PREFIX##_cplx_sub   (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		*((T*)x) = *((T*)a) - *((T*)b); \
		*((T*)x+1) = *((T*)a+1) - *((T*)b+1); \
	} \
	\
	void  PREFIX##_cplx_mul   (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		*((T*)x) = *((T*)a) * *((T*)b) - *((T*)a+1) * *((T*)b+1); \
		*((T*)x+1) = *((T*)a) * *((T*)b+1) + *((T*)a+1) * *((T*)b); \
	} \
	\
	int   PREFIX##_cplx_div   (scalar_type_t *t, void *x, void *a, void *b) \
	{ \
		*((T*)x) = *((T*)a) * *((T*)b) + *((T*)a+1) * *((T*)b+1); \
		*((T*)x+1) = *((T*)a+1) * *((T*)b) - *((T*)a)   * *((T*)b+1); \
		*((T*)x)   = *((T*)x)   / (*((T*)b) * *((T*)b) + *((T*)b+1) * *((T*)b+1)); \
		*((T*)x+1) = *((T*)x+1) / (*((T*)b) * *((T*)b) + *((T*)b+1) * *((T*)b+1)); \
		return 1; \
	} \
	\
	int   PREFIX##_cplx_rational(scalar_type_t *t, void *x, int k, int n) \
	{ \
		if(n!=0) \
		{ \
			*((T*)x) = TOFIXED(t,(int)k / (int)n); \
			*((T*)x+1) = 0; return 1; \
		} \
		else \
			return 0; \
	} \
	\
	void  PREFIX##_cplx_abs   (scalar_type_t *t, void *x, void *a) \
	{ \
		*((T*)x) = (T) SQRT(FPMUL(t,*((T*)a), *((T*)a)) + FPMUL(t,*((T*)a+1), *((T*)a+1))); \
		*((T*)x+1) = 0; \
	} \
	\
	int   PREFIX##_cplx_less  (scalar_type_t *t, void *a, void *b) \
	{ \
		void *abs_a = PREFIX##_cplx_alloc(t, 1); \
		void *abs_b = PREFIX##_cplx_alloc(t, 1); \
		int result; \
		\
		PREFIX##_cplx_abs(t, abs_a, a); \
		PREFIX##_cplx_abs(t, abs_b, b); \
		result = *((T*)abs_a) < *((T*)abs_b); \
		\
		free(abs_a); \
		free(abs_b); \
		\
		return result; \
	} \
	\
	void  PREFIX##_cplx_fprint(scalar_type_t *t, FILE *f, void *a) { fprintf(f, FMT, TODOUBLE(t, *((T*)a)), TODOUBLE(t,*((T*)a+1))); } \
	void  PREFIX##_cplx_fprintgap(scalar_type_t *t, FILE *f, void *a) { fprintf(f, GAP_FMT, TODOUBLE(t,*((T*)a)), TODOUBLE(t,*((T*)a+1))); } \
\
SCALAR_TYPE_DECL(PREFIX##_cplx, #PREFIX "_cplx_fp", sizeof(T), PREFIX##_cplx_record);


#define PRIMITIVE_PLACEHOLDER(T, PREFIX, FMT, GAP_FMT, RAND, SQRT) \
	void*  PREFIX##_cplx_alloc      (scalar_type_t *t, int N)                     { return NULL; } \
	void   PREFIX##_cplx_free       (scalar_type_t *t, void *vec)                 { ; } \
	void*  PREFIX##_cplx_nth        (scalar_type_t *t, void *vec, int n)          { return NULL; } \
	void   PREFIX##_cplx_one        (scalar_type_t *t, void *x)                   { ; } \
	void   PREFIX##_cplx_zero       (scalar_type_t *t, void *x)                   { ; } \
	void   PREFIX##_cplx_set        (scalar_type_t *t, void *x, void *a)          { ; } \
	void   PREFIX##_cplx_set_fix    (scalar_type_t *t, void *x, void *a)          { ; } \
	void   PREFIX##_cplx_set_double (scalar_type_t *t, void *x, void *a)          { ; } \
	void   PREFIX##_cplx_random     (scalar_type_t *t, void *x)                   { ; } \
	void   PREFIX##_cplx_add        (scalar_type_t *t, void *x, void *a, void *b) { ; } \
	void   PREFIX##_cplx_sub        (scalar_type_t *t, void *x, void *a, void *b) { ; } \
	void   PREFIX##_cplx_mul        (scalar_type_t *t, void *x, void *a, void *b) { ; } \
	int    PREFIX##_cplx_rational   (scalar_type_t *t, void *x, int k, int n)     { return 0; } \
	int    PREFIX##_cplx_div        (scalar_type_t *t, void *x, void *a, void *b) { return 0; } \
	void   PREFIX##_cplx_abs        (scalar_type_t *t, void *x, void *a)          { ; } \
	int    PREFIX##_cplx_less       (scalar_type_t *t, void *a, void *b)          { return 0; } \
	void   PREFIX##_cplx_fprint     (scalar_type_t *t, FILE* f, void *a)          { ; } \
	void   PREFIX##_cplx_fprintgap  (scalar_type_t *t, FILE* f, void *a)          { ; } \
\
SCALAR_TYPE_DECL(PREFIX##_cplx, #PREFIX "_cplx", sizeof(T) * 2, PREFIX##_cplx_record);

/*
 * Extern definitions
 */
EXPORT scalar_type_t * scalar_find_type(char * name);

extern scalar_type_t * float_record;
extern scalar_type_t * double_record;
extern scalar_type_t * long_double_record;

extern scalar_type_t * float_cplx_record;
extern scalar_type_t * double_cplx_record;
extern scalar_type_t * long_double_cplx_record;

#endif

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
