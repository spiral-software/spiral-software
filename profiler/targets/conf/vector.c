/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
/**************************************************************************
*  vector.c								  *
*									  *
*  Routines that manipulate vectors of fixed size (SIZE global variable)  *
*  Each function is prefixed with `real' or `cplx' to specify data type   *
*  of vector elements.							  *
*  `real' - real data							  *
*  `cplx' - interleaved complex data					  *
*									  *
*  Routines that do not have prefix, call the corresponding data-type 	  *
*  specific routine to do the task. Dispatch is based on SPL_data_type	  *
*  global variable.							  *
***************************************************************************/

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "vector.h"

/* 
 * MIPS convention for arguments:
 * <dest>, [src1], [src2]
*/

vector_t * vector_create(scalar_type_t * type, int size) 
{
    vector_t * result;

    result = (vector_t *) xmalloc(sizeof(vector_t));
    result->data = ALLOC(type, size);
    result->size = size;
    result->type = type;

    return result;
}

void     vector_destroy     (vector_t * v) {
    FREE(v->type, v->data);
    xfree(v);
}

EXPORT vector_t * vector_create_zero(scalar_type_t * type, int size) {
    vector_t * result = vector_create(type, size);
    int i;
    for(i = 0; i < size; i++)
	ZERO(type, NTH(result,i) );
    return result;
}

EXPORT vector_t * vector_create_random (scalar_type_t * type, int size) {
    vector_t * result;
	result = vector_create(type, size);
    vector_random(result);
    return result;
}

void     vector_copy(vector_t * result, vector_t * v) {
    int i;
    assert(result->type == v->type && result->size == v->size);
    for(i = 0; i < v->size; i++)
	SET(v->type, NTH(result,i), NTH(v,i) );
}

void     vector_copy_tofix(vector_t * result, vector_t * v) {
    int i;
    for(i = 0; i < v->size; i++)
	SET_FIX(result->type, NTH(result,i), NTH(v,i) );
}

void     vector_copy_todouble(vector_t * result, vector_t * v) {
    int i;
    for(i = 0; i < v->size; i++)
	SET_DOUBLE(v->type, NTH(result,i), NTH(v,i) );
}

void     vector_basis       (vector_t * result, int idx) {
    int i;
    assert(idx >= 0 && idx < result->size);
    for(i = 0; i < result->size; i++) {
	if(i == idx)
	    ONE(result->type, NTH(result,i) );
	else
	    ZERO(result->type, NTH(result,i) );
    }
}

void     vector_scalar_mul  (vector_t * result, vector_t * v, scalar_t s) {
    int i;
    assert(result->type == v->type && result->size == v->size);
    for(i = 0; i < result->size; i++)
	MUL(result->type, NTH(result,i), NTH(v,i), s );
}

void     vector_random      (vector_t * result) {
    int i;
    for(i = 0; i < result->size; i++)
	RANDOM(result->type, NTH(result,i) );
}

void     vector_add         (vector_t * result, vector_t * v1, vector_t * v2) {
    int i;
    assert(result->type == v1->type && v1->type == v2->type && 
	   result->size == v1->size && v1->size == v2->size);
    for(i = 0; i < result->size; i++)
	ADD(result->type, NTH(result,i), NTH(v1,i), NTH(v2,i) );
}

void     vector_sub         (vector_t * result, vector_t * v1, vector_t * v2) {
    int i;
    assert(result->type == v1->type && v1->type == v2->type && 
	   result->size == v1->size && v1->size == v2->size);
    for(i = 0; i < result->size; i++)
	SUB(result->type, NTH(result,i), NTH(v1,i), NTH(v2,i) );
}

void     vector_compare     (scalar_t max_diff, vector_t * v1, vector_t * v2) {
    int i;
    scalar_t diff;
    assert(v1->type == v2->type && v1->size == v2->size);
    ZERO(v1->type, max_diff);
    diff = ALLOC(v1->type, 1);

    for(i = 0; i < v1->size; i++) {
	SUB(v1->type, diff, NTH(v1,i), NTH(v2,i));
	ABS(v1->type, diff, diff);
	if(LESS(v1->type, max_diff, diff)) 
	    SET(v1->type, max_diff, diff);
    }

    FREE(v1->type, diff);
}

void     vector_print       (FILE * f, vector_t * v) {
    int i;
    for(i = 0; i < v->size; i++) {
	PRINT(v->type,  f, NTH(v,i) );
	fprintf(f, "\n");
    }
}
void     vector_print_side_by_side (FILE * f, vector_t * v1, vector_t * v2) {
    int i;
    scalar_t diff = ALLOC(v1->type, 1);
    for(i = 0; i < v1->size || i < v2->size; i++) {
	if(i < v1->size) PRINT(v1->type, f, NTH(v1,i));
	fprintf(f, "\t");
	if(i < v2->size) PRINT(v2->type, f, NTH(v2,i));
	fprintf(f, "\t");
	if(i < v1->size && i < v2->size) {
	    SUB(v1->type,  diff, NTH(v1,i), NTH(v2,i) );
	    fprintf(f, "diff=");
	    PRINT(v1->type, f, diff);
	}
	fprintf(f, "\n");
    }
    FREE(v1->type, diff);
}

/* Other */
void     update_diff(scalar_type_t * t, scalar_t old_diff, scalar_t new_diff) {
    if( LESS(t, old_diff, new_diff) )
	SET(t, old_diff, new_diff);
}

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
