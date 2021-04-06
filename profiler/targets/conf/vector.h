/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#ifndef VECTOR_H_INCLUDED
#define VECTOR_H_INCLUDED

#include <stdio.h>
#ifndef NULL
#define NULL (void*)0
#endif

#include "sys.h"
#include "vector_def.h"

#define NTH(v,index) v->type->nth(v->type, v->data, index)

vector_t * vector_create        (scalar_type_t * type, int size);
void       vector_destroy       (vector_t * v);

EXPORT vector_t * vector_create_zero   (scalar_type_t * type, int size);
EXPORT vector_t * vector_create_random (scalar_type_t * type, int size);
void       vector_copy          (vector_t * result, vector_t * src);
void       vector_copy_tofix    (vector_t * result, vector_t * src);
void       vector_copy_todouble (vector_t * result, vector_t * src);
void       vector_basis         (vector_t * result, int idx);
void       vector_random        (vector_t * result);
void       vector_scalar_mul    (vector_t * result, vector_t * v,  scalar_t s);
void       vector_add           (vector_t * result, vector_t * v1, vector_t * v2);
void       vector_sub           (vector_t * result, vector_t * v1, vector_t * v2);
void       vector_compare       (scalar_t max_diff, vector_t * v1, vector_t * v2);
void       vector_print         (FILE * f, vector_t * v);
void       vector_print_side_by_side (FILE * f, vector_t * v1, vector_t * v2);

void     update_diff(scalar_type_t * typ, scalar_t old_diff, scalar_t new_diff);


#endif

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
