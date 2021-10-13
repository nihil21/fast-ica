//
// Created by nihil on 03/10/21.
//

#ifndef FAST_ICA_VECTOR_H
#define FAST_ICA_VECTOR_H

#include <stdbool.h>
#include "fp.h"

/*
 * Data type declaration and macros
 */
typedef struct Vector {
    fp* data;
    int length;
} Vector;
#define VEC_DATA(v) (((Vector *) (v))->data)
#define VEC_CELL(v, i) (VEC_DATA(v)[i])

/*
 * Initialize and free
 */
Vector *new_vec(int length);
void free_vec(Vector *v);

/*
 * Predefined vectors
 */
Vector *vec_randint(int length, int amin, int amax);
Vector *vec_rand(int length, fp min, fp max);
Vector *vec_randn(int length);
Vector *linspace(fp start, fp stop, int n_samples);

/*
 * Basic self-operations (the trailing underscore means that the operation is performed in-place)
 */
fp vec_norm(Vector *v);
fp vec_mean(Vector *v);
Vector *vec_scale(Vector *v, fp scalar);
void vec_scale_(Vector **v, fp scalar);

/*
 * Basic operations between two vectors (the trailing underscore means that the operation is performed in-place)
 */
Vector *vec_add(Vector *v1, Vector *v2);
void vec_add_(Vector **v1, Vector *v2);
Vector *vec_sub(Vector *v1, Vector *v2);
void vec_sub_(Vector **v1, Vector *v2);
Vector *vec_scalar_add(Vector *v, fp scalar);
void vec_scalar_add_(Vector **v, fp scalar);
Vector *vec_elemwise_prod(Vector *v1, Vector *v2);
void vec_elemwise_prod_(Vector **v1, Vector *v2);
fp dot(Vector *v1, Vector *v2);

/*
 * Boolean operations
 */
bool vec_equal(Vector *v1, Vector *v2, fp tol);

/*
 * Utils
 */
void vec_print(Vector *v);
void vec_write(const char *path, Vector *v);
Vector *vec_from_array(const fp *data, int length);
Vector *vec_copy(Vector *v);

#endif //FAST_ICA_VECTOR_H
