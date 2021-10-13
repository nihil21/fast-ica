//
// Created by nihil on 20/09/21.
//

#ifndef FAST_ICA_MATRIX_H
#define FAST_ICA_MATRIX_H

#include <stdbool.h>
#include "fp.h"
#include "vector.h"

/*
 * Data type declaration and macros
 */
typedef struct Matrix {
    fp* data;
    int height;
    int width;
} Matrix;
#define MAT_DATA(m) (((Matrix *) (m))->data)
#define MAT_IDX(m, i, j) (((i) * ((m)->width)) + (j))
#define MAT_CELL(m, i, j) (MAT_DATA(m)[MAT_IDX(m, i, j)])

/*
 * Initialize and free
 */
Matrix *new_mat(int height, int width);
void free_mat(Matrix *m);

/*
 * Predefined matrices
 */
Matrix *eye(int n);
Matrix *mat_randint(int height, int width, int amin, int amax);
Matrix *mat_rand(int height, int width, fp min, fp max);
Matrix *mat_randn(int height, int width);

/*
 * Basic self-operations (the trailing underscore means that the operation is performed in-place)
 */
fp mat_norm(Matrix *m);
fp mat_mean(Matrix *m);
Vector *row_mean(Matrix *m);
Vector *col_mean(Matrix *m);
Matrix *transpose(Matrix *m);
Matrix *mat_scale(Matrix *m, fp scalar);
void mat_scale_(Matrix **m, fp scalar);

/*
 * Basic operations between two matrices (the trailing underscore means that the operation is performed in-place)
 */
Matrix *mat_add(Matrix *m1, Matrix *m2);
void mat_add_(Matrix **m1, Matrix *m2);
Matrix *mat_sub(Matrix *m1, Matrix *m2);
void mat_sub_(Matrix **m1, Matrix *m2);
Matrix *mat_scalar_add(Matrix *m, fp scalar);
void mat_scalar_add_(Matrix **m, fp scalar);
Matrix *mat_hadamard(Matrix *m1, Matrix *m2);
void mat_hadamard_(Matrix **m1, Matrix *m2);
Matrix *mat_mul(Matrix *m1, Matrix *m2);
Matrix *mat_mul_trans1(Matrix* m1, Matrix* m2);
Matrix *mat_mul_trans2(Matrix* m1, Matrix* m2);

/*
 * Boolean operations
 */
bool is_square(Matrix *m);

/*
 * Matrix and vector manipulation and operations
 */
Vector *diagonal(Matrix *m);
Matrix *read_slice(Matrix *m, int row_start, int row_stop, int col_start, int col_stop);
void write_slice(Matrix **m1, Matrix *m2, int row_start, int col_start);
Vector *mat_to_vec(Matrix *m, bool free_struct);
Matrix *vec_to_mat(Vector *v, bool free_struct);
Vector *extract_row(Matrix *m, int k);
Vector *extract_col(Matrix *m, int k);
void paste_row(Matrix **m, Vector *r, int k);
void paste_col(Matrix **m, Vector *c, int k);
Matrix *row_add(Matrix *m, Vector *v);
void row_add_(Matrix **m, Vector *v);
Matrix *col_add(Matrix *m, Vector *v);
void col_add_(Matrix **m, Vector *v);
Matrix *row_sub(Matrix *m, Vector *v);
void row_sub_(Matrix **m, Vector *v);
Matrix *col_sub(Matrix *m, Vector *v);
void col_sub_(Matrix **m, Vector *v);
Vector *mat_vec_mul(Matrix *m, Vector *v);
Vector *mat_vec_mul_trans(Matrix *m, Vector *v);
Matrix *outer(Vector *v1, Vector *v2);

/*
 * Utils
 */
void mat_print(Matrix *m);
void mat_write(const char *path, Matrix *m);
Matrix *mat_from_array(const fp *data, int height, int width);
Matrix *mat_copy(Matrix *m);

#endif //FAST_ICA_MATRIX_H
