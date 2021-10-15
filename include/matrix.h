//
// Created by nihil on 20/09/21.
//

#ifndef FAST_ICA_MATRIX_H
#define FAST_ICA_MATRIX_H

#include <stdbool.h>
#include "fp.h"

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
Matrix *new_vec(int length);  // column vector
void free_mat(Matrix *m);

/*
 * Predefined matrices
 */
Matrix *eye(int n);
Matrix *mat_randint(int height, int width, int amin, int amax);
Matrix *mat_rand(int height, int width, fp min, fp max);
Matrix *mat_randn(int height, int width);
Matrix *linspace(fp start, fp stop, int n_samples);

/*
 * Basic self-operations (the trailing underscore means that the operation is performed in-place)
 */
fp norm(Matrix *m);
fp mean(Matrix *m);
Matrix *row_mean(Matrix *m);
Matrix *col_mean(Matrix *m);
Matrix *transpose(Matrix *m);
Matrix *scale(Matrix *m, fp scalar);
void scale_(Matrix **m, fp scalar);

/*
 * Basic operations between two matrices (the trailing underscore means that the operation is performed in-place)
 */
Matrix *add_mat(Matrix *m1, Matrix *m2);
void add_mat_(Matrix **m1, Matrix *m2);
Matrix *add_row(Matrix *m, Matrix *r);
void add_row_(Matrix **m, Matrix *r);
Matrix *add_col(Matrix *m, Matrix *c);
void add_col_(Matrix **m, Matrix *c);
Matrix *sub_mat(Matrix *m1, Matrix *m2);
void sub_mat_(Matrix **m1, Matrix *m2);
Matrix *sub_row(Matrix *m, Matrix *r);
void sub_row_(Matrix **m, Matrix *r);
Matrix *sub_col(Matrix *m, Matrix *c);
void sub_col_(Matrix **m, Matrix *c);
Matrix *add_scalar(Matrix *m, fp scalar);
void add_scalar_(Matrix **m, fp scalar);
Matrix *hadamard(Matrix *m1, Matrix *m2);
void hadamard_(Matrix **m1, Matrix *m2);
Matrix *hadamard_row(Matrix *m, Matrix *r);
void hadamard_row_(Matrix **m, Matrix *r);
Matrix *hadamard_col(Matrix *m, Matrix *c);
void hadamard_col_(Matrix **m, Matrix *c);
Matrix *mat_mul(Matrix *m1, Matrix *m2);
Matrix *mat_mul_trans1(Matrix* m1, Matrix* m2);
Matrix *mat_mul_trans2(Matrix* m1, Matrix* m2);
fp dot(Matrix *v1, Matrix *v2);
Matrix *outer(Matrix *v1, Matrix *v2);

/*
 * Boolean operations
 */
bool is_vector(Matrix *m);
bool is_row_vector(Matrix *m);
bool is_col_vector(Matrix *m);
bool are_equal(Matrix *m1, Matrix *m2, fp tol);
bool is_square(Matrix *m);

/*
 * Matrix manipulation
 */
Matrix *diagonal(Matrix *m);
void tri_up(Matrix **m);
Matrix *read_slice(Matrix *m, int row_start, int row_stop, int col_start, int col_stop);
void write_slice(Matrix **m1, Matrix *m2, int row_start, int col_start);
Matrix *extract_row(Matrix *m, int k);
Matrix *extract_col(Matrix *m, int k);
void paste_row(Matrix **m, Matrix *r, int k);
void paste_col(Matrix **m, Matrix *c, int k);

/*
 * Utils
 */
void print_mat(Matrix *m);
void write_mat(const char *path, Matrix *m);
Matrix *from_array(const fp *data, int height, int width);
Matrix *copy_mat(Matrix *m);

#endif //FAST_ICA_MATRIX_H
