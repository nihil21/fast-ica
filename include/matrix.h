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
Matrix *mat_randint(int height, int width, int min, int max);
Matrix *mat_rand(int height, int width, fp min, fp max);
Matrix *mat_randn(int height, int width);
Matrix *linspace(fp start, fp stop, int n_samples);

/*
 * Basic self-operations (the trailing underscore means that the operation is performed in-place)
 */
fp norm(const Matrix *m);
Matrix *row_norm(const Matrix *m);
Matrix *col_norm(const Matrix *m);
fp mean(const Matrix *m);
Matrix *row_mean(const Matrix *m);
Matrix *col_mean(const Matrix *m);
fp std(const Matrix *m);
Matrix *row_std(const Matrix *m);
Matrix *col_std(const Matrix *m);
Matrix *transpose(const Matrix *m);

/*
 * Basic operations between two matrices (the trailing underscore means that the operation is performed in-place)
 */
Matrix *add_scalar(const Matrix *m, fp scalar);
void add_scalar_(const Matrix *m, fp scalar);
Matrix *scale(const Matrix *m, fp scalar);
void scale_(const Matrix *m, fp scalar);
Matrix *add_mat(const Matrix *m1, const Matrix *m2);
void add_mat_(const Matrix *m1, const Matrix *m2);
Matrix *add_row(const Matrix *m, const Matrix *r);
void add_row_(const Matrix *m, const Matrix *r);
Matrix *add_col(const Matrix *m, const Matrix *c);
void add_col_(const Matrix *m, const Matrix *c);
Matrix *sub_mat(const Matrix *m1, const Matrix *m2);
void sub_mat_(const Matrix *m1, const Matrix *m2);
Matrix *sub_row(const Matrix *m, const Matrix *r);
void sub_row_(const Matrix *m, const Matrix *r);
Matrix *sub_col(const Matrix *m, const Matrix *c);
void sub_col_(const Matrix *m, const Matrix *c);
Matrix *hadamard(const Matrix *m1, const Matrix *m2);
void hadamard_(const Matrix *m1, const Matrix *m2);
Matrix *hadamard_row(const Matrix *m, const Matrix *r);
void hadamard_row_(const Matrix *m, const Matrix *r);
Matrix *hadamard_col(const Matrix *m, const Matrix *c);
void hadamard_col_(const Matrix *m, const Matrix *c);
Matrix *div_mat(const Matrix *m1, const Matrix *m2);
void div_mat_(const Matrix *m1, const Matrix *m2);
Matrix *div_row(const Matrix *m, const Matrix *r);
void div_row_(const Matrix *m, const Matrix *r);
Matrix *div_col(const Matrix *m, const Matrix *c);
void div_col_(const Matrix *m, const Matrix *c);
Matrix *mat_mul(const Matrix *m1, const Matrix *m2);
Matrix *mat_mul_trans1(Matrix* m1, Matrix* m2);
Matrix *mat_mul_trans2(Matrix* m1, Matrix* m2);
fp dot(const Matrix *v1, const Matrix *v2);
Matrix *outer(const Matrix *v1, const Matrix *v2);

/*
 * Boolean operations
 */
bool is_vector(const Matrix *m);
bool is_row_vector(const Matrix *m);
bool is_col_vector(const Matrix *m);
bool are_equal(const Matrix *m1, const Matrix *m2, fp tol);
bool is_square(const Matrix *m);

/*
 * Matrix manipulation
 */
Matrix *diagonal(const Matrix *m);
void tri_up(const Matrix *m);
Matrix *read_slice(const Matrix *m, int row_start, int row_stop, int col_start, int col_stop);
void write_slice(const Matrix *m1, const Matrix *m2, int row_start, int col_start);
Matrix *extract_row(const Matrix *m, int k);
Matrix *extract_col(const Matrix *m, int k);
void paste_row(const Matrix *m, const Matrix *r, int k);
void paste_col(const Matrix *m, const Matrix *c, int k);

/*
 * Utils
 */
void print_mat(const Matrix *m);
void write_mat(const char *path, const Matrix *m);
Matrix *from_array(const fp *data, int height, int width);
Matrix *copy_mat(const Matrix *m);

#endif //FAST_ICA_MATRIX_H
