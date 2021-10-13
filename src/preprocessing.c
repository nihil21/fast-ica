//
// Created by nihil on 02/10/21.
//

#include "../include/preprocessing.h"
#include "../include/linalg.h"

/*
 * Given a (n_features, n_samples) matrix, compute its mean along columns and center the matrix
 */
Tuple *center(Matrix *x) {
    // Compute mean and subtract it from the original data to center them
    Vector *x_mean = col_mean(x);
    Matrix *x_c = col_sub(x, x_mean);
    // Pack Xc and Xm into tuple
    Tuple *CenterData = new_tuple((Tensor *) x_c, MatType, (Tensor *) x_mean, VecType);
    return CenterData;
}

Matrix *whitening(Matrix *x, bool center_data) {
    // 1. Compute covariance matrix
    Matrix *x_cov = covariance(x, center_data);

    // 2. Compute eigenvalues and eigenvectors
    Tuple *eigen = solve_eig(x_cov);
    Vector *eig_vals = (Vector *) eigen->tensor1;
    Matrix *eig_vecs = (Matrix *) eigen->tensor2;
    int n = eig_vals->length;
    Matrix *d = new_mat(n, n);
    for (int i = 0; i < n; i++)
        MAT_CELL(d, i, i) = 1 / SQRT(VEC_CELL(eig_vals, i));

    // 3. Compute whitening matrix
    Matrix *tmp = mat_mul_trans2(d, eig_vecs);
    Matrix *white_mat = mat_mul(eig_vecs, tmp);

    // 4. Whiten data
    Matrix *x_w = mat_mul(white_mat, x);

    // Free memory
    free_mat(x_cov);
    free_mat(d);
    free_mat(tmp);
    free_mat(white_mat);
    free_tuple(eigen, true);

    return x_w;
}