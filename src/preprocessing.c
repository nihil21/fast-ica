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
    Matrix *x_mean = col_mean(x);
    Matrix *x_c = sub_col(x, x_mean);
    // Pack Xc and Xm into tuple
    Tuple *CenterData = new_tuple(x_c, x_mean);
    return CenterData;
}

Tuple *whitening(Matrix *x, bool center_data) {
    // 1. Compute covariance matrix
    Matrix *x_cov = covariance(x, center_data);

    // 2. Compute eigenvalues and eigenvectors
    Tuple *eigen = solve_eig(x_cov);
    Matrix *eig_vals = eigen->m1;  // column vector
    Matrix *eig_vecs = eigen->m2;
    int n = eig_vals->height;
    Matrix *d = new_mat(n, n);
    for (int i = 0; i < n; i++)
        MAT_CELL(d, i, i) = 1 / SQRT(MAT_CELL(eig_vals, i, 0));

    // 3. Compute whitening matrix
    Matrix *tmp = mat_mul_trans2(d, eig_vecs);
    Matrix *white_mat = mat_mul(eig_vecs, tmp);

    // 4. Whiten data
    Matrix *x_w = mat_mul(white_mat, x);

    // Free memory
    free_mat(x_cov);
    free_mat(d);
    free_mat(tmp);
    free_tuple(eigen, true);

    // Pack Xw and whitening matrix into tuple
    Tuple *WhitenData = new_tuple(x_w, white_mat);
    return WhitenData;
}