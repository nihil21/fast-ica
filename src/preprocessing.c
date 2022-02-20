//
// Created by nihil on 02/10/21.
//

#include "../include/preprocessing.h"
#include "../include/linalg.h"
#include <stdio.h>

/*
 * Given a (n_features, n_samples) matrix, compute its mean along columns and center the matrix
 */
Pair *center(Matrix *x) {
    // Compute mean and subtract it from the original data to center them
    Matrix *x_mean = col_mean(x);
    Matrix *x_c = sub_col(x, x_mean);
    // Pack Xc and Xm into tuple
    Pair *CenterData = new_pair(x_c, x_mean);
    return CenterData;
}

Pair *whitening(Matrix *x, bool center_data) {
    // 1. Compute covariance matrix
    Matrix *x_cov = covariance(x, center_data);

    // 2. Compute eigenvalues and eigenvectors
    Triplet *svd_ret = svd(x_cov);
    free_mat(x_cov);
    Matrix *u = svd_ret->m1;
    Matrix *s = svd_ret->m2;
    Matrix *vt = svd_ret->m3;

    int n_eig = s->height;
    Matrix *d = new_mat(n_eig, n_eig);
    for (int i = 0; i < n_eig; i++) {
        MAT_CELL(d, i, i) = 1 / SQRT(MAT_CELL(s, i, 0));
    }
    // 3. Compute whitening matrix
    Matrix *tmp = mat_mul(d, vt);
    free_mat(d);
    Matrix *white_mat = mat_mul(u, tmp);
    free_mat(tmp);

    // 4. Whiten data
    Matrix *x_w = mat_mul(white_mat, x);

    // Free memory
    free_triplet(svd_ret, true);

    // Pack Xw and whitening matrix into tuple
    Pair *WhitenData = new_pair(x_w, white_mat);
    return WhitenData;
}