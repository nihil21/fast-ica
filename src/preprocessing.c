//
// Created by nihil on 02/10/21.
//

#include <malloc.h>
#include "../include/preprocessing.h"
#include "../include/linalg.h"
#include "../include/sorting.h"

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

void eigen_sort(Matrix *eig_vals, Matrix *eig_vecs) {
    int n_eig = eig_vals->height;
    // Sort eigenvalues in descending order
    int* sort_idx = quick_sort(eig_vals->data, n_eig, true);

    // Reorder eig_vecs according to sort_idx
    Matrix *eig_vecs_copy = copy_mat(eig_vecs);
    for (int i = 0; i < n_eig; i++) {
        int idx = sort_idx[i];
        Matrix *eig_vec = extract_col(eig_vecs_copy, idx);
        write_slice(&eig_vecs, eig_vec, 0, i);

        free_mat(eig_vec);
    }

    // Free memory
    free(sort_idx);
    free_mat(eig_vecs_copy);
}

Tuple *whitening(Matrix *x, bool center_data, int n_components) {
    // 1. Compute covariance matrix
    Matrix *x_cov = covariance(x, center_data);

    // 2. Compute eigenvalues and eigenvectors
    Tuple *eigen = solve_eig(x_cov);
    Matrix *eig_vals = eigen->m1;  // column vector
    Matrix *eig_vecs = eigen->m2;

    // Sort eigenvalues and eigenvectors
    eigen_sort(eig_vals, eig_vecs);

    int n = eig_vals->height;
    Matrix *d = new_mat(n, n);
    for (int i = 0; i < n; i++)
        MAT_CELL(d, i, i) = 1 / SQRT(MAT_CELL(eig_vals, i, 0));

    // 3. Compute whitening matrix
    Matrix *tmp = mat_mul_trans2(d, eig_vecs);
    Matrix *white_mat = read_slice(tmp, 0, n_components - 1, 0, tmp->width - 1);

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