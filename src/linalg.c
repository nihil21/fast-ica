//
// Created by nihil on 20/09/21.
//

#include <stdlib.h>
#include "../include/linalg.h"
#include "../include/utils.h"

/*
 * Reduce the given matrix to Hessenberg form
 */
Matrix *to_hessenberg(Matrix *m) {
    assert(is_square(m), "Hessenberg transformation can only be applied to square matrices.");
    int n = m->height;
    Matrix *h = mat_copy(m);

    for (int k = 0; k < n - 1; k++) {
        Vector *vk = mat_to_vec(read_slice(h, k + 1, n - 1, k, k), true);
        VEC_CELL(vk, 0) += (fp) sgn(VEC_CELL(vk, 0)) * vec_norm(vk);
        vec_scale_(&vk, 1 / vec_norm(vk));

        Matrix *h_slice = read_slice(h, k + 1, n - 1, k, n - 1);
        Vector *tmp_vec = mat_vec_mul_trans(h_slice, vk);
        Matrix *tmp_mat = outer(vk, tmp_vec);
        mat_scale_(&tmp_mat, 2);
        mat_sub_(&h_slice, tmp_mat);
        write_slice(&h, h_slice, k + 1, k);
        // Free memory
        free_mat(h_slice);
        free_mat(tmp_mat);
        free_vec(tmp_vec);

        h_slice = read_slice(h, 0, n - 1, k + 1, n - 1);
        tmp_vec = mat_vec_mul(h_slice, vk);
        tmp_mat = outer(tmp_vec, vk);
        mat_scale_(&tmp_mat, 2);
        mat_sub_(&h_slice, tmp_mat);
        write_slice(&h, h_slice, 0, k + 1);
        // Free memory
        free_mat(h_slice);
        free_mat(tmp_mat);
        free_vec(tmp_vec);

        free_vec(vk);
    }

    return h;
}

/*
 * QR decomposition with shift and Hessenberg reduction
 */
Tuple *qr_decomposition(Matrix *m) {
    int n = m->height;
    Matrix *v = new_mat(n, n);
    for (int k = 0; k < n; k++) {
        Matrix *m_slice = read_slice(m, k, n - 1, k, k);
        write_slice(&v, m_slice, k, k);
        free_mat(m_slice);

        Vector *v_slice = mat_to_vec(read_slice(v, k, n - 1, k, k), true);
        MAT_CELL(v, k, k) += (fp) sgn(MAT_CELL(v, k, k)) * vec_norm(v_slice);
        free_vec(v_slice);

        v_slice = mat_to_vec(read_slice(v, k, n - 1, k, k), true);
        vec_scale_(&v_slice, 1 / vec_norm(v_slice));
        Matrix *v_slice_m = vec_to_mat(v_slice, false);
        write_slice(&v, v_slice_m, k, k);
        free(v_slice_m);

        m_slice = read_slice(m, k, n - 1, k, n - 1);
        Vector *tmp_vec = mat_vec_mul_trans(m_slice, v_slice);
        Matrix *tmp_mat = outer(v_slice, tmp_vec);
        mat_scale_(&tmp_mat, 2);
        mat_sub_(&m_slice, tmp_mat);
        write_slice(&m, m_slice, k, k);

        // Free memory
        free_vec(v_slice);
        free_mat(m_slice);
        free_vec(tmp_vec);
        free_mat(tmp_mat);
    }
    Matrix *r = mat_copy(m);

    Matrix *q = eye(n);
    for (int k = n - 1; k >= 0; k--) {
        Vector *v_slice = mat_to_vec(read_slice(v, k, n - 1, k, k), true);
        Matrix *tmp1 = outer(v_slice, v_slice);
        mat_scale_(&tmp1, 2);
        Matrix *q_slice = read_slice(q, k, n - 1, k, n - 1);
        Matrix *tmp2 = mat_mul(tmp1, q_slice);
        mat_sub_(&q_slice, tmp2);
        write_slice(&q, q_slice, k, k);

        // Free memory
        free_vec(v_slice);
        free_mat(q_slice);
        free_mat(tmp1);
        free_mat(tmp2);
    }

    free_mat(v);

    // Pack Q and R into tuple
    Tuple *qr = new_tuple((Tensor *) q, MatType, (Tensor *) r, MatType);

    return qr;
}

/*
 * Compute the eigenvalues of a square matrix by means of QR decomposition with shift and Hessenberg reduction
 */
Vector *solve_eig_vals(Matrix *m, fp tol, int max_iter) {
    // Tri-diagonalize matrix using Hessenberg
    Matrix *t = to_hessenberg(m);
    int n = t->height;
    Vector *eig_vals = new_vec(n);

    int k = n - 1;
    int i = 0;
    while (k > 0 && i < max_iter) {
        // Obtain the shift from the lower right corner of the matrix.
        Matrix *mu = eye(k + 1);
        mat_scale_(&mu, MAT_CELL(t, k, k));

        // Shift T matrix and perform QR on shifted matrix
        mat_sub_(&t, mu);
        Tuple *qr = qr_decomposition(t);
        Matrix *q = (Matrix *) qr->tensor1;
        Matrix *r = (Matrix *) qr->tensor2;

        // Multiply R*Q and shift back result
        free_mat(t);
        t = mat_mul(r, q);
        mat_add_(&t, mu);

        // Free memory
        free_mat(mu);
        free_tuple(qr, true);

        if (ABS(MAT_CELL(t, k, k - 1)) < tol) {
            VEC_CELL(eig_vals, k) = MAT_CELL(t, k, k);
            Matrix *tmp = read_slice(t, 0, k - 1, 0, k - 1);
            // Free T data field
            free(t->data);
            // Manually change T fields to Tmp
            t->height = tmp->height;
            t->width = tmp->width;
            t->data = tmp->data;
            // Free Tmp struct (but not data field)
            free(tmp);

            k--;
        }
        i++;
    }

    VEC_CELL(eig_vals, 0) = MAT_CELL(t, 0, 0);

    free_mat(t);

    return eig_vals;
}

/*
 * Compute the eigenvector associated to an eigenvalue by means of inverse iteration
 */
Vector *inv_iter(fp eig_val, Matrix *m, fp tol, int max_iter) {
    // Perturb lambda to prevent the computed matrix from becoming singular
    fp lambda = eig_val + (fp) drand48() * 1e-6f;
    // Compute M' = M - lambda * I
    Matrix *lambda_i = eye(m->height);
    mat_scale_(&lambda_i, lambda);
    Matrix *m_prime = mat_sub(m, lambda_i);

    // Initialize vector randomly
    Vector *eig_vec = vec_randn(m->height);
    Vector *prev;

    int i = 0;
    do {
        if (i > 0)
            free_vec(prev);

        // Save previous estimate and compute the new one
        prev = vec_copy(eig_vec);
        free_vec(eig_vec);
        eig_vec = lin_solve(m_prime, prev);

        // If the first entry of the current estimate is negative,
        // swap the sign to improve convergence
        if (VEC_CELL(eig_vec, 0) < 0) {
            vec_scale_(&eig_vec, -1);
        }

        // Normalize estimate
        fp v_norm = vec_norm(eig_vec);
        vec_scale_(&eig_vec, 1 / v_norm);

        i++;
    } while ((!vec_equal(eig_vec, prev, tol)) && i < max_iter);

    free_mat(lambda_i);
    free_mat(m_prime);
    free_vec(prev);

    return eig_vec;
}

/*
 * Compute the eigenvectors of a square matrix given the eigenvalues
 */
Matrix *solve_eig_vecs(Vector *eig_vals, Matrix *m, fp tol, int max_iter) {
    Matrix *eig_vecs = new_mat(m->height, m->width);
    // Iterate over eigenvalues
    for (int i = 0; i < eig_vals->length; i++) {
        // Extract current eigenvalue
        fp eig_val = VEC_CELL(eig_vals, i);
        // Compute current eigenvector
        Vector *eig_vec = inv_iter(eig_val, m, tol, max_iter);
        // Insert it into i-th column of output matrix
        paste_col(&eig_vecs, eig_vec, i);
        free_vec(eig_vec);
    }

    return eig_vecs;
}

/*
 * Compute the eigenvalues and eigenvectors of a square matrix by means of QR decomposition
 */
Tuple *solve_eig(Matrix *m) {
    fp tol = 1e-10f;
    int max_iter = 3000;
    Vector *eig_vals = solve_eig_vals(m, tol, max_iter);
    Matrix *eig_vecs = solve_eig_vecs(eig_vals, m, tol, max_iter);

    // Pack eigenvalues and eigenvectors into tuple
    Tuple *eigen = new_tuple((Tensor *) eig_vals, VecType, (Tensor *) eig_vecs, MatType);

    return eigen;
}

/*
 * Solve a linear system Ux = y, where U is upper triangular, by means of back-substitution
 */
Vector *back_substitution(Matrix *u, Vector *y) {
    int n_eq = y->length;
    Vector *x = new_vec(n_eq);  // column vector
    // Iterate over the cells of y backwards
    for (int i = n_eq - 1; i >= 0; i--) {
        fp back_substitute = 0;
        // Iterate over the subsequent cells and accumulate back substitutions
        for (int j = i + 1; j < n_eq; j++) {
            back_substitute += VEC_CELL(x, j) * MAT_CELL(u, i, j);
        }
        // Compute i-th solution
        VEC_CELL(x, i) = (VEC_CELL(y, i) - back_substitute) / MAT_CELL(u, i, i);
    }

    return x;
}

/*
 * Solve a linear system Ax = b by means of QR decomposition:
 * A = QR => QRx = b and, thanks to the orthogonality of Q, Rx = Q.Tb => Rx = y
 */
Vector *lin_solve(Matrix *a, Vector *b) {
    // Decompose matrix A using QR
    Tuple *qr = qr_decomposition(a);
    Matrix *q = (Matrix *) qr->tensor1;
    Matrix *r = (Matrix *) qr->tensor2;

    // Multiply the transpose of Q and b
    Vector *y = mat_vec_mul_trans(q, b);
    // Solve Rx = y by means of back-substitution
    Vector *x = back_substitution(r, y);

    // Free memory
    free_tuple(qr, true);
    free_vec(y);

    return x;
}

/*
 * Given a (n_features, n_samples) matrix, compute the (n_features, n_features) covariance matrix
 */
Matrix *covariance(Matrix *x, bool center_data) {
    int n_samples = x->width;
    Matrix *x_cov;

    if (center_data) {  // center data first
        // Compute mean and center data
        Vector *x_mean = col_mean(x);
        Matrix *x_c = col_sub(x, x_mean);

        // Compute covariance matrix
        x_cov = mat_mul_trans2(x_c, x_c);  // transpose handled internally for efficiency reasons

        // Free memory
        free_mat(x_c);
        free_vec(x_mean);
    } else {  // data already centered
        // Compute covariance matrix
        x_cov = mat_mul_trans2(x, x);  // transpose handled internally for efficiency reasons
    }

    // Normalize
    fp fact = 1 / ((fp) n_samples - 1);
    mat_scale_(&x_cov, fact);

    return x_cov;
}