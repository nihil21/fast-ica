//
// Created by nihil on 08/10/21.
//

#include <stddef.h>
#include "../include/fast_ica.h"
#include "../include/linalg.h"
#include "../include/preprocessing.h"
#include "../include/utils.h"

/*
 * LogCosh function
 */
Tuple *logcosh_f(Vector *x) {
    fp alpha = 1.f;
    Vector *gx = new_vec(x->length);
    Vector *gx_prime = new_vec(x->length);

    for (int i = 0; i < x->length; i++) {
        VEC_CELL(gx, i) = TANH(alpha * VEC_CELL(x, i));
        VEC_CELL(gx_prime, i) = alpha * (1 - VEC_CELL(gx, i) * VEC_CELL(gx, i));
    }

    // Pack Gx and Gx' into a tuple
    Tuple *res = new_tuple((Tensor *) gx, VecType, (Tensor *) gx_prime, VecType);
    return res;
}

/*
 * Exponential function
 */
Tuple *exp_f(Vector *x) {
    Vector *gx = new_vec(x->length);
    Vector *gx_prime = new_vec(x->length);

    for (int i = 0; i < x->length; i++) {
        fp tmp = EXP(-VEC_CELL(x, i) * VEC_CELL(x, i) / 2);
        VEC_CELL(gx, i) = VEC_CELL(x, i) * tmp;
        VEC_CELL(gx_prime, i) = (1 - VEC_CELL(x, i) * VEC_CELL(x, i)) * tmp;
    }

    // Pack Gx and Gx' into a tuple
    Tuple *res = new_tuple((Tensor *) gx, VecType, (Tensor *) gx_prime, VecType);
    return res;
}

/*
 * Cube function
 */
Tuple *cube_f(Vector *x) {
    Vector *gx = new_vec(x->length);
    Vector *gx_prime = new_vec(x->length);

    for (int i = 0; i < x->length; i++) {
        fp tmp = VEC_CELL(x, i) * VEC_CELL(x, i);
        VEC_CELL(gx, i) = VEC_CELL(x, i) * tmp;
        VEC_CELL(gx_prime, i) = 3 * tmp;
    }

    // Pack Gx and Gx' into a tuple
    Tuple *res = new_tuple((Tensor *) gx, VecType, (Tensor *) gx_prime, VecType);
    return res;
}

/*
 * Implement Gram-Schmidt decorrelation
 */
void gram_schmidt_decorrelation(Vector **w_i_new, Matrix *w, int i) {
    if (i > 0) {
        Matrix *w_slice = read_slice(w, 0, i - 1, 0, w->width - 1);
        Matrix *tmp_mat = mat_mul_trans1(w_slice, w_slice);
        Vector *tmp_vec = mat_vec_mul_trans(tmp_mat, *w_i_new);
        vec_sub_(w_i_new, tmp_vec);

        // Free memory
        free_mat(w_slice);
        free_mat(tmp_mat);
        free_vec(tmp_vec);
    }
}

/*
 * Implement symmetric decorrelation
 */
void symmetric_decorrelation(Matrix **w) {
    Matrix *w_wt = mat_mul_trans2(*w, *w);
    // Compute eigenvalues and eigenvectors
    Tuple *eigen = solve_eig(w_wt);
    Vector *eig_vals = (Vector *) eigen->tensor1;
    Matrix *eig_vecs = (Matrix *) eigen->tensor2;
    int n = eig_vals->length;
    Matrix *d = new_mat(n, n);
    for (int i = 0; i < n; i++)
        MAT_CELL(d, i, i) = 1 / SQRT(VEC_CELL(eig_vals, i));
    // Compute new weight matrix
    Matrix *tmp1 = mat_mul_trans1(eig_vecs, *w);
    Matrix *tmp2 = mat_mul(d, tmp1);
    *w = mat_mul(eig_vecs, tmp2);

    // Free memory
    free_mat(w_wt);
    free_mat(tmp1);
    free_mat(tmp2);
    free_tuple(eigen, true);
}

/*
 * Implement FastICA deflationary strategy
 */
Matrix *ica_def(Matrix *x_w, Tuple *(*g_func)(Vector *), Matrix *w_init, fp threshold, int max_iter) {
    int n_units = x_w->height;

    // Output weights
    Matrix *w = new_mat(n_units, n_units);
    // Iterate over units
    for (int i = 0; i < n_units; i++) {
        // Initialize i-th neuron
        Vector *w_i = extract_row(w_init, i);
        vec_scale_(&w_i, 1 / vec_norm(w_i));

        for (int j = 0; j < max_iter; j++) {
            // (n_units,) @ (n_units, n_samples) -> (n_samples,)
            Vector *ws = mat_vec_mul_trans(x_w, w_i);
            // Compute G_Ws and G_Ws'
            Tuple *res = g_func(ws);
            Vector *g_ws = (Vector *) res->tensor1;  // (n_samples,)
            Vector *gw_s_prime = (Vector *) res->tensor2;  // (n_samples,)

            Matrix *tmp = mat_copy(x_w);  // (n_units, n_samples)
            // (n_samples,) * (n_units, n_samples) -> (n_units, n_samples)
            for (int ii = 0; ii < tmp->height; ii++) {
                for (int jj = 0; jj < tmp->width; jj++) {
                    MAT_CELL(tmp, ii, jj) *= VEC_CELL(g_ws, jj);
                }
            }
            // E[(n_units, n_samples)] -> (n_units,)
            Vector *a = col_mean(tmp);

            // E[(n_samples,)] * (n_units,) -> (,) * (n_units,) -> (n_units,)
            Vector *b = vec_scale(w_i, vec_mean(gw_s_prime));

            // Compute new weight
            Vector *w_i_new = vec_sub(a, b);
            // Decorrelate
            gram_schmidt_decorrelation(&w_i_new, w, i);
            // Normalize
            vec_scale_(&w_i_new, 1 / vec_norm(w_i_new));

            // Compute distance
            fp distance = ABS(dot(w_i_new, w_i) - 1.f);

            // Update weight
            free_vec(w_i);
            w_i = vec_copy(w_i_new);

            // Free memory
            free_vec(ws);
            free_vec(w_i_new);
            free_vec(a);
            free_vec(b);
            free_mat(tmp);
            free_tuple(res, true);

            if (distance < threshold)
                break;
        }
        // Save weight vector
        paste_row(&w, w_i, i);
        free_vec(w_i);
    }

    return w;
}

/*
 * Implement FastICA parallel strategy
 */
Matrix *ica_par(Matrix *x_w, Tuple *(*g_func)(Vector *), Matrix *w_init, fp threshold, int max_iter) {
    // Initialize weights and decorrelate
    Matrix *w = mat_copy(w_init);
    symmetric_decorrelation(&w);

    for (int i = 0; i < max_iter; i++) {
        // (n_units, n_units) @ (n_units, n_samples) -> (n_units, n_samples)
        Matrix *ws = mat_mul(w, x_w);
        assert(false, "TO BE IMPLEMENTED");
    }

    return w;
}

/*
 * Perform FastICA on a (n_features, n_samples) matrix of observations
 */
Matrix *fast_ica(Matrix *x, bool whiten, FastICAStrategy strategy, GFunc g_func, Matrix *w_init, fp threshold, int max_iter) {
    // Center and whiten, if specified
    Matrix *x_w;
    Vector *x_mean;
    if (whiten) {
        // Center
        Tuple *CenterData = center(x);
        Matrix *x_c = (Matrix *) CenterData->tensor1;
        x_mean = (Vector *) CenterData->tensor2;

        // Whiten
        x_w = whitening(x_c, false);

        // Free memory
        free_tuple(CenterData, false);
        free_mat(x_c);
    } else {
        x_w = x;
    }

    // Initialize weights
    int n_units = x_w->height;
    bool free_w_init = false;
    if (w_init == NULL) {
        w_init = mat_randn(n_units, n_units);
        free_w_init = true;
    }

    // Select non-quadratic function G
    Tuple *(*g)(Vector *);
    switch (g_func) {
        case LogCosh:
            g = &logcosh_f;
            break;
        case Exp:
            g = &exp_f;
            break;
        case Cube:
            g = &cube_f;
            break;
        default:
            assert(false, "Unknown function");
    }

    Matrix *w;
    // Select strategy to estimate weight matrix
    switch (strategy) {
        case Deflation:
            w = ica_def(x_w, g, w_init, threshold, max_iter);
            break;
        case Parallel:
            w = ica_par(x_w, g, w_init, threshold, max_iter);
            break;
        default:
            assert(false, "Unknown strategy.");
    }

    // Reconstruct signal
    Matrix *s = mat_mul(w, x_w);
    if (whiten) {
        Vector *s_mean = mat_vec_mul(w, x_mean);
        col_sub_(&s, s_mean);

        // Free memory
        free_mat(x_w);
        free_vec(x_mean);
        free_vec(s_mean);
    }

    // Free memory
    free_mat(w);
    if (free_w_init)
        free_mat(w_init);

    return s;
}