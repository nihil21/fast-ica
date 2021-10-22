//
// Created by nihil on 08/10/21.
//

#include "../include/fast_ica.h"
#include "../include/linalg.h"
#include "../include/preprocessing.h"
#include "../include/utils.h"

/*
 * LogCosh function
 */
Tuple *logcosh_f(Matrix *x) {
    fp alpha = 1.f;
    Matrix *gx = new_mat(x->height, x->width);
    Matrix *gx_prime = new_mat(x->height, x->width);

    for (int i = 0; i < x->height; i++) {
        for (int j = 0; j < x->width; j++) {
            MAT_CELL(gx, i, j) = TANH(alpha * MAT_CELL(x, i, j));
            MAT_CELL(gx_prime, i, j) = alpha * (1 - MAT_CELL(gx, i, j) * MAT_CELL(gx, i, j));
        }
    }

    // Pack Gx and Gx' into a tuple
    Tuple *res = new_tuple(gx, gx_prime);
    return res;
}

/*
 * Exponential function
 */
Tuple *exp_f(Matrix *x) {
    Matrix *gx = new_mat(x->height, x->width);
    Matrix *gx_prime = new_mat(x->height, x->width);

    for (int i = 0; i < x->height; i++) {
        for (int j = 0; j < x->width; j++) {
            fp tmp = EXP(-MAT_CELL(x, i, j) * MAT_CELL(x, i, j) / 2);
            MAT_CELL(gx, i, j) = MAT_CELL(x, i, j) * tmp;
            MAT_CELL(gx_prime, i, j) = (1 - MAT_CELL(x, i, j) * MAT_CELL(x, i, j)) * tmp;
        }
    }

    // Pack Gx and Gx' into a tuple
    Tuple *res = new_tuple(gx, gx_prime);
    return res;
}

/*
 * Cube function
 */
Tuple *cube_f(Matrix *x) {
    Matrix *gx = new_mat(x->height, x->width);
    Matrix *gx_prime = new_mat(x->height, x->width);

    for (int i = 0; i < x->height; i++) {
        for (int j = 0; j < x->width; j++) {
            fp tmp = MAT_CELL(x, i, j) * MAT_CELL(x, i, j);
            MAT_CELL(gx, i, j) = MAT_CELL(x, i, j) * tmp;
            MAT_CELL(gx_prime, i, j) = 3 * tmp;
        }
    }

    // Pack Gx and Gx' into a tuple
    Tuple *res = new_tuple(gx, gx_prime);
    return res;
}

/*
 * Abs function
 */
Tuple *abs_f(Matrix *x) {
    Matrix *gx = new_mat(x->height, x->width);
    Matrix *gx_prime = new_mat(x->height, x->width);

    for (int i = 0; i < x->height; i++) {
        for (int j = 0; j < x->width; j++) {
            MAT_CELL(gx, i, j) = ABS(MAT_CELL(x, i, j));
            MAT_CELL(gx_prime, i, j) = (fp) sgn(MAT_CELL(x, i, j));
        }
    }

    // Pack Gx and Gx' into a tuple
    Tuple *res = new_tuple(gx, gx_prime);
    return res;
}

/*
 * Implement Gram-Schmidt decorrelation
 */
void gram_schmidt_decorrelation(Matrix **w_i_new, Matrix *w, int i) {
    if (i > 0) {
        Matrix *w_slice = read_slice(w, 0, i - 1, 0, w->width - 1);
        Matrix *tmp_mat = mat_mul_trans1(w_slice, w_slice);
        Matrix *tmp_vec = mat_mul(*w_i_new, tmp_mat);
        sub_mat_(w_i_new, tmp_vec);

        // Free memory
        free_mat(w_slice);
        free_mat(tmp_mat);
        free_mat(tmp_vec);
    }
}

/*
 * Implement symmetric decorrelation
 */
void symmetric_decorrelation(Matrix **w) {
    Matrix *w_wt = mat_mul_trans2(*w, *w);
    // Compute eigenvalues and eigenvectors
    Tuple *eigen = solve_eig(w_wt);
    Matrix *eig_vals = eigen->m1;  // column vector
    Matrix *eig_vecs = eigen->m2;
    int n = eig_vals->height;
    Matrix *d = new_mat(n, n);
    for (int i = 0; i < n; i++)
        MAT_CELL(d, i, i) = 1 / SQRT(MAT_CELL(eig_vals, i, 0));
    // Compute new weight matrix
    Matrix *tmp1 = mat_mul_trans1(eig_vecs, *w);
    Matrix *tmp2 = mat_mul(d, tmp1);
    free_mat(*w);
    *w = mat_mul(eig_vecs, tmp2);

    // Free memory
    free_mat(w_wt);
    free_mat(d);
    free_mat(tmp1);
    free_mat(tmp2);
    free_tuple(eigen, true);
}

/*
 * Implement FastICA deflationary strategy
 */
Matrix *ica_def(Matrix *x_w, Tuple *(*g_func)(Matrix *), fp threshold, int max_iter) {
    int n_units = x_w->height;
    int n_samples = x_w->width;

    // Initialize weights randomly
    Matrix *w = mat_randn(n_units, n_units);
    // Iterate over units
    for (int k = 0; k < n_units; k++) {
        // Initialize i-th neuron
        Matrix *w_k = extract_row(w, k);  // row vector
        scale_(&w_k, 1 / norm(w_k));

        for (int i = 0; i < max_iter; i++) {
            // (1, n_units) @ (n_units, n_samples) -> (1, n_samples)
            Matrix *ws = mat_mul(w_k, x_w);
            // Compute G_Ws and G_Ws'
            Tuple *res = g_func(ws);
            Matrix *g_ws = res->m1;  // (1, n_samples)
            Matrix *gw_s_prime = res->m2;  // (1, n_samples)

            // (1, n_samples) @ (n_units, n_samples).T -> (1, n_samples) @ (n_samples, n_units) -> (1, n_units)
            Matrix *a = mat_mul_trans2(g_ws, x_w);
            scale_(&a, 1 / (fp) n_samples);
            // (1, n_units) * E[(1, n_samples)] -> (1, n_units)
            Matrix *b = scale(w_k, mean(gw_s_prime));

            // Compute new weight
            Matrix *w_k_new = sub_mat(a, b);  // (1, n_units)
            // Decorrelate
            gram_schmidt_decorrelation(&w_k_new, w, k);
            // Normalize
            scale_(&w_k_new, 1 / norm(w_k_new));

            // Compute distance
            fp distance = ABS(dot(w_k_new, w_k) - 1.f);

            // Update weight
            free_mat(w_k);
            w_k = w_k_new;

            // Free memory
            free_mat(ws);
            free_mat(a);
            free_mat(b);
            free_tuple(res, true);

            if (distance < threshold)
                break;
        }
        // Save weight vector
        paste_row(&w, w_k, k);
        free_mat(w_k);
    }

    return w;
}

/*
 * Implement FastICA parallel strategy
 */
Matrix *ica_par(Matrix *x_w, Tuple *(*g_func)(Matrix *), fp threshold, int max_iter) {
    int n_units = x_w->height;
    int n_samples = x_w->width;

    // Initialize weights randomly and decorrelate
    Matrix *w = mat_randn(n_units, n_units);
    symmetric_decorrelation(&w);

    for (int i = 0; i < max_iter; i++) {
        // (n_units, n_units) @ (n_units, n_samples) -> (n_units, n_samples)
        Matrix *ws = mat_mul(w, x_w);
        // Compute G_Ws and G_Ws'
        Tuple *res = g_func(ws);
        Matrix *g_ws = res->m1;  // (n_units, n_samples)
        Matrix *g_ws_prime = res->m2;  // (n_units, n_samples)

        Matrix *a = new_mat(n_units, n_units);
        Matrix *b = new_mat(n_units, n_units);
        // Iterate over units
        for (int k = 0; k < n_units; k++) {
            // Extract k-th row from G_Ws
            Matrix *g_ws_k = extract_row(g_ws, k);  // row vector
            // (1, n_samples) @ (n_units, n_samples).T -> (1, n_samples) @ (n_samples, n_units) -> (1, n_units)
            Matrix *a_k = mat_mul_trans2(g_ws_k, x_w);
            scale_(&a_k, 1 / (fp) n_samples);

            // Extract k-th row from G_Ws'
            Matrix *g_ws_prime_k = extract_row(g_ws_prime, k);  // row vector
            // Extract k-th row from W
            Matrix *w_k = extract_row(w, k);  // row vector
            // (1, n_units) * E[(1, n_samples)] -> (1, n_units)
            Matrix *b_k = scale(w_k, mean(g_ws_prime_k));

            // Paste rows
            paste_row(&a, a_k, k);
            paste_row(&b, b_k, k);

            // Free memory
            free_mat(a_k);
            free_mat(b_k);
            free_mat(w_k);
            free_mat(g_ws_k);
            free_mat(g_ws_prime_k);
        }

        // Compute new weight
        Matrix *w_new = sub_mat(a, b);
        // Decorrelate
        symmetric_decorrelation(&w_new);

        // Compute distance
        fp distance = 0;
        Matrix *tmp1 = mat_mul_trans2(w_new, w);
        Matrix *tmp2 = diagonal(tmp1);
        for (int ii = 0; ii < tmp2->height; ii++) {
            fp cur_dis = ABS(ABS(MAT_CELL(tmp2, ii, 0)) - 1);
            if (cur_dis > distance)
                distance = cur_dis;
        }

        // Update weights
        free_mat(w);
        w = w_new;

        // Free memory
        free_mat(ws);
        free_mat(a);
        free_mat(b);
        free_mat(tmp1);
        free_mat(tmp2);
        free_tuple(res, true);

        if (distance < threshold)
            break;
    }

    return w;
}

/*
 * Perform FastICA on a (n_features, n_samples) matrix of observations
 */
Matrix *fast_ica(Matrix *x, bool whiten, FastICAStrategy strategy, GFunc g_func, fp threshold, int max_iter) {
    // Center and whiten, if specified
    Matrix *x_w;
    Matrix *white_mtx;
    Matrix *x_mean;
    if (whiten) {
        // Center
        Tuple *CenterData = center(x);
        Matrix *x_c = CenterData->m1;
        x_mean = CenterData->m2;

        // Whiten
        Tuple *WhitenData = whitening(x_c, false);
        x_w = WhitenData->m1;
        white_mtx = WhitenData->m2;

        // Free memory
        free_tuple(CenterData, false);
        free_tuple(WhitenData, false);
        free_mat(x_c);
    } else {
        x_w = x;
    }

    // Select non-quadratic function G
    Tuple *(*g)(Matrix *);
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
        case Abs:
            g = &abs_f;
            break;
        default:
            assert(false, "Unknown function");
    }

    Matrix *w;
    // Select strategy to estimate weight matrix
    switch (strategy) {
        case Parallel:
            // Execute parallel algorithm
            w = ica_par(x_w, g, threshold, max_iter);
            break;
        case Deflation:
            // Execute deflation algorithm
            w = ica_def(x_w, g, threshold, max_iter);
            break;
        default:
            assert(false, "Unknown strategy");
    }

    // Reconstruct signal
    Matrix *s = mat_mul(w, x_w);
    if (whiten) {
        // Free memory
        free_mat(x_w);
        free_mat(white_mtx);
        free_mat(x_mean);
    }

    // Free memory
    free_mat(w);

    return s;
}