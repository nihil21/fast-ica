//
// Created by nihil on 08/10/21.
//

#include "../include/fast_ica.h"
#include "../include/linalg.h"
#include "../include/preprocessing.h"
#include "../include/utils.h"

/*
 * LogCosh function (for vectors)
 */
Tuple *logcosh_vec(Vector *x) {
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
 * Exponential function (for vectors)
 */
Tuple *exp_vec(Vector *x) {
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
 * Cube function (for vectors)
 */
Tuple *cube_vec(Vector *x) {
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
 * LogCosh function (for matrices)
 */
Tuple *logcosh_mat(Matrix *x) {
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
    Tuple *res = new_tuple((Tensor *) gx, MatType, (Tensor *) gx_prime, MatType);
    return res;
}

/*
 * Exponential function (for matrices)
 */
Tuple *exp_mat(Matrix *x) {
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
    Tuple *res = new_tuple((Tensor *) gx, MatType, (Tensor *) gx_prime, MatType);
    return res;
}

/*
 * Cube function (for matrices)
 */
Tuple *cube_mat(Matrix *x) {
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
    Tuple *res = new_tuple((Tensor *) gx, MatType, (Tensor *) gx_prime, MatType);
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
Matrix *ica_def(Matrix *x_w, Tuple *(*g_func)(Vector *), fp threshold, int max_iter) {
    int n_units = x_w->height;
    int n_samples = x_w->width;

    // Initialize weights randomly
    Matrix *w = mat_randn(n_units, n_units);
    // Iterate over units
    for (int k = 0; k < n_units; k++) {
        // Initialize i-th neuron
        Vector *w_k = extract_row(w, k);
        vec_scale_(&w_k, 1 / vec_norm(w_k));

        for (int i = 0; i < max_iter; i++) {
            // (n_units,) @ (n_units, n_samples) -> (n_samples,)
            Vector *ws = mat_vec_mul_trans(x_w, w_k);
            // Compute G_Ws and G_Ws'
            Tuple *res = g_func(ws);
            Vector *g_ws = (Vector *) res->tensor1;  // (n_samples,)
            Vector *gw_s_prime = (Vector *) res->tensor2;  // (n_samples,)

            // (n_units, n_samples) @ (n_samples,) -> (n_units,)
            Vector *a = mat_vec_mul(x_w, g_ws);
            vec_scale_(&a, 1 / (fp) n_samples);
            // (n_units,) * E[(n_samples,)] -> (n_units,)
            Vector *b = vec_scale(w_k, vec_mean(gw_s_prime));

            // Compute new weight
            Vector *w_k_new = vec_sub(a, b);
            // Decorrelate
            gram_schmidt_decorrelation(&w_k_new, w, k);
            // Normalize
            vec_scale_(&w_k_new, 1 / vec_norm(w_k_new));

            // Compute distance
            fp distance = ABS(dot(w_k_new, w_k) - 1.f);

            // Update weight
            free_vec(w_k);
            w_k = w_k_new;

            // Free memory
            free_vec(ws);
            free_vec(a);
            free_vec(b);
            free_tuple(res, true);

            if (distance < threshold)
                break;
        }
        // Save weight vector
        paste_row(&w, w_k, k);
        free_vec(w_k);
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
        Matrix *g_ws = (Matrix *) res->tensor1;  // (n_units, n_samples)
        Matrix *g_ws_prime = (Matrix *) res->tensor2;  // (n_units, n_samples)

        Matrix *a = new_mat(n_units, n_units);
        Matrix *b = new_mat(n_units, n_units);
        // Iterate over units
        for (int k = 0; k < n_units; k++) {
            // Extract k-th row from G_Ws
            Vector *g_ws_k = extract_row(g_ws, k);
            // (n_units, n_samples) @ (n_samples,) -> (n_units,)
            Vector *a_k = mat_vec_mul(x_w, g_ws_k);
            vec_scale_(&a_k, 1 / (fp) n_samples);

            // Extract k-th row from G_Ws'
            Vector *g_ws_prime_k = extract_row(g_ws_prime, k);
            // Extract k-th row from W
            Vector *w_k = extract_row(w, k);
            // (n_units,) * E[(n_samples,)] -> (n_units,)
            Vector *b_k = vec_scale(w_k, vec_mean(g_ws_prime_k));

            // Paste rows
            paste_row(&a, a_k, k);
            paste_row(&b, b_k, k);

            // Free memory
            free_vec(a_k);
            free_vec(b_k);
            free_vec(w_k);
            free_vec(g_ws_k);
            free_vec(g_ws_prime_k);
        }

        // Compute new weight
        Matrix *w_new = mat_sub(a, b);
        // Decorrelate
        symmetric_decorrelation(&w_new);

        // Compute distance
        fp distance = 0;
        Matrix *tmp1 = mat_mul_trans2(w_new, w);
        Vector *tmp2 = diagonal(tmp1);
        for (int ii = 0; ii < tmp2->length; ii++) {
            fp cur_dis = ABS(ABS(VEC_CELL(tmp2, ii)) - 1);
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
        free_vec(tmp2);
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
    Vector *x_mean;
    if (whiten) {
        // Center
        Tuple *CenterData = center(x);
        Matrix *x_c = (Matrix *) CenterData->tensor1;
        x_mean = (Vector *) CenterData->tensor2;

        // Whiten
        Tuple *WhitenData = whitening(x_c, false);
        x_w = (Matrix *) WhitenData->tensor1;
        white_mtx = (Matrix *) WhitenData->tensor2;

        // Free memory
        free_tuple(CenterData, false);
        free_tuple(WhitenData, false);
        free_mat(x_c);
    } else {
        x_w = x;
    }

    Matrix *w;
    // Select strategy to estimate weight matrix
    if (strategy == Deflation) {
        // Select non-quadratic function G
        Tuple *(*g)(Vector *);
        switch (g_func) {
            case LogCosh:
                g = &logcosh_vec;
                break;
            case Exp:
                g = &exp_vec;
                break;
            case Cube:
                g = &cube_vec;
                break;
            default:
                assert(false, "Unknown function");
        }
        // Execute deflation algorithm
        w = ica_def(x_w, g, threshold, max_iter);
    } else if (strategy == Parallel) {
        // Select non-quadratic function G
        Tuple *(*g)(Matrix *);
        switch (g_func) {
            case LogCosh:
                g = &logcosh_mat;
                break;
            case Exp:
                g = &exp_mat;
                break;
            case Cube:
                g = &cube_mat;
                break;
            default:
                assert(false, "Unknown function");
        }
        // Execute parallel algorithm
        w = ica_par(x_w, g, threshold, max_iter);
    } else {
        assert(false, "Unknown strategy.");
    }

    // Reconstruct signal
    Matrix *s = mat_mul(w, x_w);
    if (whiten) {
        // Free memory
        free_mat(x_w);
        free_mat(white_mtx);
        free_vec(x_mean);
    }

    // Free memory
    free_mat(w);

    return s;
}