//
// Created by nihil on 20/09/21.
//

#include <malloc.h>
#include <string.h>
#include "../include/matrix.h"
#include "../include/utils.h"
#include "../include/random.h"

/*
 * Initialize a zero matrix given its dimensions
 */
Matrix *new_mat(const int height, const int width) {
    assert(height > 0 && width > 0, "Matrix height and width should be greater than 0.");
    Matrix *m = NULL;
    m = calloc(1, sizeof(Matrix));
    assert(m != NULL, "Could not allocate matrix.");

    m->data = NULL;
    m->data = calloc(height * width, sizeof(fp));
    m->height = height;
    m->width = width;

    return m;
}

/*
 * Initialize a zero column vector given its dimensions
 */
Matrix *new_vec(const int length) {
    return new_mat(length, 1);
}

/*
 * Free the memory allocated for a given matrix
 */
void free_mat(Matrix *m) {
    if (m != NULL) {
        if (m->data != NULL) {
            free(m->data);
            m->data = NULL;
        }
        free(m);
        m = NULL;
    }
}

/*
 * Allocate the identity matrix
 */
Matrix *eye(const int n) {
    Matrix *i = new_mat(n, n);

    for (int k = 0; k < n; k++) {
        MAT_CELL(i, k, k) = 1;
    }

    return i;
}

/*
 * Allocate a matrix with the given dimensions and fill it with random uniform integers in a given range
 */
Matrix *mat_randint(const int height, const int width, const int min, const int max) {
    Matrix *m = new_mat(height, width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            MAT_CELL(m, i, j) = (fp) uniform_randint_range(min, max);
        }
    }

    return m;
}

/*
 * Allocate a matrix with the given dimensions and fill it with random uniform numbers in a given range
 */
Matrix *mat_rand(const int height, const int width, const fp min, const fp max) {
    Matrix *m = new_mat(height, width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            MAT_CELL(m, i, j) = uniform_rand_range(min, max);
        }
    }

    return m;
}

/*
 * Allocate a matrix with the given dimensions and fill it with random normal numbers
 */
Matrix *mat_randn(const int height, const int width) {
    Matrix *m = new_mat(height, width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            MAT_CELL(m, i, j) = standard_rand();
        }
    }

    return m;
}

/*
 * Build a linear space with the given range and number of samples
 */
Matrix *linspace(const fp start, const fp stop, const int n_samples) {
    assert(start < stop, "The stop argument should be greater than the start argument.");

    Matrix *ls = new_vec(n_samples);

    fp step = (stop - start) / (fp) (n_samples - 1);
    for (int i = 0; i < n_samples; i++) {
        MAT_CELL(ls, i, 0) = start + step * (fp) i;
    }

    return ls;
}

/*
 * Compute the L2 norm of a given matrix
 */
fp norm(const Matrix *m) {
    fp acc = 0;
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            acc += MAT_CELL(m, i, j) * MAT_CELL(m, i, j);
        }
    }
    return SQRT(acc);
}

/*
 * Get the mean of a given matrix
 */
fp mean(const Matrix *m) {
    fp acc = 0;
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            acc += MAT_CELL(m, i, j);
        }
    }
    return acc / (fp) (m->height * m->width);
}

/*
 * Get the mean of a given matrix along rows
 */
Matrix *row_mean(const Matrix *m) {
    Matrix *r = new_mat(1, m->width);  // row vector

    // Memory access not contiguous
    for (int j = 0; j < m->width; j++) {
        fp acc = 0;
        for (int i = 0; i < m->height; i++) {
            acc += MAT_CELL(m, i, j);
        }
        MAT_CELL(r, 0, j) = acc / (fp) m->height;
    }

    return r;
}

/*
 * Get the mean of a given matrix along columns
 */
Matrix *col_mean(const Matrix *m) {
    Matrix *c = new_vec(m->height);  // column vector

    for (int i = 0; i < m->height; i++) {
        fp acc = 0;
        for (int j = 0; j < m->width; j++) {
            acc += MAT_CELL(m, i, j);
        }
        MAT_CELL(c, i, 0) = acc / (fp) m->width;
    }

    return c;
}

/*
 * Get the standard deviation of a given matrix
 */
fp std(const Matrix*m) {
    // Compute mean
    fp mean_ = mean(m);

    fp acc = 0;
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            acc += POW(MAT_CELL(m, i, j) - mean_, 2);
        }
    }

    return SQRT(acc / (fp) (m->height * m->width));
}

/*
 * Get the standard deviation of a given matrix along rows
 */
Matrix *row_std(const Matrix *m) {
    Matrix *r = new_mat(1, m->width);  // row vector

    // Compute mean
    Matrix *mean_ = row_mean(m);

    // Memory access not contiguous
    for (int j = 0; j < m->width; j++) {
        fp acc = 0;
        for (int i = 0; i < m->height; i++) {
            acc += POW(MAT_CELL(m, i, j) - MAT_CELL(mean_, 0, j), 2);
        }
        MAT_CELL(r, 0, j) = SQRT(acc / (fp) m->height);
    }

    // Free memory
    free_mat(mean_);

    return r;
}

/*
 * Get the standard deviation of a given matrix along columns
 */
Matrix *col_std(const Matrix *m) {
    Matrix *c = new_mat(m->height, 1);  // column vector

    // Compute mean
    Matrix *mean_ = col_mean(m);

    for (int i = 0; i < m->height; i++) {
        fp acc = 0;
        for (int j = 0; j < m->width; j++) {
            acc += POW(MAT_CELL(m, i, j) - MAT_CELL(mean_, i, 0), 2);
        }
        MAT_CELL(c, i, 0) = SQRT(acc / (fp) m->width);
    }

    // Free memory
    free_mat(mean_);

    return c;
}

/*
 * Transpose a matrix
 */
Matrix *transpose(const Matrix * m) {
    int t_height = m->width;
    int t_width = m->height;
    Matrix *t = new_mat(t_height, t_width);

    for (int i = 0; i < t_height; i++) {
        for (int j = 0; j < t_width; j++) {
            MAT_CELL(t, i, j) = MAT_CELL(m, j, i);
        }
    }

    return t;
}

/*
 * Add scalar to matrix
 */
Matrix *add_scalar(const Matrix *m, const fp scalar) {
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) + scalar;
        }
    }

    return s;
}

/*
 * Add scalar to matrix (in-place operation, it modifies the matrix)
 */
void add_scalar_(const Matrix *m, const fp scalar) {
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) += scalar;
        }
    }
}

/*
 * Scales a matrix by a scalar constant
 */
Matrix *scale(const Matrix *m, const fp scalar) {
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) * scalar;
        }
    }

    return s;
}

/*
 * Scales a matrix by a scalar constant (in-place operation, it modifies the matrix)
 */
void scale_(const Matrix *m, const fp scalar) {
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) *= scalar;
        }
    }
}

/*
 * Perform element-wise addition between two matrices with same shape
 */
Matrix *add_mat(const Matrix *m1, const Matrix *m2) {
    assert(m1->height == m2->height && m1->width == m2->width, "The two matrices should have the same shape.");
    Matrix *s = new_mat(m1->height, m1->width);

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m1, i, j) + MAT_CELL(m2, i, j);
        }
    }

    return s;
}

/*
 * Perform element-wise addition between two matrices with same shape (in-place operation, it modifies the first matrix)
 */
void add_mat_(const Matrix *m1, const Matrix *m2) {
    assert(m1->height == m2->height && m1->width == m2->width, "The two matrices should have the same shape.");

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            MAT_CELL(m1, i, j) += MAT_CELL(m2, i, j);
        }
    }
}

/*
 * Perform row-wise addition between a matrix and a row vector with same width
 */
Matrix *add_row(const Matrix *m, const Matrix *r) {
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(r->width == m->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");
    Matrix *s = new_mat(m->height, m->width);

    // Memory access not contiguous
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) + MAT_CELL(r, 0, i);
        }
    }

    return s;
}

/*
 * Perform row-wise addition between a matrix and a row vector with same width (in-place operation, it modifies the first matrix)
 */
void add_row_(const Matrix *m, const Matrix *r) {
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(r->width == m->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) += MAT_CELL(r, 0, i);
        }
    }
}

/*
 * Perform column-wise addition between a matrix and a column vector with same height
 */
Matrix *add_col(const Matrix *m, const Matrix *c) {
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(c->height == m->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) + MAT_CELL(c, i, 0);
        }
    }

    return s;
}

/*
 * Perform column-wise addition between a matrix and a column vector with same height (in-place operation, it modifies the first matrix)
 */
void add_col_(const Matrix *m, const Matrix *c) {
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(c->height == m->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) += MAT_CELL(c, i, 0);
        }
    }
}

/*
 * Perform element-wise subtraction between two matrices with same shape
 */
Matrix *sub_mat(const Matrix *m1, const Matrix *m2) {
    assert(m1->height == m2->height && m1->width == m2->width, "The two matrices should have the same shape.");
    Matrix *s = new_mat(m1->height, m1->width);

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m1, i, j) - MAT_CELL(m2, i, j);
        }
    }

    return s;
}

/*
 * Perform element-wise subtraction between two matrices with same shape (in-place operation, it modifies the first matrix)
 */
void sub_mat_(const Matrix *m1, const Matrix *m2) {
    assert(m1->height == m2->height && m1->width == m2->width, "The two matrices should have the same shape.");

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            MAT_CELL(m1, i, j) -= MAT_CELL(m2, i, j);
        }
    }
}

/*
 * Perform row-wise subtraction between a matrix and a row vector with same width
 */
Matrix *sub_row(const Matrix *m, const Matrix *r) {
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(r->width == m->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) - MAT_CELL(r, 0, i);
        }
    }

    return s;
}

/*
 * Perform row-wise subtraction between a matrix and a row vector with same width (in-place operation, it modifies the first matrix)
 */
void sub_row_(const Matrix *m, const Matrix *r) {
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(r->width == m->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) -= MAT_CELL(r, 0, i);
        }
    }
}

/*
 * Perform column-wise subtraction between a matrix and a column vector with same height
 */
Matrix *sub_col(const Matrix *m, const Matrix *c) {
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(c->height == m->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) - MAT_CELL(c, i, 0);
        }
    }

    return s;
}

/*
 * Perform column-wise subtraction between a matrix and a column vector with same height (in-place operation, it modifies the first matrix)
 */
void sub_col_(const Matrix *m, const Matrix *c) {
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(c->height == m->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) -= MAT_CELL(c, i, 0);
        }
    }
}

/*
 * Perform element-wise product (i.e. Hadamard) between two matrices with same shape
 */
Matrix *hadamard(const Matrix *m1, const Matrix *m2) {
    assert(m1->height == m2->height && m1->width == m2->width, "The two matrices should have the same shape.");
    Matrix *s = new_mat(m1->height, m1->width);

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m1, i, j) * MAT_CELL(m2, i, j);
        }
    }

    return s;
}

/*
 * Perform element-wise product (i.e. Hadamard) between two matrices with same shape (in-place operation, it modifies the first matrix)
 */
void hadamard_(const Matrix *m1, const Matrix *m2) {
    assert(m1->height == m2->height && m1->width == m2->width, "The two matrices should have the same shape.");

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            MAT_CELL(m1, i, j) *= MAT_CELL(m2, i, j);
        }
    }
}

/*
 * Perform row-wise product (i.e. Hadamard) between a matrix and a row vector with same width
 */
Matrix *hadamard_row(const Matrix *m, const Matrix *r) {
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(r->width == m->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) * MAT_CELL(r, 0, i);
        }
    }

    return s;
}

/*
 * Perform row-wise product (i.e. Hadamard) between a matrix and a row vector with same width (in-place operation, it modifies the first matrix)
 */
void hadamard_row_(const Matrix *m, const Matrix *r) {
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(r->width == m->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) *= MAT_CELL(r, 0, i);
        }
    }
}

/*
 * Perform column-wise product (i.e. Hadamard) between a matrix and a column vector with same height
 */
Matrix *hadamard_col(const Matrix *m, const Matrix *c) {
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(c->height == m->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) * MAT_CELL(c, i, 0);
        }
    }

    return s;
}

/*
 * Perform column-wise product (i.e. Hadamard) between a matrix and a column vector with same height (in-place operation, it modifies the first matrix)
 */
void hadamard_col_(const Matrix *m, const Matrix *c) {
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(c->height == m->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) *= MAT_CELL(c, i, 0);
        }
    }
}

/*
 * Perform element-wise division between two matrices with same shape
 */
Matrix *div_mat(const Matrix *m1, const Matrix *m2) {
    assert(m1->height == m2->height && m1->width == m2->width, "The two matrices should have the same shape.");
    Matrix *s = new_mat(m1->height, m1->width);

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m1, i, j) / MAT_CELL(m2, i, j);
        }
    }
}

/*
 * Perform element-wise division between two matrices with same shape (in-place operation, it modifies the first matrix)
 */
void div_mat_(const Matrix *m1, const Matrix *m2) {
    assert(m1->height == m2->height && m1->width == m2->width, "The two matrices should have the same shape.");

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            MAT_CELL(m1, i, j) /= MAT_CELL(m2, i, j);
        }
    }
}

/*
 * Perform row-wise division between a matrix and a row vector with same width
 */
Matrix *div_row(const Matrix *m, const Matrix *r) {
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(r->width == m->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) / MAT_CELL(r, 0, i);
        }
    }
}

/*
 * Perform row-wise division between a matrix and a row vector with same width (in-place operation, it modifies the first matrix)
 */
void div_row_(const Matrix *m, const Matrix *r) {
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(r->width == m->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) /= MAT_CELL(r, 0, i);
        }
    }
}

/*
 * Perform column-wise division between a matrix and a column vector with same height
 */
Matrix *div_col(const Matrix *m, const Matrix *c) {
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(c->height == m->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) / MAT_CELL(c, i, 0);
        }
    }
}

/*
 * Perform column-wise division between a matrix and a column vector with same height (in-place operation, it modifies the first matrix)
 */
void div_col_(const Matrix *m, const Matrix *c) {
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(c->height == m->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(m, i, j) /= MAT_CELL(c, i, 0);
        }
    }
}

/*
 * Perform matrix multiplication between an AxB matrix and a BxC matrix
 */
Matrix *mat_mul(const Matrix *m1, const Matrix *m2) {
    assert(m1->width == m2->height, "The width of the first matrix and the height of the second matrix should be equal.");
    Matrix *s = new_mat(m1->height, m2->width);

    // i-k-j loop to optimize memory access
    for (int i = 0; i < m1->height; i++) {
        for (int k = 0; k < m1->width; k++) {
            for (int j = 0; j < m2->width; j++) {
                MAT_CELL(s, i, j) += MAT_CELL(m1, i, k) * MAT_CELL(m2, k, j);
            }
        }
    }

    return s;
}

/*
 * Perform matrix multiplication between a BxA matrix and a BxC matrix, without transposing the first matrix
 */
Matrix *mat_mul_trans1(Matrix* m1, Matrix* m2) {
    assert(m1->height == m2->height, "The height of the two matrices should be equal.");
    Matrix *s = new_mat(m1->width, m2->width);

    // k-i-j loop to optimize memory access
    for (int k = 0; k < m1->height; k++) {
        for (int i = 0; i < m1->width; i++) {
            for (int j = 0; j < m2->width; j++) {
                MAT_CELL(s, i, j) += MAT_CELL(m1, k, i) * MAT_CELL(m2, k, j);
            }
        }
    }

    return s;
}

/*
 * Perform matrix multiplication between an AxB matrix and a CxB matrix, without transposing the second matrix
 */
Matrix *mat_mul_trans2(Matrix* m1, Matrix* m2) {
    assert(m1->width == m2->width, "The width of the two matrices should be equal.");
    Matrix *s = new_mat(m1->height, m2->height);

    // i-j-k loop to optimize memory access
    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m2->height; j++) {
            for (int k = 0; k < m1->width; k++) {
                MAT_CELL(s, i, j) += MAT_CELL(m1, i, k) * MAT_CELL(m2, j, k);
            }
        }
    }

    return s;
}

/*
 * Perform dot product between two vectors
 */
fp dot(const Matrix *v1, const Matrix *v2) {
    assert(is_vector(v1) && is_vector(v2), "The two matrices should be vectors (i.e. either their height or width should be equal to 1).");
    assert(MAX(v1->height, v1->width) == MAX(v2->height, v2->width), "The two vectors should have the same length.");

    fp acc = 0;
    const int len = v1->height == 1 ? v1->width : v1->height;
    for (int i = 0; i < len; i++) {
        acc += MAT_DATA(v1)[i] * MAT_DATA(v2)[i];
    }

    return acc;
}

Matrix *outer(const Matrix *v1, const Matrix *v2) {
    assert(is_vector(v1) && is_vector(v2), "The v1 and v2 matrices should be vectors (i.e. either their height or width should be equal to 1).");
    Matrix *s = new_mat(MAX(v1->height, v1->width), MAX(v2->height, v2->width));

    const int len1 = MAX(v1->height, v1->width);
    const int len2 = MAX(v2->height, v2->width);
    for (int i = 0; i < len1; i++) {
        for (int j = 0; j < len2; j++) {
            MAT_CELL(s, i, j) = MAT_DATA(v1)[i] * MAT_DATA(v2)[j];
        }
    }

    return s;
}

/*
 * Check if two matrices are equal
 */
bool are_equal(const Matrix *m1, const Matrix *m2, fp tol) {
    if (m1->height != m2->height || m1->width != m2->width)
        return false;

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            if (MAT_CELL(m1, i, j) - MAT_CELL(m2, i, j) > tol)
                return false;
        }
    }
    return true;
}

/*
 * Check if a matrix is square
 */
bool is_square(const Matrix *m) {
    return m->height == m->width;
}

/*
 * Check if a matrix is a vector
 */
bool is_vector(const Matrix *m) {
    return m->height == 1 || m->width == 1;
}

/*
 * Check if a matrix is a row vector
 */
bool is_row_vector(const Matrix *m) {
    return m->height == 1;
}

/*
 * Check if a matrix is a column vector
 */
bool is_col_vector(const Matrix *m) {
    return m->width == 1;
}

/*
 * Copy the values on the diagonal of a matrix into a new column vector
 */
Matrix *diagonal(const Matrix *m) {
    // Get minimum between height and width
    int dim = (m->height > m->width ? m->width : m->height);
    Matrix *d = new_vec(dim);

    for (int i = 0; i < dim; i++) {
        MAT_CELL(d, i, 0) = MAT_CELL(m, i, i);
    }

    return d;
}

/*
 * Set cells below the main diagonal to zero (in-place operation, it modifies the matrix)
 */
void tri_up(const Matrix *m) {
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < i; j++) {
            MAT_CELL(m, i, j) = 0;
        }
    }
}

/*
 * Slice given matrix and return a new matrix
 */
Matrix *read_slice(const Matrix *m, const int  row_start, const int  row_stop, const int  col_start, const int  col_stop) {
    // Input check
    assert(0 <= row_start && row_start < m->height, "The row_start argument is out of bounds.");
    assert(0 <= row_stop && row_stop < m->height, "The row_stop argument is out of bounds.");
    assert(0 <= col_start && col_start < m->width, "The col_start argument is out of bounds.");
    assert(0 <= col_stop && col_stop < m->width, "The col_stop argument is out of bounds.");
    assert(row_start <= row_stop, "The row_stop argument must be greater than or equal to row_start.");
    assert(col_start <= col_stop, "The col_stop argument must be greater than or equal to col_start.");

    Matrix *s = new_mat(row_stop - row_start + 1, col_stop - col_start + 1);

    for (int i = row_start; i <= row_stop; i++) {
        for (int j = col_start; j <= col_stop; j++) {
            MAT_CELL(s, i - row_start, j - col_start) = MAT_CELL(m, i, j);
        }
    }

    return s;
}

/*
 * Write into sliced matrix, modifying it
 */
void write_slice(const Matrix *m1, const Matrix *m2, const int  row_start, const int  col_start) {
    int row_stop = row_start + m2->height - 1;
    int col_stop = col_start + m2->width - 1;
    // Input check
    assert(0 <= row_start && row_start < m1->height, "The row_start argument is out of bounds.");
    assert(0 <= row_stop && row_stop < m1->height, "The row_stop argument is out of bounds.");
    assert(0 <= col_start && col_start < m1->width, "The col_start argument is out of bounds.");
    assert(0 <= col_stop && col_stop < m1->width, "The col_stop argument is out of bounds.");

    for (int i = row_start; i <= row_stop; i++) {
        for (int j = col_start; j <= col_stop; j++) {
            MAT_CELL(m1, i, j) = MAT_CELL(m2, i - row_start, j - col_start);
        }
    }
}

/*
 * Extract the k-th row, creating a new vector
 */
Matrix *extract_row(const Matrix *m, const int  k) {
    assert(0 <= k && k < m->height, "Index is out of bounds for rows.");
    Matrix *r = new_mat(1, m->width);

    for (int i = 0; i < m->width; i++) {
        MAT_CELL(r, 0, i) = MAT_CELL(m, k, i);
    }

    return r;
}

/*
 * Extract the k-th column, creating a new vector
 */
Matrix *extract_col(const Matrix *m, const int  k) {
    assert(0 <= k && k < m->width, "Index is out of bounds for columns.");
    Matrix *c = new_vec(m->height);

    for (int i = 0; i < m->height; i++) {
        MAT_CELL(c, i, 0) = MAT_CELL(m, i, k);
    }

    return c;
}

/*
 * Copy the values from a vector into the specified matrix row (it modifies the matrix)
 */
void paste_row(const Matrix *m, const Matrix *r, const int  k) {
    assert(0 <= k && k < m->height, "Index is out of bounds for rows.");
    assert(is_row_vector(r), "The second matrix should be a row vector (i.e. with height equal to 1).");
    assert(m->width == r->width, "The width of the first matrix and the length of the row vector (i.e. its width) should be the same.");

    for (int i = 0; i < m->width; i++) {
        MAT_CELL(m, k, i) = MAT_CELL(r, 0, i);
    }
}

/*
 * Copy the values from a vector into the specified matrix column (it modifies the matrix)
 */
void paste_col(const Matrix *m, const Matrix *c, const int  k) {
    assert(0 <= k && k < m->width, "Index is out of bounds for columns.");
    assert(is_col_vector(c), "The second matrix should be a column vector (i.e. with width equal to 1).");
    assert(m->height == c->height, "The height of the first matrix and the length of the column vector (i.e. its height) should be the same.");

    for (int i = 0; i < m->height; i++) {
        MAT_CELL(m, i, k) = MAT_CELL(c, i, 0);
    }
}

/*
 * Prints a matrix to standard output
 */
void print_mat(const Matrix *m) {
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            printf("%.5f ", MAT_CELL(m, i, j));
        }
        printf("\n");
    }
}

/*
 * Write a matrix to a binary file
 */
void write_mat(const char *path, const Matrix *m) {
    // Open file
    FILE *file = fopen(path, "wb");
    assert(file != NULL, "Cannot open or create file.");

    // Write number of dimensions, followed by height and width
    int n_dim = 2;
    assert(fwrite(&n_dim, sizeof(int), 1, file) == 1, "Could not write to file.");
    assert(fwrite(&m->height, sizeof(int), 1, file) == 1, "Could not write to file.");
    assert(fwrite(&m->width, sizeof(int), 1, file) == 1, "Could not write to file.");

    // Write flattened matrix data
    int size = m->height * m->width;
    assert(fwrite(m->data, sizeof(fp), size, file) == size, "Could not write to file.");

    // Close file
    assert(fclose(file) == 0, "Could not close file.");
}

/*
 * Create a matrix with given height and width from an array of data
 */
Matrix *from_array(const fp *data, const int  height, const int  width) {
    Matrix *s = new_mat(height, width);

    int d_idx = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            MAT_CELL(s, i, j) = data[d_idx++];
        }
    }

    return s;
}

/*
 * Copy a matrix into a new matrix
 */
Matrix *copy_mat(const Matrix *m) {
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j);
        }
    }

    return s;
}