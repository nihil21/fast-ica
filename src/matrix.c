//
// Created by nihil on 20/09/21.
//

#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include "../include/matrix.h"
#include "../include/utils.h"

/*
 * Initialize an empty matrix given its dimensions
 */
Matrix *new_mat(int height, int width) {
    assert(height > 0 && width > 0, "Matrix height and width should be greater than 0.");
    Matrix *m = calloc(1, sizeof(Matrix));
    assert(m != NULL, "Could not allocate matrix.");

    m->data = calloc(height * width, sizeof(fp));
    m->height = height;
    m->width = width;

    return m;
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
Matrix *eye(int n) {
    Matrix *i = new_mat(n, n);

    for (int k = 0; k < n; k++) {
        MAT_CELL(i, k, k) = 1;
    }

    return i;
}

/*
 * Allocate a matrix with the given dimensions and fill it with random uniform integers in a given range
 */
Matrix *mat_randint(int height, int width, int amin, int amax) {
    Matrix *m = new_mat(height, width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            MAT_CELL(m, i, j) = (fp) (lrand48() % (amax + 1 - amin) + amin);
        }
    }

    return m;
}

/*
 * Allocate a matrix with the given dimensions and fill it with random uniform numbers in a given range
 */
Matrix *mat_rand(int height, int width, fp min, fp max) {
    Matrix *m = new_mat(height, width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            MAT_CELL(m, i, j) = (fp) drand48() * (max + min) - min;
        }
    }

    return m;
}

/*
 * Allocate a matrix with the given dimensions and fill it with random normal numbers
 */
Matrix *mat_randn(int height, int width) {
    Matrix *m = new_mat(height, width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            MAT_CELL(m, i, j) = gen_normal();
        }
    }

    return m;
}

/*
 * Compute the L2 norm of a given matrix
 */
fp mat_norm(Matrix *m) {
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
fp mat_mean(Matrix *m) {
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
Vector *row_mean(Matrix *m) {
    Vector *v = new_vec(m->width);

    for (int j = 0; j < m->width; j++) {
        fp acc = 0;
        for (int i = 0; i < m->height; i++) {
            acc += MAT_CELL(m, i, j);
        }
        VEC_CELL(v, j) = acc / (fp) m->height;
    }

    return v;
}

/*
 * Get the mean of a given matrix along columns
 */
Vector *col_mean(Matrix *m) {
    Vector *v = new_vec(m->height);

    for (int i = 0; i < m->height; i++) {
        fp acc = 0;
        for (int j = 0; j < m->width; j++) {
            acc += MAT_CELL(m, i, j);
        }
        VEC_CELL(v, i) = acc / (fp) m->width;
    }

    return v;
}

/*
 * Transpose a matrix
 */
Matrix *transpose(Matrix* m) {
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
 * Scales a matrix by a scalar constant
 */
Matrix *mat_scale(Matrix *m, fp scalar) {
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
void mat_scale_(Matrix **m, fp scalar) {
    for (int i = 0; i < (*m)->height; i++) {
        for (int j = 0; j < (*m)->width; j++) {
            MAT_CELL(*m, i, j) *= scalar;
        }
    }
}

/*
 * Perform element-wise addition between two matrices
 */
Matrix *mat_add(Matrix *m1, Matrix *m2) {
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
 * Perform element-wise addition between two matrices (in-place operation, it modifies the first matrix)
 */
void mat_add_(Matrix **m1, Matrix *m2) {
    assert((*m1)->height == m2->height && (*m1)->width == m2->width, "The two matrices should have the same shape.");

    for (int i = 0; i < (*m1)->height; i++) {
        for (int j = 0; j < (*m1)->width; j++) {
            MAT_CELL(*m1, i, j) += MAT_CELL(m2, i, j);
        }
    }
}

/*
 * Perform element-wise subtraction between two matrices
 */
Matrix *mat_sub(Matrix *m1, Matrix *m2) {
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
 * Perform element-wise subtraction between two matrices (in-place operation, it modifies the first matrix)
 */
void mat_sub_(Matrix **m1, Matrix *m2) {
    assert((*m1)->height == m2->height && (*m1)->width == m2->width, "The two matrices should have the same shape.");

    for (int i = 0; i < (*m1)->height; i++) {
        for (int j = 0; j < (*m1)->width; j++) {
            MAT_CELL(*m1, i, j) -= MAT_CELL(m2, i, j);
        }
    }
}

/*
 * Add scalar to matrix
 */
Matrix *mat_scalar_add(Matrix *m, fp scalar) {
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
void mat_scalar_add_(Matrix **m, fp scalar) {
    for (int i = 0; i < (*m)->height; i++) {
        for (int j = 0; j < (*m)->width; j++) {
            MAT_CELL(*m, i, j) += scalar;
        }
    }
}

/*
 * Perform element-wise product (i.e. Hadamard) between two matrices
 */
Matrix *mat_hadamard(Matrix *m1, Matrix *m2) {
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
 * Perform element-wise product (i.e. Hadamard) between two matrices (in-place operation, it modifies the first matrix)
 */
void mat_hadamard_(Matrix **m1, Matrix *m2) {
    assert((*m1)->height == m2->height && (*m1)->width == m2->width, "The two matrices should have the same shape.");

    for (int i = 0; i < (*m1)->height; i++) {
        for (int j = 0; j < (*m1)->width; j++) {
            MAT_CELL(*m1, i, j) *= MAT_CELL(m2, i, j);
        }
    }
}

/*
 * Perform matrix multiplication between an AxB matrix and a BxC matrix
 */
Matrix *mat_mul(Matrix *m1, Matrix *m2) {
    assert(m1->width == m2->height, "The width of the first matrix and the height of the second matrix should be equal.");
    Matrix *s = new_mat(m1->height, m2->width);

    // i-k-j loop to optimize memory access
    for (int i = 0; i < m1->height; i++) {
        for (int k = 0; k < m1->height; k++) {
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
    for (int k = 0; k < m2->height; k++) {
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
            for (int k = 0; k < m2->width; k++) {
                MAT_CELL(s, i, j) += MAT_CELL(m1, i, k) * MAT_CELL(m2, j, k);
            }
        }
    }

    return s;
}

/*
 * Check if a matrix is square
 */
bool is_square(Matrix *m) {
    return m->height == m->width;
}

/*
 * Copy the values on the diagonal of a matrix into a new vector
 */
Vector *diagonal(Matrix *m) {
    // Get minimum between height and width
    int dim = (m->height > m->width ? m->width : m->height);
    Vector *d = new_vec(dim);

    for (int i = 0; i < dim; i++) {
        VEC_CELL(d, i) = MAT_CELL(m, i, i);
    }

    return d;
}

/*
 * Slice given matrix and return a new matrix
 */
Matrix *read_slice(Matrix *m, int row_start, int row_stop, int col_start, int col_stop) {
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
void write_slice(Matrix **m1, Matrix *m2, int row_start, int col_start) {
    int row_stop = row_start + m2->height - 1;
    int col_stop = col_start + m2->width - 1;
    // Input check
    assert(0 <= row_start && row_start < (*m1)->height, "The row_start argument is out of bounds.");
    assert(0 <= row_stop && row_stop < (*m1)->height, "The row_stop argument is out of bounds.");
    assert(0 <= col_start && col_start < (*m1)->width, "The col_start argument is out of bounds.");
    assert(0 <= col_stop && col_stop < (*m1)->width, "The col_stop argument is out of bounds.");

    for (int i = row_start; i <= row_stop; i++) {
        for (int j = col_start; j <= col_stop; j++) {
            MAT_CELL(*m1, i, j) = MAT_CELL(m2, i - row_start, j - col_start);
        }
    }
}

/*
 * Convert 1D Matrix into Vector (destructive process)
 */
Vector *mat_to_vec(Matrix *m, bool free_struct) {
    assert(m->height == 1 || m->width == 1, "Matrix must be 1D.");
    // Allocate Vector and manually set fields
    Vector *s = malloc(sizeof(Vector));
    s->length = m->height == 1 ? m->width : m->height;
    s->data = m->data;  // shared pointer to data field

    // Free matrix struct (not data)
    if (free_struct)
        free(m);

    return s;
}

/*
 * Convert Vector into 1D column Matrix (destructive process)
 */
Matrix *vec_to_mat(Vector *v, bool free_struct) {
    // Allocate 1D column Matrix and manually set fields
    Matrix *s = malloc(sizeof(Matrix));
    s->height = v->length;
    s->width = 1;
    s->data = v->data;  // shared pointer to data field

    // Free vector struct (not data)
    if (free_struct)
        free(v);

    return s;
}

/*
 * Extract the k-th row, creating a new vector
 */
Vector *extract_row(Matrix *m, int k) {
    assert(0 <= k && k < m->height, "Index is out of bounds for rows.");
    Vector *r = new_vec(m->width);

    for (int i = 0; i < m->width; i++) {
        VEC_CELL(r, i) = MAT_CELL(m, k, i);
    }

    return r;
}

/*
 * Extract the k-th column, creating a new vector
 */
Vector *extract_col(Matrix *m, int k) {
    assert(0 <= k && k < m->width, "Index is out of bounds for columns.");
    Vector *c = new_vec(m->height);

    for (int i = 0; i < m->height; i++) {
        VEC_CELL(c, i) = MAT_CELL(m, i, k);
    }

    return c;
}

/*
 * Copy the values from a vector into the specified matrix row (it modifies the matrix)
 */
void paste_row(Matrix **m, Vector *R, int k) {
    assert(0 <= k && k < (*m)->height, "Index is out of bounds for rows.");
    assert((*m)->width == R->length, "Vector length should be equal to matrix width.");

    for (int i = 0; i < (*m)->width; i++) {
        MAT_CELL(*m, k, i) = VEC_CELL(R, i);
    }
}

/*
 * Copy the values from a vector into the specified matrix column (it modifies the matrix)
 */
void paste_col(Matrix **m, Vector *C, int k) {
    assert(0 <= k && k < (*m)->width, "Index is out of bounds for columns.");
    assert((*m)->height == C->length, "Vector length should be equal to matrix height.");

    for (int i = 0; i < (*m)->height; i++) {
        MAT_CELL(*m, i, k) = VEC_CELL(C, i);
    }
}

/*
 * Perform row-wise addition between a matrix and a vector
 */
Matrix *row_add(Matrix *m, Vector *v) {
    assert(m->width == v->length, "Vector length should be equal to matrix width.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) + VEC_CELL(v, j);
        }
    }

    return s;
}

/*
 * Perform row-wise addition between a matrix and a vector (in-place operation, it modifies the matrix)
 */
void row_add_(Matrix **m, Vector *v) {
    assert((*m)->width == v->length, "Vector length should be equal to matrix width.");

    for (int i = 0; i < (*m)->height; i++) {
        for (int j = 0; j < (*m)->width; j++) {
            MAT_CELL(*m, i, j) += VEC_CELL(v, j);
        }
    }
}

/*
 * Perform column-wise addition between a matrix and a vector
 */
Matrix *col_add(Matrix *m, Vector *v) {
    assert(m->height == v->length, "Vector length should be equal to matrix height.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) + VEC_CELL(v, i);
        }
    }

    return s;
}

/*
 * Perform column-wise addition between a matrix and a vector (in-place operation, it modifies the matrix)
 */
void col_add_(Matrix **m, Vector *v) {
    assert((*m)->height == v->length, "Vector length should be equal to matrix height.");

    for (int i = 0; i < (*m)->height; i++) {
        for (int j = 0; j < (*m)->width; j++) {
            MAT_CELL(*m, i, j) += VEC_CELL(v, i);
        }
    }
}

/*
 * Perform row-wise subtraction between a matrix and a vector
 */
Matrix *row_sub(Matrix *m, Vector *v) {
    assert(m->width == v->length, "Vector length should be equal to matrix width.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) - VEC_CELL(v, j);
        }
    }

    return s;
}

/*
 * Perform row-wise subtraction between a matrix and a vector (in-place operation, it modifies the matrix)
 */
void row_sub_(Matrix **m, Vector *v) {
    assert((*m)->width == v->length, "Vector length should be equal to matrix width.");

    for (int i = 0; i < (*m)->height; i++) {
        for (int j = 0; j < (*m)->width; j++) {
            MAT_CELL(*m, i, j) -= VEC_CELL(v, j);
        }
    }
}

/*
 * Perform column-wise subtraction between a matrix and a vector
 */
Matrix *col_sub(Matrix *m, Vector *v) {
    assert(m->height == v->length, "Vector length should be equal to matrix height.");
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j) - VEC_CELL(v, i);
        }
    }

    return s;
}

/*
 * Perform column-wise subtraction between a matrix and a vector (in-place operation, it modifies the matrix)
 */
void col_sub_(Matrix **m, Vector *v) {
    assert((*m)->height == v->length, "Vector length should be equal to matrix height.");

    for (int i = 0; i < (*m)->height; i++) {
        for (int j = 0; j < (*m)->width; j++) {
            MAT_CELL(*m, i, j) -= VEC_CELL(v, i);
        }
    }
}

/*
 * Perform Matrix-Vector multiplication between an AxB matrix and a B vector
 */
Vector *mat_vec_mul(Matrix *m, Vector *v) {
    assert(m->width == v->length, "The width of the matrix and the length of the vector should be equal.");
    Vector *s = new_vec(m->height);

    for (int i = 0; i < m->height; i++) {
        for (int k = 0; k < v->length; k++) {
            VEC_CELL(s, i) += MAT_CELL(m, i, k) * VEC_CELL(v, k);
        }
    }

    return s;
}

/*
 * Perform Matrix-Vector multiplication between a BxA matrix and a B vector, without transposing the matrix
 */
Vector *mat_vec_mul_trans(Matrix *m, Vector *v) {
    assert(m->height == v->length, "The height of the matrix and the length of the vector should be equal.");
    Vector *s = new_vec(m->width);

    for (int i = 0; i < m->width; i++) {
        for (int k = 0; k < v->length; k++) {
            VEC_CELL(s, i) += MAT_CELL(m, k, i) * VEC_CELL(v, k);
        }
    }

    return s;
}

/*
 * Perform outer product between two vectors
 */
Matrix *outer(Vector *v1, Vector *v2) {
    Matrix *s = new_mat(v1->length, v2->length);

    for (int i = 0; i < v1->length; i++) {
        for (int j = 0; j < v2->length; j++) {
            MAT_CELL(s, i, j) = VEC_CELL(v1, i) * VEC_CELL(v2, j);
        }
    }

    return s;
}

/*
 * Prints a matrix to standard output
 */
void mat_print(Matrix *m) {
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
void mat_write(const char *path, Matrix *m) {
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
Matrix *mat_from_array(const fp *data, int height, int width) {
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
Matrix *mat_copy(Matrix *m) {
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = MAT_CELL(m, i, j);
        }
    }

    return s;
}