//
// Created by nihil on 03/10/21.
//

#include <stdlib.h>
#include <malloc.h>
#include "../include/vector.h"
#include "../include/utils.h"

/*
 * Initialize an empty vector given its length
 */
Vector *new_vec(int length) {
    assert(length > 0, "Vector length should be greater than 0.");
    Vector *v = calloc(1, sizeof(Vector));
    assert(v != NULL, "Could not allocate vector.");

    v->data = calloc(length, sizeof(fp));
    v->length = length;

    return v;
}

/*
 * Free the memory allocated for a given vector
 */
void free_vec(Vector *v) {
    if (v != NULL) {
        if (v->data != NULL) {
            free(v->data);
            v->data = NULL;
        }
        free(v);
        v = NULL;
    }
}

/*
 * Allocate a vector with the given length and fill it with random uniform integers in a given range
 */
Vector *vec_randint(int length, int amin, int amax) {
    Vector *v = new_vec(length);

    for (int i = 0; i < length; i++) {
        VEC_CELL(v, i) = (fp) (lrand48() % (amax + 1 - amin) + amin);
    }

    return v;
}

/*
 * Allocate a vector with the given length and fill it with random uniform numbers in a given range
 */
Vector *vec_rand(int length, fp min, fp max) {
    Vector *v = new_vec(length);

    for (int i = 0; i < length; i++) {
        VEC_CELL(v, i) = (fp) drand48() * (max + min) - min;
    }

    return v;
}

/*
 * Allocate a vector with the given length and fill it with random normal numbers
 */
Vector *vec_randn(int length) {
    Vector *v = new_vec(length);

    for (int i = 0; i < length; i++) {
        VEC_CELL(v, i) = gen_normal();
    }

    return v;
}

/*
 * Build a linear space with the given range and number of samples
 */
Vector *linspace(fp start, fp stop, int n_samples) {
    assert(start < stop, "");

    Vector *v = new_vec(n_samples);

    fp step = (stop - start) / (fp) (n_samples - 1);
    for (int i = 0; i < n_samples; i++) {
        VEC_CELL(v, i) = start + step * (fp) i;
    }

    return v;
}

/*
 * Compute the L2 norm of a given vector
 */
fp vec_norm(Vector *v) {
    return SQRT(dot(v, v));
}

/*
 * Get the mean of a given vector
 */
fp vec_mean(Vector *v) {
    fp acc = 0;
    for (int i = 0; i < v->length; i++) {
        acc += VEC_CELL(v, i);
    }
    return acc / (fp) (v->length);
}

/*
 * Scales a vector by a scalar constant
 */
Vector *vec_scale(Vector *v, fp scalar) {
    Vector *s = new_vec(v->length);

    for (int i = 0; i < v->length; i++) {
        VEC_CELL(s, i) = VEC_CELL(v, i) * scalar;
    }

    return s;
}

/*
 * Scales a vector by a scalar constant (in-place operation, it modifies the vector)
 */
void vec_scale_(Vector **v, fp scalar) {
    for (int i = 0; i < (*v)->length; i++) {
        VEC_CELL(*v, i) *= scalar;
    }
}

/*
 * Perform element-wise addition between two vectors
 */
Vector *vec_add(Vector *v1, Vector *v2) {
    assert(v1->length == v2->length, "The two vectors should have the same length.");
    Vector *s = new_vec(v1->length);

    for (int i = 0; i < v1->length; i++) {
        VEC_CELL(s, i) = VEC_CELL(v1, i) + VEC_CELL(v2, i);
    }

    return s;
}

/*
 * Perform element-wise addition between two vectors (in-place operation, it modifies the t1 vector)
 */
void vec_add_(Vector **v1, Vector *v2) {
    assert((*v1)->length == v2->length, "The two vectors should have the same length.");

    for (int i = 0; i < (*v1)->length; i++) {
        VEC_CELL(*v1, i) += VEC_CELL(v2, i);
    }
}

/*
 * Perform element-wise subtraction between two vectors
 */
Vector *vec_sub(Vector *v1, Vector *v2) {
    assert(v1->length == v2->length, "The two vectors should have the same length.");
    Vector *s = new_vec(v1->length);

    for (int i = 0; i < v1->length; i++) {
        VEC_CELL(s, i) = VEC_CELL(v1, i) - VEC_CELL(v2, i);
    }

    return s;
}

/*
 * Perform element-wise subtraction between two vectors (in-place operation, it modifies the t1 vector)
 */
void vec_sub_(Vector **v1, Vector *v2) {
    assert((*v1)->length == v2->length, "The two vectors should have the same length.");

    for (int i = 0; i < (*v1)->length; i++) {
        VEC_CELL(*v1, i) -= VEC_CELL(v2, i);
    }
}

/*
 * Add scalar to vector
 */
Vector *vec_scalar_add(Vector *v, fp scalar) {
    Vector *s = new_vec(v->length);

    for (int i = 0; i < v->length; i++) {
        VEC_CELL(s, i) = VEC_CELL(v, i) + scalar;
    }

    return s;
}

/*
 * Add scalar to vector (in-place operation, it modifies the vector)
 */
void vec_scalar_add_(Vector **v, fp scalar) {
    for (int i = 0; i < (*v)->length; i++) {
        VEC_CELL(*v, i) += scalar;
    }
}

/*
 * Perform element-wise product between two vectors
 */
Vector *vec_elemwise_prod(Vector *v1, Vector *v2) {
    assert(v1->length == v2->length, "The two vectors should have the same length.");
    Vector *s = new_vec(v1->length);

    for (int i = 0; i < v1->length; i++) {
        VEC_CELL(s, i) = VEC_CELL(v1, i) * VEC_CELL(v2, i);
    }

    return s;
}

/*
 * Perform element-wise product between two vectors (in-place operation, it modifies the t1 vector)
 */
void vec_elemwise_prod_(Vector **v1, Vector *v2) {
    assert((*v1)->length == v2->length, "The two vectors should have the same length.");

    for (int i = 0; i < (*v1)->length; i++) {
        VEC_CELL(*v1, i) *= VEC_CELL(v2, i);
    }
}

/*
 * Perform the dot product between two vectors
 */
fp dot(Vector *v1, Vector *v2) {
    assert(v1->length == v2->length, "The two vectors should have the same length.");

    fp acc = 0;
    for (int i = 0; i < v1->length; i++) {
        acc += VEC_CELL(v1, i) * VEC_CELL(v2, i);
    }

    return acc;
}

/*
 * Check if two vectors are equal up to a certain tolerance
 */
bool vec_equal(Vector *v1, Vector *v2, fp tol) {
    assert(v1->length == v2->length, "The two vectors should have the same length.");

    for (int i = 0; i < v1->length; i++) {
        if (VEC_CELL(v1, i) - VEC_CELL(v2, i) > tol) {
            return false;
        }
    }

    return true;
}

/*
 * Prints a vector to standard output
 */
void vec_print(Vector *v) {
    for (int i = 0; i < v->length; i++) {
        printf("%.5f ", VEC_CELL(v, i));
    }
    printf("\n");
}

/*
 * Write a vector to a binary file
 */
void vec_write(const char *path, Vector *v) {
    // Open file
    FILE *file = fopen(path, "wb");
    assert(file != NULL, "Cannot open or create file.");

    // Write number of dimensions, followed by length
    int n_dim = 1;
    assert(fwrite(&n_dim, sizeof(int), 1, file) == 1, "Could not write to file.");
    assert(fwrite(&v->length, sizeof(int), 1, file) == 1, "Could not write to file.");

    // Write vector data
    assert(fwrite(v->data, sizeof(fp), v->length, file) == v->length, "Could not write to file.");

    // Close file
    assert(fclose(file) == 0, "Could not close file.");
}

/*
 * Create a vector with given length from an array of data
 */
Vector *vec_from_array(const fp *data, int length) {
    Vector *s = new_vec(length);

    int d_idx = 0;
    for (int i = 0; i < length; i++) {
        VEC_CELL(s, i) = data[d_idx++];
    }

    return s;
}

/*
 * Copy a vector into a new vector
 */
Vector *vec_copy(Vector *v) {
    Vector *s = new_vec(v->length);

    for (int i = 0; i < v->length; i++) {
            VEC_CELL(s, i) = VEC_CELL(v, i);
    }

    return s;
}