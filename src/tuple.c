//
// Created by nihil on 04/10/21.
//

#include <malloc.h>
#include "../include/tuple.h"
#include "../include/utils.h"

/*
 * Creates a tuple from a pair of matrices
 */
Tuple *new_tuple(Matrix *m1, Matrix *m2) {
    Tuple *tuple;
    tuple = malloc(sizeof(Tuple));
    assert(tuple != NULL, "Could not allocate tuple.");

    // Set fields
    tuple->m1 = m1;
    tuple->m2 = m2;

    return tuple;
}

/*
 * Free the memory allocated for a given tuple
 */
void free_tuple(Tuple *tuple, bool free_members) {
    if (tuple != NULL) {
        // Optionally free Matrix members
        if (free_members) {
            // Free first matrix
            if (tuple->m1 != NULL) {
                free_mat(tuple->m1);
            }
            // Free second matrix
            if (tuple->m2 != NULL) {
                free_mat(tuple->m2);
            }
        }

        // Free tuple
        free(tuple);
        tuple = NULL;
    }
}