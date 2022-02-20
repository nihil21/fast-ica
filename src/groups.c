//
// Created by nihil on 04/10/21.
//

#include <malloc.h>
#include "../include/groups.h"
#include "../include/utils.h"

/*
 * Create a pair from two matrices
 */
Pair *new_pair(Matrix *m1, Matrix *m2) {
    Pair *pair;
    pair = malloc(sizeof(Pair));
    assert(pair != NULL, "Could not allocate pair.");

    // Set fields
    pair->m1 = m1;
    pair->m2 = m2;

    return pair;
}

/*
 * Free the memory allocated for a given pair
 */
void free_pair(Pair *pair, bool free_members) {
    if (pair != NULL) {
        // Optionally free Matrix members
        if (free_members) {
            // Free first matrix
            if (pair->m1 != NULL) {
                free_mat(pair->m1);
            }
            // Free second matrix
            if (pair->m2 != NULL) {
                free_mat(pair->m2);
            }
        }

        // Free pair
        free(pair);
        pair = NULL;
    }
}

/*
 * Create a triplet from three matrices
 */
Triplet *new_triplet(Matrix *m1, Matrix *m2, Matrix *m3) {
    Triplet *triplet;
    triplet = malloc(sizeof(Triplet));
    assert(triplet != NULL, "Could not allocate triplet.");

    // Set fields
    triplet->m1 = m1;
    triplet->m2 = m2;
    triplet->m3 = m3;

    return triplet;
}

/*
 * Free the memory allocated for a given triplet
 */
void free_triplet(Triplet *triplet, bool free_members) {
    if (triplet != NULL) {
        // Optionally free Matrix members
        if (free_members) {
            // Free first matrix
            if (triplet->m1 != NULL) {
                free_mat(triplet->m1);
            }
            // Free second matrix
            if (triplet->m2 != NULL) {
                free_mat(triplet->m2);
            }
            // Free third matrix
            if (triplet->m3 != NULL) {
                free_mat(triplet->m3);
            }
        }

        // Free triplet
        free(triplet);
        triplet = NULL;
    }
}