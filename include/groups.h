//
// Created by nihil on 04/10/21.
//

#ifndef FAST_ICA_GROUPS_H
#define FAST_ICA_GROUPS_H

#include "../include/matrix.h"

/*
 * Data type declaration
 */
typedef struct Pair {
    Matrix *m1;
    Matrix *m2;
} Pair;

typedef struct Triplet {
    Matrix *m1;
    Matrix *m2;
    Matrix *m3;
} Triplet;

/*
 * Initialize and free
 */
Pair *new_pair(Matrix *m1, Matrix *m2);
void free_pair(Pair *pair, bool free_members);
Triplet *new_triplet(Matrix *m1, Matrix *m2, Matrix *m3);
void free_triplet(Triplet *triplet, bool free_members);

#endif //FAST_ICA_GROUPS_H
