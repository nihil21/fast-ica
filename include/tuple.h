//
// Created by nihil on 04/10/21.
//

#ifndef FAST_ICA_TUPLE_H
#define FAST_ICA_TUPLE_H

#include "../include/matrix.h"

/*
 * Data type declaration
 */
typedef struct Tuple {
    Matrix *m1;
    Matrix *m2;
} Tuple;

/*
 * Initialize and free
 */
Tuple *new_tuple(Matrix *m1, Matrix *m2);
void free_tuple(Tuple *tuple, bool free_members);

#endif //FAST_ICA_TUPLE_H
