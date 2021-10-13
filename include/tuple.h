//
// Created by nihil on 04/10/21.
//

#ifndef FAST_ICA_TUPLE_H
#define FAST_ICA_TUPLE_H

#include "../include/vector.h"
#include "../include/matrix.h"

/*
 * Data type declaration
 */
typedef union Tensor {
    Vector vec;
    Matrix mat;
} Tensor;
typedef enum TensorType {
    VecType,
    MatType
} TensorType;
typedef struct Tuple {
    Tensor *tensor1;
    TensorType type1;
    Tensor *tensor2;
    TensorType type2;
} Tuple;

/*
 * Initialize and free
 */
Tuple *new_tuple(Tensor *tensor1, TensorType type1, Tensor *tensor2, TensorType type2);
void free_tuple(Tuple *tuple, bool free_members);

#endif //FAST_ICA_TUPLE_H
