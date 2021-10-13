//
// Created by nihil on 04/10/21.
//

#include <malloc.h>
#include "../include/tuple.h"
#include "../include/utils.h"

/*
 * Creates a tuple from a pair of tensors (and related type, namely matrix or vector)
 */
Tuple *new_tuple(Tensor *tensor1, TensorType type1, Tensor *tensor2, TensorType type2) {
    Tuple *tuple;
    tuple = malloc(sizeof(Tuple));
    assert(tuple != NULL, "Could not allocate tuple.");

    tuple->tensor1 = tensor1;
    tuple->type1 = type1;
    tuple->tensor2 = tensor2;
    tuple->type2 = type2;

    return tuple;
}

/*
 * Free the memory allocated for a given tuple
 */
void free_tuple(Tuple *tuple, bool free_members) {
    if (tuple != NULL) {
        // Optionally free Tensor members
        if (free_members) {
            // Free first tensor
            if (tuple->tensor1 != NULL) {
                switch (tuple->type1) {
                    case VecType:
                        free_vec((Vector *) tuple->tensor1);
                        break;
                    case MatType:
                        free_mat((Matrix *) tuple->tensor1);
                        break;
                }
            }
            // Free second tensor
            if (tuple->tensor2 != NULL) {
                switch (tuple->type2) {
                    case VecType:
                        free_vec((Vector *) tuple->tensor2);
                        break;
                    case MatType:
                        free_mat((Matrix *) tuple->tensor2);
                        break;
                }
            }
        }

        // Free tuple
        free(tuple);
        tuple = NULL;
    }
}