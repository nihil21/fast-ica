//
// Created by nihil on 08/10/21.
//

#ifndef FAST_ICA_FAST_ICA_H
#define FAST_ICA_FAST_ICA_H

#include "../include/matrix.h"

typedef enum FastICAStrategy {
    Deflation,
    Parallel
} FastICAStrategy;
typedef enum GFunc {
    LogCosh,
    Exp,
    Cube
} GFunc;

Matrix *fast_ica(Matrix *x, bool whiten, FastICAStrategy strategy, GFunc g_func, Matrix *w_init, fp threshold, int max_iter);

#endif //FAST_ICA_FAST_ICA_H
