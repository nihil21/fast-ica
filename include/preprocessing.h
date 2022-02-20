//
// Created by nihil on 02/10/21.
//

#ifndef FAST_ICA_PREPROCESSING_H
#define FAST_ICA_PREPROCESSING_H

#include "groups.h"

Pair *center(Matrix *x);
Pair *whitening(Matrix *x, bool center_data);

#endif //FAST_ICA_PREPROCESSING_H
