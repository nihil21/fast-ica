//
// Created by nihil on 20/09/21.
//

#ifndef FAST_ICA_LINALG_H
#define FAST_ICA_LINALG_H

#include "../include/matrix.h"
#include "../include/tuple.h"

Matrix *to_hessenberg(Matrix *m);
Tuple *qr_decomposition(Matrix *m);
Tuple *solve_eig(Matrix *m);
Vector *lin_solve(Matrix *a, Vector *b);
Matrix *covariance(Matrix *x, bool center_data);

#endif //FAST_ICA_LINALG_H
