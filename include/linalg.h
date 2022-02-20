//
// Created by nihil on 20/09/21.
//

#ifndef FAST_ICA_LINALG_H
#define FAST_ICA_LINALG_H

#include "../include/matrix.h"
#include "../include/groups.h"

Matrix *generate_householder(Matrix *m, fp *tau);
Matrix *to_hessenberg(Matrix *m);
Pair *qr_decomposition(Matrix *m);
Pair *solve_eig(Matrix *m);
Matrix *lin_solve(Matrix *a, Matrix *b);
Triplet *svd(Matrix *x);
Matrix *covariance(Matrix *x, bool center_data);

#endif //FAST_ICA_LINALG_H
