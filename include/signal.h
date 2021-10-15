//
// Created by nihil on 03/10/21.
//

#ifndef FAST_ICA_SIGNAL_H
#define FAST_ICA_SIGNAL_H

#include "../include/matrix.h"

/*
 * Math operations
 */
Matrix *sine_mat(Matrix *m);
Matrix *sgn_mat(Matrix *m);
Matrix *mod_mat(Matrix *m, fp k);

/*
 * Signals
 */
Matrix *generate_sine_wave(fp amp, fp freq, fp phase, int n_samples, fp range);
Matrix *generate_square_wave(fp amp, fp freq, fp phase, int n_samples, fp range);
Matrix *generate_sawtooth_wave(fp amp, fp freq, fp phase, int n_samples, fp range);
Matrix *generate_signals(int n_samples, fp range, bool add_noise);

#endif //FAST_ICA_SIGNAL_H
