//
// Created by nihil on 03/10/21.
//

#ifndef FAST_ICA_SIGNAL_H
#define FAST_ICA_SIGNAL_H

#include "../include/vector.h"
#include "../include/matrix.h"

/*
 * Math operations
 */
Vector *sine_vec(Vector *v);
Vector *sgn_vec(Vector *v);
Vector *mod_vec(Vector *v, fp k);

/*
 * Signals
 */
Vector *generate_sine_wave(fp amp, fp freq, fp phase, int n_samples, fp range);
Vector *generate_square_wave(fp amp, fp freq, fp phase, int n_samples, fp range);
Vector *generate_sawtooth_wave(fp amp, fp freq, fp phase, int n_samples, fp range);
Matrix *generate_signals(int n_samples, fp range, bool add_noise);

#endif //FAST_ICA_SIGNAL_H
