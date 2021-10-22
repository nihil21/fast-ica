//
// Created by nihil on 03/10/21.
//

#include "../include/signal.h"
#include "../include/utils.h"

/*
 * Apply sine function to a matrix
 */
Matrix *sine_mat(Matrix *m) {
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = SIN(MAT_CELL(m, i, j));
        }
    }

    return s;
}

/*
 * Apply SGN function to a matrix
 */
Matrix *sgn_mat(Matrix *m) {
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = (fp) sgn(MAT_CELL(m, i, j));
        }
    }

    return s;
}

/*
 * Apply modulus function to a matrix
 */
Matrix *mod_mat(Matrix *m, fp k) {
    Matrix *s = new_mat(m->height, m->width);

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            MAT_CELL(s, i, j) = fmodf(MAT_CELL(m, i, j), k);
        }
    }

    return s;
}

Matrix *generate_sine_wave(fp amp, fp freq, fp phase, int n_samples, fp range) {
    // Generate a vector with n_samples values from 0 to range, representing time slices
    Matrix *t = linspace(0, range, n_samples);
    // Multiply it by angular velocity and translate it by phase (in-place)
    fp omega = 2 * PI * freq;
    scale_(&t, omega);
    add_scalar_(&t, phase);

    // Generate sine wave
    Matrix *st = sine_mat(t);
    // Multiply by amplitude (in-place)
    scale_(&st, amp);
    // Transpose into row vector
    Matrix *s = transpose(st);

    // Free memory
    free_mat(t);
    free_mat(st);

    return s;
}

Matrix *generate_square_wave(fp amp, fp freq, fp phase, int n_samples, fp range) {
    // Generate a vector with n_samples values from 0 to range, representing time slices
    Matrix *t = linspace(0, range, n_samples);
    // Multiply it by angular velocity and translate it by phase (in-place)
    fp omega = 2 * PI * freq;
    scale_(&t, omega);
    add_scalar_(&t, phase);

    // Generate sine wave
    Matrix *tmp = sine_mat(t);
    Matrix *st = sgn_mat(tmp);
    // Multiply by amplitude (in-place)
    scale_(&st, amp);
    // Transpose into row vector
    Matrix *s = transpose(st);

    // Free memory
    free_mat(t);
    free_mat(tmp);
    free_mat(st);

    return s;
}

Matrix *generate_sawtooth_wave(fp amp, fp freq, fp phase, int n_samples, fp range) {
    // Generate a vector with n_samples values from 0 to range, representing time slices
    Matrix *t = linspace(0, range, n_samples);
    // Multiply it by angular velocity and translate it by phase (in-place)
    fp omega = 2 * PI * freq;
    scale_(&t, omega);
    add_scalar_(&t, phase);

    // Generate sawtooth wave
    Matrix *st = mod_mat(t, 2 * PI);
    scale_(&st, 1 / PI);
    add_scalar_(&st, -1.f);
    // Multiply by amplitude (in-place)
    scale_(&st, amp);
    // Transpose into row vector
    Matrix *s = transpose(st);

    // Free memory
    free_mat(t);
    free_mat(st);

    return s;
}

Matrix *generate_signals(int n_samples, fp range, bool add_noise) {
    // First component: sine wave
    Matrix *s1 = generate_sine_wave(1.5f, 0.3f, -PI, n_samples, range);
    // Second component: square wave
    Matrix *s2 = generate_square_wave(1, 0.5f, 0, n_samples, range);
    // Third component: sawtooth wave
    Matrix *s3 = generate_sawtooth_wave(0.5f, 0.7f, PI, n_samples, range);

    // Stack them
    Matrix *s = new_mat(3, n_samples);
    paste_row(&s, s1, 0);
    paste_row(&s, s2, 1);
    paste_row(&s, s3, 2);

    // Apply gaussian noise
    if (add_noise) {
        Matrix *ns = mat_randn(s->height, s->width);
        scale_(&ns, 0.2f);
        add_mat_(&s, ns);
        free_mat(ns);
    }

    // Free memory
    free_mat(s1);
    free_mat(s2);
    free_mat(s3);

    return s;
}
