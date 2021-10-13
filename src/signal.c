//
// Created by nihil on 03/10/21.
//

#include <math.h>
#include "../include/signal.h"
#include "../include/utils.h"

/*
 * Apply sine function to vector
 */
Vector *sine_vec(Vector *v) {
    Vector *s = new_vec(v->length);

    for (int i = 0; i < v->length; i++) {
        VEC_CELL(s, i) = SIN(VEC_CELL(v, i));
    }

    return s;
}

/*
 * Apply SGN function to vector
 */
Vector *sgn_vec(Vector *v) {
    Vector *s = new_vec(v->length);

    for (int i = 0; i < v->length; i++) {
        VEC_CELL(s, i) = (fp) sgn(VEC_CELL(v, i));
    }

    return s;
}

/*
 * Apply modulus function to vector
 */
Vector *mod_vec(Vector *v, fp k) {
    Vector *s = new_vec(v->length);

    for (int i = 0; i < v->length; i++) {
        VEC_CELL(s, i) = fmodf(VEC_CELL(v, i), k);
    }

    return s;
}

Vector *generate_sine_wave(fp amp, fp freq, fp phase, int n_samples, fp range) {
    // Generate a vector with n_samples values from 0 to range, representing time slices
    Vector *t = linspace(0, range, n_samples);
    // Multiply it by angular velocity and translate it by phase (in-place)
    fp omega = 2 * PI * freq;
    vec_scale_(&t, omega);
    vec_scalar_add_(&t, phase);

    // Generate sine wave
    Vector *s = sine_vec(t);
    // Multiply by amplitude (in-place)
    vec_scale_(&s, amp);

    // Free temp vector
    free_vec(t);

    return s;
}

Vector *generate_square_wave(fp amp, fp freq, fp phase, int n_samples, fp range) {
    // Generate a vector with n_samples values from 0 to range, representing time slices
    Vector *t = linspace(0, range, n_samples);
    // Multiply it by angular velocity and translate it by phase (in-place)
    fp omega = 2 * PI * freq;
    vec_scale_(&t, omega);
    vec_scalar_add_(&t, phase);

    // Generate sine wave
    Vector *tmp = sine_vec(t);
    Vector *s = sgn_vec(tmp);
    // Multiply by amplitude (in-place)
    vec_scale_(&s, amp);

    // Free temp vectors
    free_vec(t);
    free_vec(tmp);

    return s;
}

Vector *generate_sawtooth_wave(fp amp, fp freq, fp phase, int n_samples, fp range) {
    // Generate a vector with n_samples values from 0 to range, representing time slices
    Vector *t = linspace(0, range, n_samples);
    // Multiply it by angular velocity and translate it by phase (in-place)
    fp omega = 2 * PI * freq;
    vec_scale_(&t, omega);
    vec_scalar_add_(&t, phase);

    // Generate sawtooth wave
    Vector *s = mod_vec(t, 2 * PI);
    vec_scale_(&s, 1 / PI);
    vec_scalar_add_(&s, -1.f);
    // Multiply by amplitude (in-place)
    vec_scale_(&s, amp);

    // Free temp vectors
    free_vec(t);

    return s;
}

Matrix *generate_signals(int n_samples, fp range) {
    // First component: sine wave
    Vector *s1 = generate_sine_wave(1.5f, 0.3f, -PI, n_samples, range);
    // Second component: square wave
    Vector *s2 = generate_square_wave(1, 0.5f, 0, n_samples, range);
    // Third component: sawtooth wave
    Vector *s3 = generate_sawtooth_wave(0.5f, 0.7f, PI, n_samples, range);

    // Stack them
    Matrix *s = new_mat(3, n_samples);
    paste_row(&s, s1, 0);
    paste_row(&s, s2, 1);
    paste_row(&s, s3, 2);

    // Apply gaussian noise
    Matrix *ns = mat_randn(s->height, s->width);
    mat_scale_(&ns, 0.2f);
    mat_add_(&s, ns);

    // Free temp vectors
    free_vec(s1);
    free_vec(s2);
    free_vec(s3);
    free_mat(ns);

    return s;
}
