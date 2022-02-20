#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.h"
#include "../include/random.h"
#include "../include/matrix.h"
#include "../include/signal.h"
#include "../include/fast_ica.h"

int main(int argc, char **argv) {
    // Set default arguments
    FastICAStrategy strategy = Parallel;
    GFunc g_function = LogCosh;
    int g_selector;
    fp threshold = 1e-4f;
    int max_iter = 3000;
    int s_len = 10;
    int s_rate = 100;
    bool add_noise = true;
    bool verbose = false;
    // Read input args
    char *end;
    switch (argc) {
        case 9:
            verbose = (int) strtol(argv[8], &end, 2);
        case 8:
            add_noise = (int) strtol(argv[7], &end, 2);
        case 7:
            s_rate = (int) strtol(argv[6], &end, 10);
            assert(s_rate > 0, "The sampling rate must be positive.");
        case 6:
            s_len = (int) strtol(argv[5], &end, 10);
            assert(s_len > 0, "The length of the signal must be positive.");
        case 5:
            max_iter = (int) strtol(argv[4], &end, 10);
            assert(max_iter > 0, "The maximum number of iteration must be positive.");
        case 4:
            threshold = (fp) strtof(argv[3], &end);
            assert(threshold > 0, "The threshold must be positive.");
        case 3:
            g_selector = (int) strtol(argv[2], &end, 4);
            switch (g_selector) {
                case 0:
                    g_function = LogCosh;
                    break;
                case 1:
                    g_function = Exp;
                    break;
                case 2:
                    g_function = Cube;
                    break;
                case 3:
                    g_function = Abs;
                    break;
                default:
                    assert(false, "Unknown function.");
            }
        case 2:
            if ((int) strtol(argv[1], &end, 2))
                strategy = Deflation;
        case 1:
            break;
        default:
            printf("Usage: fast_ica [STRATEGY [G_FUNCTION [THRESHOLD [MAX_ITER [S_LEN [S_RATE [ADD_NOISE [VERBOSE]]]]]]]]");
            exit(-1);
    }
    int n_samples = s_len * s_rate;
    set_prng_seed(42);  // set seed

    // Create matrix S of original signals (n_components, n_samples)
    Matrix *s = generate_signals(n_samples, (fp) s_len, add_noise);
    // Standardize signal
    Matrix *s_std = col_std(s);
    div_col_(s, s_std);
    if (verbose) {
        printf("Original signals:\n");
        print_mat(s);
    }

    write_mat("./S.bin", s);

    // Create mixing matrix A (n_components, n_components)
    fp a_data[] = {1, 1, 1, 0.5f, 2, 1, 1.5f, 1, 2};
    Matrix *a = from_array(a_data, 3, 3);
    if (verbose) {
        printf("Mixing matrix:\n");
        print_mat(a);
    }

    // Create observation X by mixing signal S with matrix A (n_components, n_samples)
    Matrix *x = mat_mul(a, s);
    if (verbose) {
        printf("Observations (mixed signals):\n");
        print_mat(x);
        printf("\n");
    }

    write_mat("./X.bin", x);

    // Perform FastICA
    Matrix *s_ = fast_ica(x, true, strategy, g_function, threshold, max_iter);
    if (verbose) {
        printf("Restored signals:\n");
        print_mat(s_);
    }
    write_mat("./S_.bin", s_);

    // Free memory
    free_mat(s);
    free_mat(s_std);
    free_mat(a);
    free_mat(x);
    free_mat(s_);

    return 0;
}
