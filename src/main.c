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
    int n_samples = 1000;
    fp sampling_window_size = 10;
    fp threshold = 1e-4f;
    int max_iter = 3000;
    bool add_noise = true;
    bool verbose = true;
    // Read input args
    char *end;
    switch (argc) {
        case 9:
            verbose = (int) strtol(argv[8], &end, 2);
        case 8:
            add_noise = (int) strtol(argv[7], &end, 2);
        case 7:
            max_iter = (int) strtol(argv[6], &end, 10);
            assert(max_iter > 0, "The maximum number of iteration must be positive.");
        case 6:
            threshold = (fp) strtof(argv[5], &end);
            assert(threshold > 0, "The threshold must be positive.");
        case 5:
            sampling_window_size = (fp) strtof(argv[4], &end);
            assert(sampling_window_size > 0, "The sampling window size must be positive.");
        case 4:
            n_samples = (int) strtol(argv[3], &end, 10);
            assert(n_samples > 0, "The number of samples must be positive.");
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
            printf("Usage: fast_ica [STRATEGY [G_FUNCTION [N_SAMPLES [SAMPLING_WINDOW_SIZE [THRESHOLD [ MAX_ITER [ADD_NOISE [VERBOSE]]]]]]]");
            exit(-1);
    }
    set_prng_seed(42);  // set seed

    // Create matrix S of original signals (n_components, n_samples)
    Matrix *s = generate_signals(n_samples, sampling_window_size, add_noise);
    if (verbose) {
        printf("Original signals:\n");
        print_mat(s);
    }

    write_mat("../S.bin", s);

    // Create mixing matrix A (n_components, n_components)
    fp a_data[] = {1, 1, 1, 0.5f, 2, 1, 1.5f, 1, 2, 2, 1, 1.7f};
    Matrix *a = from_array(a_data, 4, 3);
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

    write_mat("../X.bin", x);

    // Perform FastICA
    Matrix *s_ = fast_ica(x, 3, true, strategy, g_function, threshold, max_iter);
    if (verbose) {
        printf("Restored signals:\n");
        print_mat(s_);
    }
    write_mat("../S_.bin", s_);

    // Free memory
    free_mat(s);
    free_mat(a);
    free_mat(x);
    free_mat(s_);

    return 0;
}
