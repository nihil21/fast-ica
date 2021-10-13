#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/utils.h"
#include "../include/vector.h"
#include "../include/matrix.h"
#include "../include/signal.h"
#include "../include/fast_ica.h"

int main(int argc, char **argv) {
    // Set default arguments
    FastICAStrategy strategy = Deflation;
    GFunc g_function = LogCosh;
    int g_selector;
    int n_samples = 1000;
    fp sampling_window_size = 10;
    fp threshold = 1e-4f;
    int max_iter = 3000;
    bool verbose = true;
    // Read input args
    char *end;
    switch (argc) {
        case 8:
            verbose = (int) strtol(argv[7], &end, 2);
        case 7:
            max_iter = (int) strtol(argv[6], &end, 10);
            assert(sampling_window_size >= 0, "The maximum number of iteration must be non-negative.");
        case 6:
            threshold = (fp) strtol(argv[5], &end, 10);
            assert(sampling_window_size >= 0, "The threshold must be non-negative.");
        case 5:
            sampling_window_size = (fp) strtol(argv[4], &end, 10);
            assert(sampling_window_size >= 0, "The sampling window size must be non-negative.");
        case 4:
            n_samples = (int) strtol(argv[3], &end, 10);
            assert(sampling_window_size >= 0, "The number of samples must be non-negative.");
        case 3:
            g_selector = (int) strtol(argv[2], &end, 3);
            g_function = g_selector == 0 ? LogCosh : (g_selector == 1 ? Exp : Cube);
        case 2:
            if ((int) strtol(argv[1], &end, 2))
                strategy = Parallel;
        case 1:
            break;
        default:
            printf("Usage: fast_ica [STRATEGY [G_FUNCTION [N_SAMPLES [SAMPLING_WINDOW_SIZE [THRESHOLD [ MAX_ITER [VERBOSE]]]]]]");
            exit(-1);
    }
    srand48(time(NULL));  // set seed

    // Create matrix S of original signals (n_components, n_samples)
    Matrix *s = generate_signals(n_samples, sampling_window_size);
    if (verbose) {
        printf("Original signals:\n");
        mat_print(s);
    }

    mat_write("../S.bin", s);

    // Create mixing matrix A (n_components, n_components)
    fp a_data[] = {1, 1, 1, 0.5f, 2, 1, 1.5f, 1, 2};
    Matrix *a = mat_from_array(a_data, 3, 3);
    if (verbose) {
        printf("Mixing matrix:\n");
        mat_print(a);
    }

    // Create observation X by mixing signal S with matrix A (n_components, n_samples)
    Matrix *x = mat_mul(a, s);
    if (verbose) {
        printf("Observations (mixed signals):\n");
        mat_print(x);
        printf("\n");
    }

    mat_write("../X.bin", x);

    // Perform FastICA
    Matrix *s_res = fast_ica(x, true, strategy, g_function, NULL, threshold, max_iter);
    if (verbose) {
        printf("Restored signals:\n");
        mat_print(s_res);
    }

    mat_write("../S_res.bin", s_res);

    // Free memory
    free_mat(s);
    free_mat(a);
    free_mat(x);
    free_mat(s_res);

    return 0;
}
