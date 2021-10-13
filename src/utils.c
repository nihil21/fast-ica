//
// Created by nihil on 21/09/21.
//

#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.h"

/*
 * Verify if condition holds, otherwise terminate program
 */
void assert(bool condition, char *message) {
    if (!condition) {
        fprintf(stderr, "%s\n", message);
        exit(-1);
    }
}

/*
 * Normal number generator (Marsaglia polar method)
 */
fp gen_normal() {
    static fp cached = 0;
    fp r;

    if (cached != 0) {  // result cached
        r = cached;
        cached = 0;
    } else {
        fp u, v, s;
        // Generate two uniform values between -1 and 1 s.t. the sum of their squares is less than 1
        do {
            u = (fp) drand48() * 2 - 1;
            v = (fp) drand48() * 2 - 1;
            s = u * u + v * v;
        } while (s >= 1);
        // Obtain two normal_mat variables
        r = u * SQRT((-2 * LOG(s)) / s);
        cached = v * SQRT((-2 * LOG(s)) / s);
    }

    return r;
}

/*
 * Function returning the sign of a number
 */
int sgn(fp x) {
    return (x >= 0) ? 1 : -1;
}