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
 * Function returning the sign of a number
 */
int sgn(fp x) {
    return (x >= 0) ? 1 : -1;
}