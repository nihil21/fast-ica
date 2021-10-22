//
// Created by nihil on 22/10/21.
//

#ifndef FAST_ICA_RANDOM_H
#define FAST_ICA_RANDOM_H

#include "fp.h"

void set_prng_seed(unsigned int seed);
fp uniform_rand();
fp uniform_rand_range(fp min, fp max);
int uniform_randint_range(int min, int max);
fp standard_rand();

#endif //FAST_ICA_RANDOM_H
