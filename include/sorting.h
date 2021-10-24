//
// Created by nihil on 24/10/21.
//

#ifndef FAST_ICA_SORTING_H
#define FAST_ICA_SORTING_H

#include <stdbool.h>
#include "fp.h"

typedef struct Data {
    int index;
    fp value;
} Data;

void quick_sort(Data v[], int len, bool desc);

#endif //FAST_ICA_SORTING_H
