//
// Created by nihil on 24/10/21.
//

#include "../include/sorting.h"

/*
 * Swap two elements
 */
void swap(Data *a, Data *b){
    Data tmp = *a;
    *a = *b;
    *b = tmp;
}

/*
 * Compare two elements
 */
int compare(Data a, Data b){
    if (a.value > b.value)
        return 1;
    else if (a.value == b.value)
        return 0;

    return -1;
}

/*
 * Recursive sub-routine of QuickSort
 */
void quick_sort_rec(Data v[], int start, int end, int desc_fact){
    int i, j, i_pivot;
    Data pivot;

    if(start < end) {
        i = start;
        j = end;
        i_pivot = end;
        pivot = v[i_pivot];

        do {
            while (i < j && desc_fact * compare(v[i], pivot) <= 0)
                i++;
            while (j > i && desc_fact * compare(v[j], pivot) >= 0)
                j--;
            if (i < j)
                swap(&(v[i]), &(v[j]));
        } while (i < j);

        if (i != i_pivot && desc_fact * compare(v[i], v[i_pivot])) {
            swap(&(v[i]), &(v[i_pivot]));
            i_pivot = i;
        }

        if (start < i_pivot - 1)
            quick_sort_rec(v, start, i_pivot - 1, desc_fact);
        if (i_pivot + 1 < end)
            quick_sort_rec(v, i_pivot + 1, end, desc_fact);
    }
}

/*
 * QuickSort implementation
 */
void quick_sort(Data v[], int len, bool desc) {
    int desc_fact = desc ? -1 : 1;
    quick_sort_rec(v, 0, len - 1, desc_fact);
}