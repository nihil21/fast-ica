//
// Created by nihil on 24/10/21.
//

#include "../include/sorting.h"
#include <malloc.h>

/*
 * Swap two elements
 */
void swap(fp *a, fp *b){
    fp tmp = *a;
    *a = *b;
    *b = tmp;
}

/*
 * Swap two indexes
 */
void swap_i(int *a, int *b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/*
 * Compare two elements
 */
int compare(fp a, fp b){
    if (a > b)
        return 1;
    else if (a == b)
        return 0;

    return -1;
}

/*
 * Recursive sub-routine of QuickSort
 */
void quick_sort_rec(fp v[], int *sort_id, int start, int end, int desc_fact){
    int i, j, i_pivot;
    fp pivot;

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
            if (i < j) {
                swap(&(v[i]), &(v[j]));
                swap_i(&(sort_id[i]), &(sort_id[j]));
            }
        } while (i < j);

        if (i != i_pivot && desc_fact * compare(v[i], v[i_pivot])) {
            swap(&(v[i]), &(v[i_pivot]));
            swap_i(&(sort_id[i]), &(sort_id[i_pivot]));
            i_pivot = i;
        }

        if (start < i_pivot - 1)
            quick_sort_rec(v, sort_id, start, i_pivot - 1, desc_fact);
        if (i_pivot + 1 < end)
            quick_sort_rec(v, sort_id, i_pivot + 1, end, desc_fact);
    }
}

/*
 * QuickSort implementation (it returns also the list of sorted indexes)
 */
int *quick_sort(fp v[], int len, bool desc) {
    int desc_fact = desc ? -1 : 1;
    // Create array of indexes
    int *sort_id = malloc(len * sizeof(int));
    for (int i = 0; i < len; i++)
        sort_id[i] = i;

    quick_sort_rec(v, sort_id, 0, len - 1, desc_fact);
    return sort_id;
}