#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <math.h>
#ifdef _FOR_R
    #include <R_ext/Print.h>
    #define fprintf(f, message) REprintf(message)
#else
    #include <stdio.h>
#endif

/* TODO: use qsort_s for argsorting, or switch to C++ std::sort */

typedef struct indexed_double {double x; size_t ix;} indexed_double;

int comp_for_argsort_dbl(const void *a, const void *b)
{
    return ( ((indexed_double*)a)->x > ((indexed_double*)b)->x )? 1 : -1;
}

int comp_for_argsort_szt(const void *a, const void *b)
{
    return (((size_t*)a)[0] > ((size_t*)b)[0])? 1 : -1;
}

void argsort_naive_dbl(double a[], size_t out[], indexed_double buffer[], size_t n)
{
    /* Note: this is a rather inefficient procedure and can be improve with e.g. C++'s sort */
    for (size_t i = 0; i < n; i++) {
        buffer[i].x = a[i];
        buffer[i].ix = i;
    }
    qsort(buffer, n, sizeof(indexed_double), comp_for_argsort_dbl);
    for (size_t i = 0; i < n; i++) { out[i] = buffer[i].ix; }
}

void argsort_naive_szt(size_t a[], size_t out[], size_t buffer[], size_t n)
{
    for (size_t i = 0; i < n; i++) {
        buffer[i * 2] = a[i];
        buffer[i * 2 + 1] = i;
    }
    qsort(buffer, n, sizeof(size_t) * 2, comp_for_argsort_szt);
    for (size_t i = 0; i < n; i++) { out[i] = buffer[i * 2 + 1]; }
}

double find_min(double a[], size_t n)
{
    double out = HUGE_VAL;
    for (size_t i = 0; i < n; i++) { out = (out > a[i])? a[i] : out; }
    return out;
}

void calc_cost(double row_C[], double out[], size_t inner_order[], size_t ncol)
{
    double min_in_row = find_min(row_C, ncol);
    for (size_t i = 0; i < ncol; i++) { out[i] = row_C[inner_order[i]] - min_in_row; }
}

void calc_rectangle_width(double cost[], double out[], size_t ncol)
{
    for (size_t i = 0; i < ncol - 1; i++) { out[i] = cost[i + 1] - cost[i]; }
}

void sort_by_ix(double a[], size_t ix[], double buffer[], size_t n)
{
    for (size_t i = 0; i < n; i++){ buffer[i] = a[ix[i]]; }
    memcpy(a, buffer, sizeof(double) * n);
}

size_t *inner_order;
size_t *out_order;
indexed_double *buffer_argsort_dbl;
size_t *buffer_argsort_szt;
double *cost_buffer;
double *rectangle_width_arr;
#pragma omp threadprivate(inner_order, out_order, buffer_argsort_dbl, buffer_argsort_szt, cost_buffer, rectangle_width_arr)

int calculate_V(double C[], double V[], size_t nrow, size_t ncol, int nthreads)
{
    int out_of_mem = 0;

    /* Note: MSVC is stuck with an older version of OpenMP (17 years old at the time or writing this)
       which does not support 'max' reductions */
    #ifdef _OPENMP
        #if !defined(_MSC_VER) && _OPENMP>20080101
            #pragma omp parallel reduction(max:out_of_mem)
        #endif
    #endif
    {
        inner_order = (size_t*) malloc(sizeof(size_t) * ncol);
        out_order = (size_t*) malloc(sizeof(size_t) * ncol);
        buffer_argsort_dbl = (indexed_double*) malloc(sizeof(indexed_double) * ncol);
        buffer_argsort_szt = (size_t*) malloc(sizeof(size_t) * ncol * 2);
        cost_buffer = (double*) malloc(sizeof(double) * ncol);
        rectangle_width_arr = (double*) malloc(sizeof(double) * (ncol - 1));
        if (inner_order == NULL || out_order == NULL || buffer_argsort_dbl == NULL || buffer_argsort_szt == NULL ||
            cost_buffer == NULL || rectangle_width_arr == NULL) {
            out_of_mem = 1;
        }
    }
    
    if (out_of_mem) {
        fprintf(stderr, "Error: Could not allocate memory for the procedure.\n");
        goto cleanup;
    }

    #pragma omp parallel for schedule(static) num_threads(nthreads) firstprivate(C, V, nrow, ncol)
    for (size_t row = 0; row < nrow; row++) {
        argsort_naive_dbl(C + row * ncol, inner_order, buffer_argsort_dbl, ncol);
        calc_cost(C + row * ncol, cost_buffer, inner_order, ncol);
        argsort_naive_szt(inner_order, out_order, buffer_argsort_szt, ncol);
        calc_rectangle_width(cost_buffer, rectangle_width_arr, ncol);
        V[row * ncol] = 0;
        for (size_t col = 0; col < ncol - 1; col++) { V[row * ncol + col + 1] = V[row * ncol + col] + rectangle_width_arr[col] / ((double) col + 1); }
        sort_by_ix(V + row * ncol, out_order, cost_buffer, ncol);
    }

    cleanup:
        #pragma omp parallel
        {
            free(inner_order);
            free(buffer_argsort_dbl);
            free(cost_buffer);
            free(rectangle_width_arr);
        }
    return out_of_mem;
}
