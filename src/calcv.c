#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

int comp_for_argsort_dbl(const void *a, const void *b)
{
    return (((double*)a)[0] > ((double*)b)[0])? 1 : -1;
}

int comp_for_argsort_szt(const void *a, const void *b)
{
    return (((size_t*)a)[0] > ((size_t*)b)[0])? 1 : -1;
}

void argsort_naive_dbl(double a[], size_t out[], double buffer[], size_t n)
{
    /* Note: this is a rather inefficient procedure and can be improve with e.g. C++'s sort */
    for (size_t i = 0; i < n; i++) {
        buffer[i * 2] = a[i];
        buffer[i * 2 + 1] = (double) i;
    }
    qsort(buffer, n, sizeof(double) * 2, comp_for_argsort_dbl);
    for (size_t i = 0; i < n; i++) { out[i] = (size_t) buffer[i * 2 + 1]; }
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
double *buffer_argsort_dbl;
size_t *buffer_argsort_szt;
double *cost_buffer;
double *rectangle_width_arr;
#pragma omp threadprivate(inner_order, out_order, buffer_argsort_dbl, buffer_argsort_szt, cost_buffer, rectangle_width_arr)

int calculate_V(double C[], double V[], size_t nrow, size_t ncol, int nthreads)
{
    int out_of_mem = 0;
    #pragma omp parallel shared(out_of_mem)
    {
        inner_order = (size_t*) malloc(sizeof(size_t) * ncol);
        out_order = (size_t*) malloc(sizeof(size_t) * ncol);
        buffer_argsort_dbl = (double*) malloc(sizeof(double) * ncol * 2);
        buffer_argsort_szt = (size_t*) malloc(sizeof(size_t) * ncol * 2);
        cost_buffer = (double*) malloc(sizeof(double) * ncol);
        rectangle_width_arr = (double*) malloc(sizeof(double) * (ncol - 1));
        if (inner_order == NULL || out_order == NULL || buffer_argsort_dbl == NULL || buffer_argsort_szt == NULL ||
            cost_buffer == NULL || rectangle_width_arr == NULL) {
            out_of_mem = 1;
        }
    }
    #pragma omp barrier
    {
        if (out_of_mem) {
            fprintf(stderr, "Error: Could not allocate memory for the procedure.\n");
            goto cleanup;
        }
    }

    #pragma omp parallel for schedule(static) num_threads(nthreads) firstprivate(C, V, nrow, ncol)
    for (size_t row = 0; row < nrow; row++) {
        argsort_naive_dbl(C + row * ncol, inner_order, buffer_argsort_dbl, ncol);
        calc_cost(C + row * ncol, cost_buffer, inner_order, ncol);
        argsort_naive_szt(inner_order, out_order, buffer_argsort_szt, ncol);
        calc_rectangle_width(cost_buffer, rectangle_width_arr, ncol);
        V[row * ncol] = 0;
        for (size_t col = 0; col < ncol - 1; col++) { V[row * ncol + col + 1] = V[row * ncol + col] + rectangle_width_arr[col] / ((double) col + 1); }
        sort_by_ix(V + row * ncol, out_order, buffer_argsort_dbl, ncol);
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
