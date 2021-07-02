#pragma once
#include <complex.h>

typedef void (*zaxpy_func)(int * n,
                           double complex *alpha,
                           double complex *x,
                           int *incx,
                           double complex *y,
                           int *incy);

typedef struct blasfunctions {
        zaxpy_func zaxpy;
} blasfunctions;
