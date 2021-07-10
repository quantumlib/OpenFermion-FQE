#pragma once
#include <complex.h>

typedef void (*zaxpy_func)(const int * n,
                           const double complex *alpha,
                           const double complex *x,
                           const int * incx,
                           double complex *y,
                           const int * incy);

typedef struct blasfunctions {
        zaxpy_func zaxpy;
} blasfunctions;
