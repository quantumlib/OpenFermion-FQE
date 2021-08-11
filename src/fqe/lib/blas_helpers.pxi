cdef extern from "_blas_helpers.h":
    ctypedef void (*zaxpy_func)(const int * n,
                                const double complex *alpha,
                                const double complex *x,
                                const int *incx,
                                double complex *y,
                                const int *incy)
    ctypedef void (*zscal_func)(const int * n,
                                const double complex *alpha,
                                double complex *x,
                                const int *incx)

    cdef struct blasfunctions:
        zaxpy_func zaxpy
        zscal_func zscal
