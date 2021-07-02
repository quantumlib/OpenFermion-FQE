cdef extern from "_blas_helpers.h":
    ctypedef void (*zaxpy_func)(int * n,
                                double complex *alpha,
                                double complex *x,
                                int *incx,
                                double complex *y,
                                int *incy)

    cdef struct blasfunctions:
        zaxpy_func zaxpy
