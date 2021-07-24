/*
    Copyright 2021 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/
#ifndef SRC_FQE_LIB__BLAS_HELPERS_H_
#define SRC_FQE_LIB__BLAS_HELPERS_H_

#include <complex.h>

typedef void (*zaxpy_func)(const int * n,
                           const double complex *alpha,
                           const double complex *x,
                           const int * incx,
                           double complex *y,
                           const int * incy);

typedef void (*zscal_func)(const int * n,
                           const double complex *alpha,
                           double complex *x,
                           const int * incx);

typedef struct blasfunctions {
        zaxpy_func zaxpy;
        zscal_func zscal;
} blasfunctions;

#endif  // SRC_FQE_LIB__BLAS_HELPERS_H_
