#pragma once
#include <complex.h>
#include "_blas_helpers.h"

void zimatadd(int nrows,
              int ncols,
              double complex * out,
              double complex * in,
              double complex alpha,
              blasfunctions * blas_fu);
