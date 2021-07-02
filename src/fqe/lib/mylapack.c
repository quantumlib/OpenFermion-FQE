#include <stdlib.h>

#include "mylapack.h"

void zimatadd(int nrows,
              int ncols,
              double complex * out,
              double complex * in,
              double complex alpha,
              blasfunctions * blas_fu) {
#ifndef USE_NEW_CODE
  const int ONE = 1;
#pragma omp parallel for schedule(static)
  for (int row = 0; row < nrows; ++row) {
    blas_fu->zaxpy(&ncols, &alpha, &in[row], &nrows, &out[row * ncols], &ONE);
  }
#else
#define N_ 16
  // variables m and n can be eliminated
  const int m = nrows;
  const int n = ncols;

  const int mresidual = m % N_;
  const int nresidual = n % N_;
  const int mlim = m - mresidual;
  const int nlim = n - nresidual;
  for (int i = 0; i < nlim; i += N_) {
    for (int j = 0; j < mlim; j += N_) {
      for (int ii = 0; ii != N_; ++ii)
        for (int jj = 0; jj != N_; ++jj)
          out[i + ii + n * (j + jj)] += alpha * in[j + jj + m * (i + ii)];
    }
    for (int j = mlim; j != m; ++j) {
      for (int ii = 0; ii != N_; ++ii)
        out[i + ii + n * j] += alpha * in[j  + m * (i + ii)];
    }
  }
  for (int i = nlim; i != n; ++i) {
    for (int j = 0; j != mlim; j += N_) {
      for (int jj = 0; jj != N_; ++jj)
        out[i + n * (j + jj)] += alpha * in[j + jj + m * i];
    }
    for (int j = mlim; j !=  m; ++j) {
      out[i + n * j] += alpha * in[j + m * i];
    }
  }
#endif
}
