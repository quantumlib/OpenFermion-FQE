#include <stdlib.h>

#include "mylapack.h"
#define N_ 16

void zimatadd(const int nrows,
              const int ncols,
              double complex *out,
              const double complex *in,
              const double complex alpha) {
  const int m = nrows;
  const int n = ncols;

  const int mresidual = m % N_;
  const int nresidual = n % N_;
  const int mlim = m - mresidual;
  const int nlim = n - nresidual;
#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < nlim; i += N_) {
    for (int j = 0; j < mlim; j += N_) {
      for (int ii = 0; ii != N_; ++ii)
        for (int jj = 0; jj != N_; ++jj)
          out[i + ii + n * (j + jj)] += alpha * in[j + jj + m * (i + ii)];
    }
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nlim; i += N_) {
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
}


void transpose(const int nrows,
               const int ncols,
               double complex *out,
               const double complex *in) {
  const int m = nrows;
  const int n = ncols;
  // TODO when MKL is linked, this can be replaced by 
  // mkl_zomatcopy_("c", "t", &m, &n, &fac, in, &m, out, &n);

  const int mresidual = m % N_;
  const int nresidual = n % N_;
  const int mlim = m - mresidual;
  const int nlim = n - nresidual;
#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < nlim; i += N_) {
    for (int j = 0; j < mlim; j += N_) {
      for (int ii = 0; ii != N_; ++ii)
        for (int jj = 0; jj != N_; ++jj)
          out[i + ii + n * (j + jj)] = in[j + jj + m * (i + ii)];
    }
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nlim; i += N_) {
    for (int j = mlim; j != m; ++j) {
      for (int ii = 0; ii != N_; ++ii)
        out[i + ii + n * j] = in[j  + m * (i + ii)];
    }
  }
  for (int i = nlim; i != n; ++i) {
    for (int j = 0; j != mlim; j += N_) {
      for (int jj = 0; jj != N_; ++jj)
        out[i + n * (j + jj)] = in[j + jj + m * i];
    }
    for (int j = mlim; j !=  m; ++j) {
      out[i + n * j] = in[j + m * i];
    }
  }
}
