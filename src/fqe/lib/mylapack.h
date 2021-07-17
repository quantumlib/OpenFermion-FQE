#pragma once
#include <complex.h>

void zimatadd(const int nrows,
              const int ncols,
              double complex *out,
              const double complex *in,
              const double complex alpha);

void transpose(const int nrows,
               const int ncols,
               double complex *out,
               const double complex *in);
