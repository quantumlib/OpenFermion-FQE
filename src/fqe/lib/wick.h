#pragma once
#include <complex.h>

int wickfill(double complex *target,
             const double complex *source,
             const unsigned int *indices,
             const double factor,
             const unsigned int *delta,
             const int norb,
             const int trank,
             const int srank);
