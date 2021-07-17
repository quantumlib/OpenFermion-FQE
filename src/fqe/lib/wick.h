#pragma once
#include <stdint.h>
#include <complex.h>

int wickfill(double complex *target,
             const double complex *source,
             const uint32_t *indices,
             const double factor,
             const uint32_t *delta,
             const int norb,
             const int trank,
             const int srank);
