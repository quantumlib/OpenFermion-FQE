#pragma once
#include <complex.h>
#include <stdbool.h>
#include "_blas_helpers.h"

int lm_apply_array12_same_spin(const double complex *coeff,
                               double complex *out,
                               const int (*dexc)[3],
                               const int alpha_states,
                               const int beta_states,
                               const int ndexc,
                               const double complex *h2e,
                               const int norbs,
                               const bool is_alpha,
                               const struct blasfunctions * blasfunc);

int lm_apply_array12_diff_spin(const double complex *coeff,
                               double complex *out,
                               const int (*adexc)[3],
                               const int (*bdexc)[3],
                               const int alpha_states,
                               const int beta_states,
                               const int nadexc,
                               const int nbdexc,
                               const double complex *h2e,
                               const int norbs);

int lm_apply_array1(const double complex *coeff,
                    double complex *out,
                    const int (*dexc)[3],
                    const int astates,
                    const int bstates,
                    const int ndexc,
                    const double complex *h1e,
                    const int norbs,
                    const bool is_alpha,
                    const struct blasfunctions * blasfunc);

int zdvec_make(const int (**map)[3],
               const int *map_els,
               const int nr_maps,
               const double complex *coeff,
               double complex *dvec,
               const int alpha_states,
               const int beta_states,
               const bool is_alpha,
               const struct blasfunctions * blasfunc);

int zcoeff_make(const int (**map)[3],
                const int *map_els,
                const int nr_maps,
                double complex *coeff,
                const double complex * dvec,
                const int alpha_states,
                const int beta_states,
                const bool is_alpha,
                const struct blasfunctions * blasfunc);

int zdiagonal_coulomb(const unsigned int *alpha_strings,
                      const unsigned int *beta_strings,
                      const double complex *diag,
                      const double complex *array,
                      double complex *output,
                      const int alpha_states,
                      const int beta_states,
                      const int nalpha,
                      const int nbeta,
                      const int norbs);

int make_dvec_part(const int astates_full,
                   const int bstates_full,
                   const int astates_part_begin,
                   const int bstates_part_begin,
                   const int astates_part_num,
                   const int bstates_part_num,
                   const int (*maps)[4],
                   const int nmaps,
                   const double complex * input,
                   double complex * output,
                   bool isalph,
                   const struct blasfunctions * blasfunc);

int make_coeff_part(const int states1_full,
                   const int states2_full,
                   const int states1_part_begin,
                   const int states2_part_begin,
                   const int states1_part_num,
                   const int states2_part_num,
                   const int (*maps)[4],
                   const int nmaps,
                   const double complex *input,
                   double complex *output,
                   const struct blasfunctions * blasfunc);

int lm_apply_array12_same_spin_opt(const double complex *coeff,
                                   double complex *out,
                                   const int *dexc,
                                   const int alpha_states,
                                   const int beta_states,
                                   const int ndexc,
                                   const double complex *h1e,
                                   const double complex *h2e,
                                   const int norbs,
                                   const bool is_alpha,
                                   const struct blasfunctions * blasfunc);

int lm_apply_array12_diff_spin_opt(const double complex *coeff,
                                   double complex *out,
                                   const int *adexc,
                                   const int *bdexc,
                                   const int alpha_states,
                                   const int beta_states,
                                   const int nadexc,
                                   const int nbdexc,
                                   const double complex *h2e,
                                   const int norbs,
                                   const struct blasfunctions * blasfunc);
