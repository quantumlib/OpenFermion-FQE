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

#ifndef FQE_DATA_H_
#define FQE_DATA_H_

#include <complex.h>
#include <stdbool.h>
#include <stdint.h>
#include "_blas_helpers.h"

void lm_apply_array12_same_spin(const double complex *coeff,
                               double complex *out,
                               const int (*dexc)[3],
                               const int alpha_states,
                               const int beta_states,
                               const int ndexc,
                               const double complex *h2e,
                               const int norbs,
                               const bool is_alpha,
                               const struct blasfunctions * blasfunc);

void lm_apply_array12_diff_spin(const double complex *coeff,
                               double complex *out,
                               const int (*adexc)[3],
                               const int (*bdexc)[3],
                               const int alpha_states,
                               const int beta_states,
                               const int nadexc,
                               const int nbdexc,
                               const double complex *h2e,
                               const int norbs);

int sparse_apply_array1(const double complex *coeff,
                        double complex *out,
                        const int *dexc,
                        const int astates,
                        const int bstates,
                        const int ndexc,
                        const double complex *h1e,
                        const int norbs,
                        const int jorb,
                        const bool is_alpha,
                        const struct blasfunctions * blasfunc);

void lm_apply_array1(const double complex *coeff,
                    double complex *out,
                    const int *dexc,
                    const int astates,
                    const int bstates,
                    const int ndexc,
                    const double complex *h1e,
                    const int norbs,
                    const bool is_alpha,
                    const struct blasfunctions * blasfunc);

void lm_apply_array1_column_alpha(double complex *coeff,
                                  const int *index,
                                  const int *dexc,
                                  const int *dexc2,
                                  const int astates,
                                  const int bstates,
                                  const int nexc0,
                                  const int nexc1,
                                  const int nexc2,
                                  const double complex *h1e,
                                  const int norbs,
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

void zdiagonal_coulomb_apply(const uint64_t *alpha_strings,
                            const uint64_t *beta_strings,
                            const double complex *diag,
                            const double complex *array,
                            double complex *output,
                            const int alpha_states,
                            const int beta_states,
                            const int nalpha,
                            const int nbeta,
                            const int norbs);

int zdiagonal_coulomb(const uint64_t *alpha_strings,
                      const uint64_t *beta_strings,
                      const double complex *diag,
                      const double complex *array,
                      double complex *output,
                      const int alpha_states,
                      const int beta_states,
                      const int nalpha,
                      const int nbeta,
                      const int norbs);

void make_dvec_part(const int astates_full,
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

void make_coeff_part(const int states1_full,
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

void apply_individual_nbody1_accumulate(const double complex coeff,
                                        double complex *ocoeff,
                                        const double complex * icoeff,
                                        const int n,
                                        const int nao,
                                        const int nbo,
                                        const int nai,
                                        const int nbi,
                                        const int nt,
                                        const int64_t *amap,
                                        const int64_t *btarget,
                                        const int64_t *bsource,
                                        const int64_t *bparity);

void lm_apply_array12_same_spin_opt(const double complex *coeff,
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

void lm_apply_array12_diff_spin_opt(const double complex *coeff,
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

void apply_array12_lowfillingab(const double complex *coeff,
                               const int *alpha_array,
                               const int *beta_array,
                               const int nalpha,
                               const int nbeta,
                               const int na1,
                               const int nb1,
                               const int nca,
                               const int ncb,
                               const int nia,
                               const int nib,
                               const int norb,
                               double complex *intermediate);

void apply_array12_lowfillingab2(const double complex *intermediate,
                                const int *alpha_array,
                                const int *beta_array,
                                const int nalpha,
                                const int nbeta,
                                const int na1,
                                const int nb1,
                                const int nia,
                                const int nib,
                                const int noa,
                                const int nob,
                                const int norb,
                                double complex *out);

void apply_array12_lowfillingaa(const double complex *coeff,
                               const int *alpha_array,
                               const bool alpha,
                               const int nlt,
                               const int na,
                               const int ni1,
                               const int ni2,
                               const int ni3,
                               const int nc1,
                               const int nc2,
                               double complex *intermediate);

void apply_array12_lowfillingaa2(const double complex *intermediate,
                               const int *alpha_array,
                               const bool alpha,
                               const int nlt,
                               const int na,
                               const int ni1,
                               const int ni2,
                               const int ni3,
                               const int no1,
                               const int no2,
                               double complex *out);

void make_Hcomp(const int norb,
                const int nlt,
                const double complex *h2e,
                double complex *h2ecomp);

void from_cirq(double complex * const cirq_wfn,
               double complex * const fqe_wfn,
               const int alpha_states,
               const int beta_states,
               const long long * cirq_alpha_id,
               const long long * cirq_beta_id,
               const int * const alpha_swaps,
               const int * const beta_occs,
               const int nbeta,
               const int norbs);

void to_cirq(double complex * const cirq_wfn,
             double complex * const fqe_wfn,
             const int alpha_states,
             const int beta_states,
             const long long * cirq_alpha_id,
             const long long * cirq_beta_id,
             const int * const alpha_swaps,
             const int * const beta_occs,
             const int nbeta,
             const int norbs);

void sparse_scale(const long long *xi,
                  const long long *yi,
                  const double complex fac,
                  const int ni,
                  const int nd1,
                  const int nd2,
                  double complex *data);

void integer_index_accumulate_real(double *out,
                                   const double *data,
                                   uint64_t string);

void integer_index_accumulate(double complex *out,
                              const double complex *data,
                              uint64_t string);

void apply_diagonal_inplace_real(double complex *data,
                                 const double *aarray,
                                 const double *barray,
                                 const uint64_t *astrings,
                                 const uint64_t *bstrings,
                                 const int lena,
                                 const int lenb);

void apply_diagonal_inplace(double complex *data,
                            const double complex *aarray,
                            const double complex *barray,
                            const uint64_t *astrings,
                            const uint64_t *bstrings,
                            const int lena,
                            const int lenb);

void evolve_diagonal_inplace_real(double complex *data,
                                  const double *aarray,
                                  const double *barray,
                                  const uint64_t *astrings,
                                  const uint64_t *bstrings,
                                  const int lena,
                                  const int lenb);

void evolve_diagonal_inplace(double complex *data,
                             const double complex *aarray,
                             const double complex *barray,
                             const uint64_t *astrings,
                             const uint64_t *bstrings,
                             const int lena,
                             const int lenb);

int evaluate_map_each(int64_t *out,
                      const uint64_t *strings,
                      const int length,
                      const int pmask,
                      const int hmask);

void calculate_dvec1(const int32_t *aarray,
                     const int32_t *barray,
                     const int norb,
                     const int nalpha,
                     const int nbeta,
                     const int na,
                     const int nb,
                     const int nc1,
                     const int nc2,
                     const int nd1,
                     const int nd2,
                     const int nd3,
                     const int nd4,
                     const double complex *coeff,
                     double complex *dvec);

void calculate_dvec2(const int32_t *aarray,
                     const int32_t *barray,
                     const int norb,
                     const int nalpha,
                     const int nbeta,
                     const int na,
                     const int nb,
                     const int nc1,
                     const int nc2,
                     const int nd1,
                     const int nd2,
                     const int nd3,
                     const int nd4,
                     const double complex *coeff,
                     double complex *dvec);

void calculate_coeff1(const int32_t *aarray,
                      const int32_t *barray,
                      const int norb,
                      const int i,
                      const int j,
                      const int nalpha,
                      const int nbeta,
                      const int na,
                      const int nb,
                      const int nd1,
                      const int nd2,
                      const int nd3,
                      const int nd4,
                      const int no1,
                      const int no2,
                      const double complex *dvec,
                      double complex *out);

void calculate_coeff2(const int32_t *aarray,
                      const int32_t *barray,
                      const int norb,
                      const int i,
                      const int j,
                      const int nalpha,
                      const int nbeta,
                      const int na,
                      const int nb,
                      const int nd1,
                      const int nd2,
                      const int nd3,
                      const int nd4,
                      const int no1,
                      const int no2,
                      const double complex *dvec,
                      double complex *out);

void calculate_dvec1_j(const int32_t *aarray,
                       const int32_t *barray,
                       const int norb,
                       const int j,
                       const int nalpha,
                       const int nbeta,
                       const int na,
                       const int nb,
                       const int nc1,
                       const int nc2,
                       const int nd1,
                       const int nd2,
                       const int nd3,
                       const double complex *coeff,
                       double complex *dvec);

void calculate_dvec2_j(const int32_t *aarray,
                       const int32_t *barray,
                       const int norb,
                       const int j,
                       const int nalpha,
                       const int nbeta,
                       const int na,
                       const int nb,
                       const int nc1,
                       const int nc2,
                       const int nd1,
                       const int nd2,
                       const int nd3,
                       const double complex *coeff,
                       double complex *dvec);

void make_nh123_real(const int norb,
                     const double *h4e,
                     double *nh1e,
                     double *nh2e,
                     double *nh3e);

void make_nh123(const int norb,
                const double complex *h4e,
                double complex *nh1e,
                double complex *nh2e,
                double complex *nh3e);

#endif // FQE_DATA_H_
