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

#include "fqe_data.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#ifdef _OPENMP
  #include <omp.h>
#endif
#include <complex.h>

#include "bitstring.h"
#include "macros.h"

#define MIN(a, b) (a) < (b) ? (a) : (b)
#define STATES_PER_SET 10

static void zdvec_make_part(const int (*map)[3],
                            const int map_els,
                            double complex *dvec,
                            const double complex *coeff,
                            const int inc_A,
                            const int inc_B,
                            const int maxj,
                            const struct blasfunctions * blasfunc) {
  for (const int (*cmap)[3] = map; cmap < &map[map_els]; ++cmap) {
    const double complex * coeff_p = coeff + inc_A * (*cmap)[0];
    double complex * dvec_p = dvec + inc_A * (*cmap)[1];
    const double complex parity =  (*cmap)[2];
    blasfunc->zaxpy(&maxj, &parity, coeff_p, &inc_B, dvec_p, &inc_B);
  }
}

/*
 * Low memory version for the _apply_array_spatial12_halffilling.
 */
void lm_apply_array12_same_spin(const double complex *coeff,
                               double complex *out,
                               const int (*dexc)[3],
                               const int alpha_states,
                               const int beta_states,
                               const int ndexc,
                               const double complex *h2e,
                               const int norbs,
                               const bool is_alpha,
                               const struct blasfunctions * blasfunc) {
  const int states1 = is_alpha ? alpha_states : beta_states;
  const int states2 = is_alpha ? beta_states : alpha_states;
  const int inc1 = is_alpha ? beta_states : 1;
  const int inc2 = is_alpha ? 1: beta_states;

#pragma omp parallel for schedule(static) shared(dexc, out, h2e, coeff)
  for (int s1 = 0; s1 < states1; ++s1) {
    const int (*cdexc)[3] = &dexc[s1 * ndexc];
    double complex * cout = &out[s1 * inc1];
    for (; cdexc < &dexc[(s1 + 1) * ndexc]; ++cdexc) {
      const int s2 = (*cdexc)[0];
      const int ijshift = (*cdexc)[1];
      const int parity1 = (*cdexc)[2];
      const int (*cdexc2)[3] = &dexc[s2 * ndexc];
      const int h2e_id = ijshift * norbs * norbs;
      for (; cdexc2 < &dexc[(s2 + 1) * ndexc]; ++cdexc2) {
        const int target = (*cdexc2)[0];
        const int klshift = (*cdexc2)[1];
        const int parity2 = (*cdexc2)[2];
        const double complex pref = parity1 * parity2
            * h2e[h2e_id + klshift];
        const double complex * ccoeff = &coeff[target * inc1];
        blasfunc->zaxpy(&states2, &pref, ccoeff, &inc2, cout, &inc2);
      }
    }
  }
}

/* Expects alpha excitation for ij and beta excitation for kl */
void lm_apply_array12_diff_spin(const double complex *coeff,
                               double complex *out,
                               const int (*adexc)[3],
                               const int (*bdexc)[3],
                               const int alpha_states,
                               const int beta_states,
                               const int nadexc,
                               const int nbdexc,
                               const double complex *h2e,
                               const int norbs) {
  const int nbdexc_tot = beta_states * nbdexc;
  int *targetbs = safe_malloc(targetbs, nbdexc_tot);
  double complex *prefactors = safe_malloc(
      prefactors, norbs * norbs * nbdexc_tot);
  double complex *blockc = safe_malloc(
      blockc, alpha_states * STATES_PER_SET * nbdexc);
  const int pref_inc = beta_states * nbdexc;
  const int norbs2 = norbs * norbs;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < beta_states * nbdexc; ++i) {
    targetbs[i] = bdexc[i][0];
    const int orbkl = bdexc[i][1];
    const int parity = bdexc[i][2];

    for (int orbid = 0; orbid < norbs2; ++orbid) {
      prefactors[orbid * pref_inc + i] = h2e[orbid * norbs2 + orbkl] * parity;
    }
  }

  int start_set = 0;
  while (start_set < beta_states) {
    const int end_set = MIN(beta_states, start_set + STATES_PER_SET);
    const int num_set = end_set - start_set;
    for (int i = 0; i < alpha_states; ++i) {
      int *cbeta = &targetbs[start_set * nbdexc];
      for (int j = 0; j < num_set * nbdexc; ++j) {
        blockc[i * num_set * nbdexc + j] = coeff[i * beta_states + (*cbeta++)];
      }
    }

#pragma omp parallel for schedule(static) shared(adexc, bdexc, out, h2e, coeff)
    for (int s1 = 0; s1 < alpha_states; ++s1) {
      const int (*cadexc)[3] = &adexc[s1 * nadexc];
      double complex *cout = &out[s1 * beta_states];
      for (; cadexc < &adexc[(s1 + 1) * nadexc]; ++cadexc) {
        const int targeta = (*cadexc)[0];
        const int orbij = (*cadexc)[1];
        const int parity1 = (*cadexc)[2];
        const double complex *ccoeff = &blockc[targeta * num_set * nbdexc];
        double complex *c_prefactor = &prefactors[orbij * pref_inc + start_set * nbdexc];

        for (int s2 = start_set; s2 < end_set; ++s2) {
          double complex coutel = 0;
          for (int bdid=0; bdid < nbdexc; ++bdid) {
            coutel += (*ccoeff++) * (*c_prefactor++);
          }
          cout[s2] += coutel * parity1;
        }
      }
    }
    start_set = end_set;
  }

  free(targetbs);
  free(prefactors);
  free(blockc);
}

void lm_apply_array1_old(const double complex *coeff,
                        double complex *out,
                        const int (*dexc)[3],
                        const int astates,
                        const int bstates,
                        const int ndexc,
                        const double complex *h1e,
                        const int norbs,
                        const bool is_alpha,
                        const struct blasfunctions * blasfunc) {
  const int inc1 = is_alpha ? bstates : 1;
  const int inc2 = is_alpha ? 1 : bstates;
  const int states1 = is_alpha ? astates : bstates;
  const int states2 = is_alpha ? bstates : astates;
  const int ONE = 1;

#pragma omp parallel for schedule(static) shared(dexc, out, h1e, coeff)
  for (int s1 = 0; s1 < states1; ++s1) {
    const int (*cdexc)[3] = &dexc[s1 * ndexc];
    double complex * cout = &out[s1 * states2];
    for (; cdexc < &dexc[(s1 + 1) * ndexc]; ++cdexc) {
      const int target = (*cdexc)[0];
      const int ijshift = (*cdexc)[1];
      const int parity = (*cdexc)[2];

      const double complex *ccoeff = &coeff[target * inc1];
      const double complex pref = parity * h1e[ijshift];
      blasfunc->zaxpy(&states2, &pref, ccoeff, &inc2, cout, &ONE);
    }
  }
}


void lm_apply_array1_sparse(const double complex *coeff,
                           double complex *out,
                           const int *dexc,
                           const int astates,
                           const int bstates,
                           const int ndexc,
                           const double complex *h1e,
                           const int norbs,
                           const int jorb,
                           const bool is_alpha,
                           const struct blasfunctions * blasfunc) {
  const int states1 = is_alpha ? astates : bstates;
  const int states2 = is_alpha ? bstates : astates;
  const int inc1 = is_alpha ? bstates : 1;
  const int inc2 = is_alpha ? 1 : bstates;
  const int ONE = 1;
  const int norbs2 = norbs * norbs;

  const int ndexc_tot = states1 * ndexc;
  int nest = 0;
  for (int s1 = 0; s1 < states1; ++s1)
  for (int i = 0; i < ndexc; ++i) {
    const int orbkl = dexc[3*(s1*ndexc + i) + 1];
    if (orbkl == 0) ++nest;
  }

  int *signs = safe_malloc(signs, ndexc_tot);
  int *coff = safe_malloc(coff, ndexc_tot);
  int *boff = safe_malloc(boff, ndexc_tot);
  double complex *ctemp = safe_malloc(ctemp, nest*states2);

  for (int orbid = 0; orbid < norbs2; ++orbid) {
    const int col = orbid % norbs;
    if (jorb > -1 && jorb != col)
      continue;
    int nsig = 0;
    for (int s1 = 0; s1 < states1; ++s1)
    for (int i = 0; i < ndexc; ++i) {
      const int orbkl = dexc[3*(s1*ndexc + i) + 1];
      if (orbkl == orbid) {
        signs[nsig] = dexc[3*(s1*ndexc + i) + 2];
        coff[nsig] = dexc[3*(s1*ndexc + i)];
        boff[nsig] = s1*inc1;
        ++nsig;
      }
    }
#pragma omp parallel
    {
#pragma omp for schedule(static)
    for (int ii = 0; ii < nsig*states2; ++ii) ctemp[ii] = 0.0;

#pragma omp for schedule(static)
    for (int isig = 0; isig < nsig; ++isig) {
      const int offset = coff[isig]*inc1;
      const double complex *cptr = coeff + offset;
      double complex *tptr = ctemp + isig*states2;
      double complex zsign = signs[isig] * h1e[orbid];
      blasfunc->zaxpy(&states2, &zsign, cptr, &inc2, tptr, &ONE);
    }
#pragma omp for schedule(static)
    for (int s2 = 0; s2 < states2; ++s2) {
      double complex * optr = out + s2*inc2;
      double complex * cptr = ctemp + s2;
      int * btemp = boff;
      int cidx = 0;
      for (int isig = 0; isig < nsig; ++isig, ++btemp, cidx += states2) {
        optr[*btemp] += cptr[cidx];
      }
    }
    }
  }
  free(ctemp);
  free(signs);
  free(coff);
  free(boff);
}

void lm_apply_array1(const double complex *coeff,
                    double complex *out,
                    const int *dexc,
                    const int astates,
                    const int bstates,
                    const int ndexc,
                    const double complex *h1e,
                    const int norbs,
                    const bool is_alpha,
                    const struct blasfunctions * blasfunc) {
  const int states1 = is_alpha ? astates : bstates;
  const int states2 = is_alpha ? bstates : astates;
  const int inc1 = is_alpha ? bstates : 1;
  const int inc2 = is_alpha ? 1 : bstates;

#pragma omp parallel for schedule(static)
  for (int s1 = 0; s1 < states1; ++s1) {
    const int *cdexc = dexc + 3*s1*ndexc;
    const int *lim1 = cdexc + 3*ndexc;
    double complex * cout = &out[s1 * inc1];
    for (; cdexc < lim1; cdexc = cdexc + 3) {
      const int target = cdexc[0];
      const int ijshift = cdexc[1];
      const int parity = cdexc[2];

      const double complex pref = parity * h1e[ijshift];
      const double complex *xptr = coeff + target*inc1;
      blasfunc->zaxpy(&states2, &pref, xptr, &inc2, cout, &inc2);
    }
  }
}


void lm_apply_array1_column_alpha(double complex *coeff,
                                  const int *index,
                                  const int *exc,
                                  const int *exc2,
                                  const int lena,
                                  const int lenb,
                                  const int nexc0,
                                  const int nexc1,
                                  const int nexc2,
                                  const double complex *h1e,
                                  const int icol,
                                  const struct blasfunctions * blasfunc) {
  const int nbin = (lenb - 1) / ZAXPY_STRIDE + 1;
  for (int nbatch = 0; nbatch < nbin; ++nbatch) {
#pragma omp parallel for schedule(static)
    for (int a = 0; a < nexc0; ++a) {
      const int clenb = MIN(ZAXPY_STRIDE, lenb - nbatch * ZAXPY_STRIDE);
      double complex *ccoeff = coeff + nbatch * ZAXPY_STRIDE;
      double complex *output = ccoeff + index[a] * lenb;
      for (int i = 0; i < nexc1; ++i) {
        const int *cexc = exc + 3 * (i + nexc1 * a);
        const int source = cexc[0];
        const int ishift = cexc[1];
        const int parity = cexc[2];

        const double complex *input = ccoeff + source * lenb;

        const double complex pref = parity * h1e[ishift];
        const int ONE = 1;
        blasfunc->zaxpy(&clenb, &pref, input, &ONE, output, &ONE);
      }
    }
    const double complex prefac = 1.0 + h1e[icol];
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nexc2; ++i) {
      const int clenb = MIN(ZAXPY_STRIDE, lenb - nbatch * ZAXPY_STRIDE);  
      double complex *ccoeff = coeff + nbatch * ZAXPY_STRIDE;
      const int target = exc2[i];
      const int ONE = 1;
      blasfunc->zscal(&clenb, &prefac, ccoeff + target * lenb, &ONE);
    }
  }
}


int zdvec_make(const int (**map)[3],
               const int *map_els,
               const int nr_maps,
               const double complex *coeff,
               double complex *dvec,
               const int alpha_states,
               const int beta_states,
               const bool is_alpha,
               const struct blasfunctions * blasfunc) {
  const int inc_A = is_alpha ? beta_states : 1;
  const int inc_B = is_alpha ? 1: beta_states;
  const int maxj = is_alpha ? beta_states : alpha_states;
  const int dvec_inc = alpha_states * beta_states;

#pragma omp parallel for schedule(dynamic)
  for (int mapno = 0; mapno < nr_maps; ++mapno) {
    double complex *cdvec = dvec + dvec_inc * mapno;
    zdvec_make_part(map[mapno],
                    map_els[mapno],
                    cdvec,
                    coeff,
                    inc_A,
                    inc_B,
                    maxj,
                    blasfunc);
  }
  return 0;
}

int zcoeff_make(const int (**map)[3],
                const int *map_els,
                const int nr_maps,
                double complex *coeff,
                const double complex *dvec,
                const int alpha_states,
                const int beta_states,
                const bool is_alpha,
                const struct blasfunctions * blasfunc) {
  const int inc_A = is_alpha ? beta_states : 1;
  const int inc_B = is_alpha ? 1: beta_states;
  const int maxj = is_alpha ? beta_states : alpha_states;
  const int dvec_inc = alpha_states * beta_states;

  for (int mapno = 0; mapno < nr_maps; ++mapno) {
    const int (*cmap)[3] = map[mapno];
    const double complex *cdvec = dvec + dvec_inc * mapno;

    for (; cmap < &map[mapno][map_els[mapno]]; ++cmap) {
      double complex *coeff_p = coeff + inc_A * (*cmap)[0];
      const double complex *dvec_p = cdvec + inc_A * (*cmap)[1];
      const double complex parity = (*cmap)[2];
      blasfunc->zaxpy(&maxj, &parity, dvec_p, &inc_B, coeff_p, &inc_B);
    }
  }

  return 0;
}

static int zdiagonal_coulomb_apply_part(const int *occ,
                                        const double complex *diag,
                                        const double complex *array,
                                        double complex *output,
                                        const int states,
                                        const int nel,
                                        const int norbs) {
  for (int i = 0; i < states; ++i) {
    double complex p_adiag = 0.0;
    const int *c_occ = &occ[i * nel];

    for (int el = 0; el < nel; ++el) {
      p_adiag += diag[c_occ[el]];
      const double complex *c_array = array + norbs * c_occ[el];
      for (int el2 = 0; el2 < nel; ++el2) {
        p_adiag += c_array[c_occ[el2]];
      }
    }
    output[i] = p_adiag;
  }
  return 0;
}

static int zdiagonal_coulomb_part(const int *occ,
                                  const double complex *diag,
                                  const double complex *array,
                                  double complex *output,
                                  const int states,
                                  const int nel,
                                  const int norbs) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < states; ++i) {
    double complex p_adiag = 1.0;
    const int *c_occ = &occ[i * nel];

    for (int el = 0; el < nel; ++el) {
      p_adiag *= diag[c_occ[el]];
      const double complex *c_array = array + norbs * c_occ[el];
      for (int el2 = 0; el2 < nel; ++el2) {
        p_adiag *= c_array[c_occ[el2]];
      }
    }
    output[i] = p_adiag;
  }
  return 0;
}

void zdiagonal_coulomb_apply(const uint64_t *alpha_strings,
                            const uint64_t *beta_strings,
                            const double complex *diag,
                            const double complex *array,
                            double complex *output,
                            const int alpha_states,
                            const int beta_states,
                            const int nalpha,
                            const int nbeta,
                            const int norbs) {

  double complex *adiag = safe_malloc(adiag, alpha_states);
  double complex *bdiag = safe_malloc(bdiag, beta_states);

  int *aocc = safe_malloc(aocc, alpha_states * nalpha);
  int *bocc = safe_malloc(bocc, beta_states * nbeta);

#pragma omp parallel for schedule(static)
  for (int as = 0; as < alpha_states; ++as) {
    get_occupation(&aocc[as * nalpha], alpha_strings[as]);
  }
#pragma omp parallel for schedule(static)
  for (int bs = 0; bs < beta_states; ++bs) {
    get_occupation(&bocc[bs * nbeta], beta_strings[bs]);
  }

  zdiagonal_coulomb_apply_part(
        aocc, diag, array, adiag, alpha_states, nalpha, norbs);
  zdiagonal_coulomb_apply_part(
        bocc, diag, array, bdiag, beta_states, nbeta, norbs);

#pragma omp parallel shared(aocc, bocc, output, array, adiag, bdiag)
  {
  double complex aarrays[MAX_ORBS];
  assert(norbs < (int)MAX_ORBS);

#pragma omp for schedule(static)
  for (int as = 0; as < alpha_states; ++as) {
    double complex *c_output = output + as * beta_states;
    const int *caocc = &aocc[as * nalpha];

    for (int i = 0; i < norbs; ++i) {
      aarrays[i] = 0.0;
    }

    for (int ela = 0; ela < nalpha; ++ela) {
      const double complex *carray1 = array + caocc[ela] * norbs;
      const double complex *carray2 = array + caocc[ela];
      for (int i = 0; i < norbs; ++i) {
        aarrays[i] += carray1[i];
        aarrays[i] += carray2[i*norbs];
      }
    }

    for (int bs = 0; bs < beta_states; ++bs) {
      double complex diag_ele = 0.0;
      const int *cbocc = &bocc[bs * nbeta];
      for (int elb = 0; elb < nbeta; ++elb) {
        diag_ele += aarrays[cbocc[elb]];
      }
      c_output[bs] *= (diag_ele +  bdiag[bs] +  adiag[as]);
    }
  }
  }

  free(adiag);
  free(bdiag);
  free(aocc);
  free(bocc);
}

int zdiagonal_coulomb(const uint64_t *alpha_strings,
                      const uint64_t *beta_strings,
                      const double complex *diag,
                      const double complex *array,
                      double complex *output,
                      const int alpha_states,
                      const int beta_states,
                      const int nalpha,
                      const int nbeta,
                      const int norbs) {
  double complex *adiag = safe_malloc(adiag, alpha_states);
  double complex *bdiag = safe_malloc(bdiag, beta_states);

  int *aocc = safe_malloc(aocc, alpha_states * nalpha);
  int *bocc = safe_malloc(bocc, beta_states * nbeta);

#pragma omp parallel for schedule(static)
  for (int as = 0; as < alpha_states; ++as) {
    get_occupation(&aocc[as * nalpha], alpha_strings[as]);
  }
#pragma omp parallel for schedule(static)
  for (int bs = 0; bs < beta_states; ++bs) {
    get_occupation(&bocc[bs * nbeta], beta_strings[bs]);
  }

  double complex *diagexp = safe_malloc(diagexp, norbs);
  double complex *arrayexp = safe_malloc(arrayexp, norbs * norbs);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < norbs; ++i)
    diagexp[i] = cexp(diag[i]);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < norbs * norbs; ++i)
    arrayexp[i] = cexp(array[i]);

  zdiagonal_coulomb_part(aocc, diagexp, arrayexp, adiag, alpha_states, nalpha, norbs);
  zdiagonal_coulomb_part(bocc, diagexp, arrayexp, bdiag, beta_states, nbeta, norbs);

#pragma omp parallel shared(aocc, bocc, output, arrayexp, adiag, bdiag)
  {
    double complex aarrays[MAX_ORBS];
    assert(norbs < (int)MAX_ORBS);

#pragma omp for schedule(static)
    for (int as = 0; as < alpha_states; ++as) {
      double complex *c_output = output + as * beta_states;
      const int *caocc = &aocc[as * nalpha];

      for (int i = 0; i < norbs; ++i) {
        aarrays[i] = 1.0;
      }

      for (int ela = 0; ela < nalpha; ++ela) {
        const double complex *carray = arrayexp + caocc[ela] * norbs;
        for (int i = 0; i < norbs; ++i) {
          aarrays[i] *= carray[i];
        }
      }

      for (int bs = 0; bs < beta_states; ++bs) {
        double complex diag_ele = 1.0;
        const int *cbocc = &bocc[bs * nbeta];
        for (int elb = 0; elb < nbeta; ++elb) {
          diag_ele *= aarrays[cbocc[elb]];
        }
        c_output[bs] *= diag_ele * diag_ele * bdiag[bs] * adiag[as];
      }
    }
  }

  free(adiag);
  free(bdiag);
  free(aocc);
  free(bocc);
  free(diagexp);
  free(arrayexp);
  return 0;
}

void make_dvec_part(const int astates_full,
                   const int bstates_full,
                   const int astates_part_begin,
                   const int bstates_part_begin,
                   const int astates_part_num,
                   const int bstates_part_num,
                   const int32_t (*maps)[4],
                   const int nmaps,
                   const double complex *input,
                   double complex *output,
                   const bool isalph,
                   const struct blasfunctions * blasfunc) {
  const int ONE = 1;
  const int totstates = astates_part_num * bstates_part_num;
  const int INC = isalph ? bstates_part_num : ONE;
  const int SHIFT = isalph ? ONE : bstates_part_num;
  const int INC2 = isalph ? bstates_full : astates_full;

  const int states1_b = isalph ? astates_part_begin : bstates_part_begin;
  const int states2_b = isalph ? bstates_part_begin : astates_part_begin;
  const int states_num = isalph ? bstates_part_num : astates_part_num;

#pragma omp parallel for schedule(static)
  for (const int (*map)[4] = maps; map < &maps[nmaps]; ++map) {
    const int ijshift = (*map)[0] * totstates;
    const int target_state1 = (*map)[1] - states1_b;
    const int orig_state1 = (*map)[2];
    const double complex parity = (*map)[3];
    blasfunc->zaxpy(&states_num,
                    &parity,
                    &input[orig_state1 * INC2 + states2_b],
                    &ONE,
                    &output[ijshift +  target_state1 * INC],
                    &SHIFT);
  }
}

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
                   const struct blasfunctions * blasfunc) {
  const int ONE = 1;
  const int totstates = states1_part_num * states2_part_num;
  for (const int (*map)[4] = maps; map < &maps[nmaps]; ++map) {
    const int ijshift = (*map)[0] * totstates;
    const int target_state1 = (*map)[1];
    const int orig_state1 = (*map)[2] - states1_part_begin;
    const double complex parity = (*map)[3];
    blasfunc->zaxpy(&states2_part_num,
                    &parity,
                    &input[ijshift +  orig_state1 * states2_part_num],
                    &ONE,
                    &output[target_state1 * states2_full + states2_part_begin],
                    &ONE);
  }
}

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
                                   const struct blasfunctions * blasfunc) {
  const int states1 = is_alpha ? alpha_states : beta_states;
  const int states2 = is_alpha ? beta_states : alpha_states;
  const int inc1 = is_alpha ? beta_states : 1;
  const int inc2 = is_alpha ? 1: beta_states;


#pragma omp parallel
  {
  double complex *temp = safe_malloc(temp, states1);
#pragma omp for schedule(static, 1)
  for (int s1 = 0; s1 < states1; ++s1) {
    for (int ii = 0; ii < states1; ii++) temp[ii] = 0.0;
    const int *cdexc = dexc + 3*s1*ndexc;
    const int *lim1 = cdexc + 3*ndexc;
    double complex *cout = out + s1*inc1;
    for (; cdexc < lim1; cdexc = cdexc + 3) {
      const int s2 = cdexc[0];
      const int ijshift = cdexc[1];
      const int parity1 = cdexc[2];
      const int *cdexc2 = dexc + 3*s2*ndexc;
      const int *lim2 = cdexc2 + 3*ndexc;
      const int h2e_id = ijshift * norbs * norbs;
      const double complex *h2etmp = h2e + h2e_id;
      temp[s2] += parity1 * h1e[ijshift];
      for (; cdexc2 < lim2; cdexc2 = cdexc2 + 3) {
        const int target = cdexc2[0];
        const int klshift = cdexc2[1];
        const int parity = cdexc2[2] * parity1;
        const double complex pref = parity * (h2etmp + klshift)[0];
        temp[target] += pref;
      }
    }
    const double complex *xptr = coeff;
    for (int ii = 0; ii < states1; ii++) {
      const double complex ttt = temp[ii];
      blasfunc->zaxpy(&states2, &ttt, xptr, &inc2, cout, &inc2);
      xptr += inc1;
    }
  }
  free(temp);
  }
}

#ifdef _UNUSED_
/* Expects alpha excitation for ij and beta excitation for kl */
static void lm_apply_array12_diff_spin_opt2(const double complex *coeff,
                                            double complex *out,
                                            const int *adexc,
                                            const int *bdexc,
                                            const int alpha_states,
                                            const int beta_states,
                                            const int nadexc,
                                            const int nbdexc,
                                            const double complex *h2e,
                                            const int norbs,
                                            const struct blasfunctions * blasfunc) {
  const int nbdexc_tot = beta_states * nbdexc;
  const int norbs2 = norbs * norbs;
  const int one = 1;
  int *signs = safe_malloc(signs, nbdexc_tot);
  int *coff = safe_malloc(coff, nbdexc_tot);
  int *boff = safe_malloc(boff, nbdexc_tot);

  int nest = 0;
  for (int s2 = 0; s2 < beta_states; ++s2)
  for (int i = 0; i < nbdexc; ++i) {
    const int orbkl = bdexc[3*(s2*nbdexc + i) + 1];
    if (orbkl == 0) ++nest;
  }
  double complex *vtemp = safe_malloc(vtemp, nest);
  double complex *ctemp = safe_malloc(ctemp, nest * alpha_states);

  for (int orbid = 0; orbid < norbs2; ++orbid) {
    int nsig = 0;
    for (int s2 = 0; s2 < beta_states; ++s2)
    for (int i = 0; i < nbdexc; ++i) {
      const int orbkl = bdexc[3*(s2*nbdexc + i) + 1];
      if (orbkl == orbid) {
        signs[nsig] = bdexc[3*(s2*nbdexc + i) + 2];
        coff[nsig] = bdexc[3*(s2*nbdexc + i)];
        boff[nsig] = s2;
        ++nsig;
      }
    }

    for (int ii = 0; ii < nsig*alpha_states; ++ii) ctemp[ii] = 0.0;

    for (int isig = 0; isig < nsig; ++isig) {
      const int offset = coff[isig];
      const double complex *cptr = coeff + offset;
      double complex *tptr = ctemp + isig;
      double complex zsign = signs[isig];
      blasfunc->zaxpy(&alpha_states, &zsign, cptr, &beta_states, tptr, &nsig);
    }

    const complex double *tmperi = h2e + orbid*norbs2;
    for (int s1 = 0; s1 < alpha_states; ++s1) {
      for (int kk = 0; kk < nsig; ++kk) vtemp[kk] = 0.0;
      for (int j = 0; j < nadexc; ++j) {
        int idx2 = adexc[3*(s1*nadexc + j)];
        const int parity = adexc[3*(s1*nadexc + j) + 2];
        const int orbij = adexc[3*(s1*nadexc + j) + 1];
        const double complex ttt = parity * tmperi[orbij];
        const double complex *cctmp = ctemp + idx2 * nsig;
        blasfunc->zaxpy(&nsig, &ttt, cctmp, &one, vtemp, &one);
      }

      double complex *tmpout = out + s1*beta_states;
      for (int isig = 0; isig < nsig; ++isig) {
        tmpout[boff[isig]] += vtemp[isig];
      }
    }
  }

  free(vtemp);
  free(signs);
  free(ctemp);
  free(coff);
  free(boff);
}

/* Expects alpha excitation for ij and beta excitation for kl */
static void lm_apply_array12_diff_spin_opt1(const double complex *coeff,
                                            double complex *out,
                                            const int *adexc,
                                            const int *bdexc,
                                            const int alpha_states,
                                            const int beta_states,
                                            const int nadexc,
                                            const int nbdexc,
                                            const double complex *h2e,
                                            const int norbs,
                                            const struct blasfunctions * blasfunc) {
  const int nadexc_tot = alpha_states * nadexc;
  const int norbs2 = norbs * norbs;
  const int one = 1;
  int *signs = safe_malloc(signs, nadexc_tot);
  int *coff = safe_malloc(coff, nadexc_tot);
  int *boff = safe_malloc(boff, nadexc_tot);

  int nest = 0;
  for (int s1 = 0; s1 < alpha_states; ++s1)
  for (int i = 0; i < nadexc; ++i) {
    const int orbij = adexc[3*(s1*nadexc + i) + 1];
    if (orbij == 0) ++nest;
  }
  double complex *vtemp = safe_malloc(vtemp, nest);
  double complex *ctemp = safe_malloc(ctemp, nest * beta_states);

  for (int orbid = 0; orbid < norbs2; ++orbid) {
    int nsig = 0;
    for (int s1 = 0; s1 < alpha_states; ++s1)
    for (int i = 0; i < nadexc; ++i) {
      const int orbij = adexc[3*(s1*nadexc + i) + 1];
      if (orbij == orbid) {
        signs[nsig] = adexc[3*(s1*nbdexc + i) + 2];
        coff[nsig] = adexc[3*(s1*nbdexc + i)];
        boff[nsig] = s1;
        ++nsig;
      }
    }

    for (int ii = 0; ii < nsig*beta_states; ++ii) ctemp[ii] = 0.0;

    for (int isig = 0; isig < nsig; ++isig) {
      const int offset = coff[isig];
      const double complex *cptr = coeff + offset*beta_states;
      double complex *tptr = ctemp + isig;
      double complex zsign = signs[isig];
      blasfunc->zaxpy(&beta_states, &zsign, cptr, &one, tptr, &nsig);
    }

    const complex double *tmperi = h2e + orbid*norbs2;
    for (int s2 = 0; s2 < beta_states; ++s2) {
      for (int kk = 0; kk < nsig; ++kk) vtemp[kk] = 0.0;
      for (int j = 0; j < nbdexc; ++j) {
        int idx2 = bdexc[3*(s2*nbdexc + j)];
        const int parity = bdexc[3*(s2*nbdexc + j) + 2];
        const int orbkl = bdexc[3*(s2*nbdexc + j) + 1];
        const double complex ttt = parity * tmperi[orbkl];
        const double complex * cctmp = ctemp + idx2 * nsig;
        blasfunc->zaxpy(&nsig, &ttt, cctmp, &one, vtemp, &one);
      }

      double complex *tmpout = out + s2;
      for (int isig = 0; isig < nsig; ++isig) {
        tmpout[beta_states*boff[isig]] += vtemp[isig];
      }
    }
  }

  free(vtemp);
  free(signs);
  free(ctemp);
  free(coff);
  free(boff);
}
#endif

/* Expects alpha excitation for ij and beta excitation for kl */
void lm_apply_array12_diff_spin_omp1(const double complex *coeff,
                                    double complex *out,
                                    const int *adexc,
                                    const int *bdexc,
                                    const int alpha_states,
                                    const int beta_states,
                                    const int nadexc,
                                    const int nbdexc,
                                    const double complex *h2e,
                                    const int norbs,
                                    const struct blasfunctions * blasfunc) {
  const int nadexc_tot = alpha_states * nadexc;
  const int norbs2 = norbs * norbs;
  const int one = 1;
#ifdef _OPENMP
  const int nthrds = omp_get_max_threads();
#else
  const int nthrds = 1;
#endif
  int *signs = safe_malloc(signs, nadexc_tot);
  int *coff = safe_malloc(coff, nadexc_tot);
  int *boff = safe_malloc(boff, nadexc_tot);

  int nest = 0;
  for (int s1 = 0; s1 < alpha_states; ++s1)
  for (int i = 0; i < nadexc; ++i) {
    const int orbij = adexc[3*(s1*nadexc + i) + 1];
    if (orbij == 0) ++nest;
  }
  double complex *vtemp = safe_malloc(vtemp, nthrds * nest);
  double complex *ctemp = safe_malloc(ctemp, nest * beta_states);

  for (int orbid = 0; orbid < norbs2; ++orbid) {
    int nsig = 0;
    for (int s1 = 0; s1 < alpha_states; ++s1)
    for (int i = 0; i < nadexc; ++i) {
      const int orbij = adexc[3*(s1*nadexc + i) + 1];
      if (orbij == orbid) {
        signs[nsig] = adexc[3*(s1*nadexc + i) + 2];
        coff[nsig] = adexc[3*(s1*nadexc + i)];
        boff[nsig] = s1;
        ++nsig;
      }
    }

#pragma omp parallel
    {
#pragma omp for schedule(static)
    for (int ii = 0; ii < nsig*beta_states; ++ii) ctemp[ii] = 0.0;

#pragma omp for schedule(static)
    for (int isig = 0; isig < nsig; ++isig) {
      const int offset = coff[isig];
      const double complex *cptr = coeff + offset*beta_states;
      double complex *tptr = ctemp + isig;
      double complex zsign = signs[isig];
      blasfunc->zaxpy(&beta_states, &zsign, cptr, &one, tptr, &nsig);
    }

    const complex double *tmperi = h2e + orbid*norbs2;
#pragma omp for schedule(static)
    for (int s2 = 0; s2 < beta_states; ++s2) {
#ifdef _OPENMP
      const int ithrd = omp_get_thread_num();
#else
      const int ithrd = 0;
#endif
      double complex *vpt = vtemp + ithrd*nsig;
      for (int kk = 0; kk < nsig; ++kk) vpt[kk] = 0.0;
      for (int j = 0; j < nbdexc; ++j) {
        int idx2 = bdexc[3*(s2*nbdexc + j)];
        const int parity = bdexc[3*(s2*nbdexc + j) + 2];
        const int orbkl = bdexc[3*(s2*nbdexc + j) + 1];
        const double complex ttt = parity * tmperi[orbkl];
        const double complex * cctmp = ctemp + idx2 * nsig;
        blasfunc->zaxpy(&nsig, &ttt, cctmp, &one, vpt, &one);
      }

      double complex *tmpout = out + s2;
      for (int isig = 0; isig < nsig; ++isig) {
        tmpout[beta_states*boff[isig]] += vpt[isig];
      }
    }
    }
  }

  free(vtemp);
  free(signs);
  free(ctemp);
  free(coff);
  free(boff);
}

/* Expects alpha excitation for ij and beta excitation for kl */
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
                                   const struct blasfunctions * blasfunc) {
  lm_apply_array12_diff_spin_omp1(
    coeff,
    out,
    adexc,
    bdexc,
    alpha_states,
    beta_states,
    nadexc,
    nbdexc,
    h2e,
    norbs,
    blasfunc);
}

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
                               double complex *intermediate) {
  const int tpar = nalpha % 2 == 0 ? -1 : 1;
#pragma omp parallel for schedule(dynamic) collapse(2)
  for (int i = 0; i < norb; ++i)
  for (int j = 0; j < norb; ++j) {
    complex double *iptr = intermediate + i*nia*nib*norb + j*nia*nib;
    const int *aptr = alpha_array + i*na1*3;
    const int *bptr = beta_array + j*nb1*3;
    for (int k = 0; k < na1; ++k) {
      const int sourcea = aptr[k*3];
      const int targeta = aptr[k*3 + 1];
      const int paritya = aptr[k*3 + 2];
      int sign = tpar*paritya;
      for (int l = 0; l < nb1; ++l) {
        const int sourceb = bptr[l*3];
        const int targetb = bptr[l*3 + 1];
        const int parityb = bptr[l*3 + 2];
        complex double work = coeff[sourcea*ncb + sourceb] * sign * parityb;
        iptr[targeta*nib + targetb] += 2 * work;
      }
    }
  }
}

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
                                double complex *out) {
  const int tpar = nalpha % 2 == 0 ? 1 : -1;
  for (int i = 0; i < norb; ++i) {
    for (int j = 0; j < norb; ++j) {
      const double complex *iptr = intermediate + i*norb*nia*nib + j*nia*nib;
      const int *aptr = alpha_array + i*na1*3;
      const int *bptr = beta_array + j*nb1*3;
      for (int k = 0; k < na1; ++k) {
        const int sourcea = aptr[k*3];
        const int targeta = aptr[k*3 + 1];
        const int paritya = aptr[k*3 + 2];
        int sign = tpar*paritya;
        for (int l = 0; l < nb1; ++l) {
          const int sourceb = bptr[l*3];
          const int targetb = bptr[l*3 + 1];
          const int parityb = bptr[l*3 + 2];
          double complex work = iptr[targeta*nib + targetb] * sign;
          out[sourcea*nob + sourceb] += work * parityb;
        }
      }
    }
  }
}

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
                                double complex *intermediate) {
#pragma omp parallel for schedule(static)
  for (int ijn = 0; ijn < nlt; ++ijn) {
    const int *aptr = alpha_array + ijn*na*3;
    for (int k = 0; k < na; ++k) {
      int source = aptr[k*3];
      int target = aptr[k*3 + 1];
      int parity = aptr[k*3 + 2];
      double complex * iptr = alpha ?
          intermediate + ijn*ni2*ni3 + target*ni3:
          intermediate + ijn*ni2*ni3 + target;
      const double complex * cptr = alpha ?
          coeff + source*nc2 :
          coeff + source;
      if (alpha) {
        for (int i = 0; i < ni3; ++i) {
          iptr[i] += cptr[i]*parity;
        }
      } else {
        for (int i = 0; i < ni2; ++i) {
          iptr[i*ni3] += cptr[i*nc2]*parity;
        }
      }
    }
  }
}

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
                               double complex *out) {
  for (int ijn = 0; ijn < nlt; ++ijn) {
    const double complex *iptr = intermediate + ijn*ni2*ni3;
    const int *aptr = alpha_array + ijn*na*3;
    for (int k = 0; k < na; ++k) {
      const int source = aptr[k*3];
      const int target = aptr[k*3 + 1];
      const int parity = aptr[k*3 + 2];
      if (alpha) {
        for (int i = 0; i < no2; ++i) {
          out[source*no2 + i] -= iptr[target*ni3 + i] * parity;
        }
      } else {
        for (int i = 0; i < no1; ++i) {
          out[i*no2 + source] -= iptr[i*ni3 + target] * parity;
        }
      }
    }
  }
}

void make_Hcomp(const int norb,
                const int nlt,
                const double complex *h2e,
                double complex *h2ecomp) {
  for (int i = 0; i < norb; ++i)
  for (int j = i + 1; j < norb; ++j) {
    int ijn = i + j*(j + 1)/2;
    for (int k = 0; k < norb; ++k)
    for (int l = k + 1; l < norb; ++l) {
      int kln = k + l*(l + 1)/2;
      h2ecomp[ijn*nlt + kln] =
          h2e[i*norb*norb*norb + j*norb*norb + k*norb + l]
        - h2e[i*norb*norb*norb + j*norb*norb + l*norb + k]
        - h2e[j*norb*norb*norb + i*norb*norb + k*norb + l]
        + h2e[j*norb*norb*norb + i*norb*norb + l*norb + k];
    }
  }
}

void apply_individual_nbody1_accumulate(const double complex coeff,
                                        double complex *ocoeff,
                                        const double complex *icoeff,
                                        const int n,
                                        const int nao,
                                        const int nbo,
                                        const int nai,
                                        const int nbi,
                                        const int nt,
                                        const int64_t *amap,
                                        const int64_t *btarget,
                                        const int64_t *bsource,
                                        const int64_t *bparity) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    const int64_t sourcea = amap[i*3 + 0];
    const int64_t targeta = amap[i*3 + 1];
    const int64_t paritya = amap[i*3 + 2];
    double complex *optr = ocoeff + targeta*nbo;
    const double complex *iptr = icoeff + sourcea*nbi;
    const double complex pref = coeff*paritya;
    for (int j = 0; j < nt; ++j) {
      optr[btarget[j]] += pref * iptr[bsource[j]] * bparity[j];
    }
  }
}

/* C-kernel for filling a FqeData from a cirq state */
static void _from_to_cirq(double complex * const cirq_wfn,
                          double complex * const fqe_wfn,
                          const int alpha_states,
                          const int beta_states,
                          const long long * cirq_alpha_id,
                          const long long * cirq_beta_id,
                          const int * const alpha_swaps,
                          const int * const beta_occs,
                          const int nbeta,
                          const int norbs,
                          void (*fill_func)(double complex * cirq_el,
                                            double complex * fqe_el,
                                            double parity) ) {
  const int nqubits = norbs * 2;
#pragma omp parallel for schedule(static)
  for (int beta_id = 0; beta_id < beta_states; ++beta_id) {
    const int * beta_occ = &beta_occs[beta_id * nbeta];
    const long long this_cirq_beta_id = cirq_beta_id[beta_id];

    for (int alpha_id = 0; alpha_id < alpha_states; ++alpha_id) {
      int swaps = 0;
      const int * this_alpha_swaps = &alpha_swaps[alpha_id * nqubits];
      for (int beta_el = 0; beta_el < nbeta; ++beta_el) {
        swaps += this_alpha_swaps[beta_occ[beta_el]];
      }

      const double parity = swaps % 2 == 0 ? 1.0 : -1.0;
      const long long cirq_id = this_cirq_beta_id ^ cirq_alpha_id[alpha_id];

      (*fill_func)(&cirq_wfn[cirq_id],
                   &fqe_wfn[alpha_id * beta_states + beta_id],
                   parity);
    }
  }
}

static void fill_to_cirq(double complex * cirq_el,
                         double complex * fqe_el,
                         double parity) {
  *cirq_el = parity * (*fqe_el);
}

static void fill_from_cirq(double complex * cirq_el,
                           double complex * fqe_el,
                           double parity) {
  *fqe_el = parity * (*cirq_el);
}

/* C-kernel for filling a cirq state from FqeData */
void to_cirq(double complex * const cirq_wfn,
             double complex * const fqe_wfn,
             const int alpha_states,
             const int beta_states,
             const long long * cirq_alpha_id,
             const long long * cirq_beta_id,
             const int * const alpha_swaps,
             const int * const beta_occs,
             const int nbeta,
             const int norbs) {
  _from_to_cirq(cirq_wfn, fqe_wfn, alpha_states, beta_states, cirq_alpha_id,
                cirq_beta_id, alpha_swaps, beta_occs, nbeta, norbs,
                &fill_to_cirq);
}

/* C-kernel for filling a FqeData from a cirq state */
void from_cirq(double complex * const cirq_wfn,
               double complex * const fqe_wfn,
               const int alpha_states,
               const int beta_states,
               const long long * cirq_alpha_id,
               const long long * cirq_beta_id,
               const int * const alpha_swaps,
               const int * const beta_occs,
               const int nbeta,
               const int norbs) {
  _from_to_cirq(cirq_wfn, fqe_wfn, alpha_states, beta_states, cirq_alpha_id,
                cirq_beta_id, alpha_swaps, beta_occs, nbeta, norbs,
                &fill_from_cirq);
}

void sparse_scale(const long long *xi,
                  const long long *yi,
                  const double complex fac,
                  const int ni,
                  const int nd1,
                  const int nd2,
                  double complex *data) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < ni; ++i) {
      const long long ixx = xi[i];
      const long long iyy = yi[i];
      data[ixx*nd2 + iyy] *= fac;
  }
}

void integer_index_accumulate_real(double *result,
                                   const double *data,
                                   uint64_t occupation) {
  int id = -1;
  *result = 0.0;
  while (gbit_index(&occupation, &id)) { *result += data[id]; }
}

void integer_index_accumulate(double complex *result,
                              const double complex *data,
                              uint64_t occupation) {
  int id = -1;
  *result = 0.0;
  while (gbit_index(&occupation, &id)) { *result += data[id]; }
}


void apply_diagonal_inplace(double complex *data,
                            const double complex *aarray,
                            const double complex *barray,
                            const uint64_t *astrings,
                            const uint64_t *bstrings,
                            const int lena,
                            const int lenb) {
  double complex *alpha = safe_malloc(alpha, lena);
  double complex *beta = safe_malloc(beta, lenb);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < lena; ++i) {
    integer_index_accumulate(alpha + i, aarray, astrings[i]);
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < lenb; ++i) {
    integer_index_accumulate(beta + i, barray, bstrings[i]);
  }
#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < lena; ++i) {
    for (int j = 0; j < lenb; ++j) {
      data[j + lenb * i] *= alpha[i] + beta[j];
    }
  }
  free(alpha);
  free(beta);
}


void apply_diagonal_inplace_real(double complex *data,
                                 const double *aarray,
                                 const double *barray,
                                 const uint64_t *astrings,
                                 const uint64_t *bstrings,
                                 const int lena,
                                 const int lenb) {
  double *alpha = safe_malloc(alpha, lena);
  double *beta = safe_malloc(beta, lenb);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < lena; ++i) {
    integer_index_accumulate_real(alpha + i, aarray, astrings[i]);
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < lenb; ++i) {
    integer_index_accumulate_real(beta + i, barray, bstrings[i]);
  }
#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < lena; ++i) {
    for (int j = 0; j < lenb; ++j) {
      data[j + lenb * i] *= alpha[i] + beta[j];
    }
  }
  free(alpha);
  free(beta);
}


void evolve_diagonal_inplace(double complex *data,
                             const double complex *aarray,
                             const double complex *barray,
                             const uint64_t *astrings,
                             const uint64_t *bstrings,
                             const int lena,
                             const int lenb) {
  double complex *alpha = safe_malloc(alpha, lena);
  double complex *beta = safe_malloc(beta, lenb);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < lena; ++i) {
    integer_index_accumulate(alpha + i, aarray, astrings[i]);
    alpha[i] = cexp(alpha[i]);
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < lenb; ++i) {
    integer_index_accumulate(beta + i, barray, bstrings[i]);
    beta[i] = cexp(beta[i]);
  }
#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < lena; ++i) {
    for (int j = 0; j < lenb; ++j) {
      data[j + lenb * i] *= alpha[i] * beta[j];
    }
  }
  free(alpha);
  free(beta);
}


void evolve_diagonal_inplace_real(double complex *data,
                                  const double *aarray,
                                  const double *barray,
                                  const uint64_t *astrings,
                                  const uint64_t *bstrings,
                                  const int lena,
                                  const int lenb) {
  double *alpha = safe_malloc(alpha, lena);
  double *beta = safe_malloc(beta, lenb);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < lena; ++i) {
    integer_index_accumulate_real(alpha + i, aarray, astrings[i]);
    alpha[i] = exp(alpha[i]);
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < lenb; ++i) {
    integer_index_accumulate_real(beta + i, barray, bstrings[i]);
    beta[i] = exp(beta[i]);
  }
#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < lena; ++i) {
    for (int j = 0; j < lenb; ++j) {
      data[j + lenb * i] *= alpha[i] * beta[j];
    }
  }
  free(alpha);
  free(beta);
}


int evaluate_map_each(int64_t *out,
                      const uint64_t *strings,
                      const int length,
                      const int ipmask,
                      const int ihmask) {
  const uint64_t pmask = ipmask;
  const uint64_t hmask = ihmask;
  int counter = 0;
  for (int i = 0; i < length; ++i) {
    const uint64_t current = strings[i];
    if (((~current) & pmask) == 0 && (current & hmask) == 0) {
      out[counter] = i;
      ++counter;
    }
  }
  return counter;
}


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
                     double complex *dvec) {
#pragma omp parallel for schedule(dynamic) collapse(2)
  for (int i = 0; i < norb; ++i)
  for (int j = 0; j < norb; ++j) {
    double complex *dptr = dvec + i*nd2*nd3*nd4 + (j + norb)*nd3*nd4;
    const int32_t *aptr = aarray + i*na*3;
    const int32_t *bptr = barray + j*nb*3;
    for (int k = 0; k < na; ++k) {
      const int sourcea = aptr[k*3];
      const int targeta = aptr[k*3 + 1];
      int paritya = aptr[k*3 + 2];
      paritya *= (nalpha % 2 == 0 ? 1 : -1);
      for (int l = 0; l < nb; ++l) {
        const int sourceb = bptr[l*3];
        const int targetb = bptr[l*3 + 1];
        const int parityb = bptr[l*3 + 2];
        const double complex work = coeff[sourcea*nc2 + sourceb];
        dptr[targeta*nd4 + targetb] += work * paritya * parityb;
      }
    }
  }
}

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
                     double complex *dvec) {
#pragma omp parallel for schedule(dynamic) collapse(2)
  for (int i = 0; i < norb; ++i)
  for (int j = 0; j < norb; ++j) {
    double complex *dptr = dvec + (i + norb)*nd2*nd3*nd4 + j*nd3*nd4;
    const int32_t *aptr = aarray + j*na*3;
    const int32_t *bptr = barray + i*nb*3;
    for (int k = 0; k < na; ++k) {
      const int sourcea = aptr[k*3];
      const int targeta = aptr[k*3 + 1];
      int paritya = aptr[k*3 + 2];
      paritya *= (nalpha % 2 == 0 ? -1 : 1);
      for (int l = 0; l < nb; ++l) {
        const int sourceb = bptr[l*3];
        const int targetb = bptr[l*3 + 1];
        const int parityb = bptr[l*3 + 2];
        const double complex work = coeff[sourcea*nc2 + sourceb];
        dptr[targeta*nd4 + targetb] += work * paritya * parityb;
      }
    }
  }
}

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
                      double complex *out) {
  const double complex *dptr = dvec + (i + norb)*nd2*nd3*nd4 + j*nd3*nd4;
  const int32_t *aptr = aarray + j*na*3;
  const int32_t *bptr = barray + i*nb*3;
  for (int k = 0; k < na; ++k) {
    const int sourcea = aptr[k*3];
    const int targeta = aptr[k*3 + 1];
    int paritya = aptr[k*3 + 2];
    paritya *= (nalpha % 2 == 0 ? 1 : -1);
    for (int l = 0; l < nb; ++l) {
      const int sourceb = bptr[l*3];
      const int targetb = bptr[l*3 + 1];
      const int parityb = bptr[l*3 + 2];
      const double complex work = dptr[targeta*nd4 + targetb];
      out[sourcea*no2 + sourceb] += work * paritya * parityb;
    }
  }
}

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
                      double complex *out) {

  const double complex *dptr = dvec + i*nd2*nd3*nd4 + (j + norb)*nd3*nd4;
  const int32_t *aptr = aarray + i*na*3;
  const int32_t *bptr = barray + j*nb*3;
  for (int k = 0; k < na; ++k) {
    const int sourcea = aptr[k*3];
    const int targeta = aptr[k*3 + 1];
    int paritya = aptr[k*3 + 2];
    paritya *= (nalpha % 2 == 0 ? -1 : 1);
    for (int l = 0; l < nb; ++l) {
      const int sourceb = bptr[l*3];
      const int targetb = bptr[l*3 + 1];
      const int parityb = bptr[l*3 + 2];
      const double complex work = dptr[targeta*nd4 + targetb];
      out[sourcea*no2 + sourceb] += work * paritya * parityb;
    }
  }
}

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
                       double complex *dvec) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < norb; ++i) {
    double complex *dptr = dvec + (i + norb)*nd2*nd3;
    const int32_t *aptr = aarray + j*na*3;
    const int32_t *bptr = barray + i*nb*3;
    for (int k = 0; k < na; ++k) {
      const int sourcea = aptr[k*3];
      const int targeta = aptr[k*3 + 1];
      int paritya = aptr[k*3 + 2];
      paritya *= (nalpha % 2 == 0 ? -1 : 1);
      for (int l = 0; l < nb; ++l) {
        const int sourceb = bptr[l*3];
        const int targetb = bptr[l*3 + 1];
        const int parityb = bptr[l*3 + 2];
        const double complex work = coeff[sourcea*nc2 + sourceb];
        dptr[targeta*nd3 + targetb] += work * paritya * parityb;
      }
    }
  }
}

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
                       double complex *dvec) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < norb; ++i) {
    double complex *dptr = dvec + i*nd2*nd3;
    const int32_t *aptr = aarray + i*na*3;
    const int32_t *bptr = barray + (j - norb)*nb*3;
    for (int k = 0; k < na; ++k) {
      const int sourcea = aptr[k*3];
      const int targeta = aptr[k*3 + 1];
      int paritya = aptr[k*3 + 2];
      paritya *= (nalpha % 2 == 0 ? 1 : -1);
      for (int l = 0; l < nb; ++l) {
        const int sourceb = bptr[l*3];
        const int targetb = bptr[l*3 + 1];
        const int parityb = bptr[l*3 + 2];
        const double complex work = coeff[sourcea*nc2 + sourceb];
        dptr[targeta*nd3 + targetb] += work * paritya * parityb;
      }
    }
  }
}

void make_nh123_real(const int norb,
                     const double *h4e,
                     double *nh1e,
                     double *nh2e,
                     double *nh3e) {
  const int twon = 2 * norb;
  const int s2 = twon * twon;
  const int s3 = s2 * twon;
  const int s4 = s3 * twon;
  const int s5 = s4 * twon;
  const int s6 = s5 * twon;
  const int s7 = s6 * twon;
  for (int i = 0; i < twon; ++i)
  for (int j = 0; j < twon; ++j)
  for (int k = 0; k < twon; ++k) {
    for (int xx = 0; xx < twon; ++xx)
    for (int yy = 0; yy < twon; ++yy) {
      nh1e[xx*twon + yy] -= h4e[
        xx*s7 + j*s6 + i*s5 + k*s4 + j*s3 + i*s2 + k*twon + yy];
    }
  }

#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < twon; ++i)
  for (int j = 0; j < twon; ++j)
  for (int k = 0; k < twon; ++k) {
    double *temp2 = nh2e + i*s3 + j*s2;
    double *temp3 = nh3e + i*s5 + j*s4 + k*s3;
    for (int l = 0; l < twon; ++l) {
      const double *h4jlik = h4e + j*s7 + l*s6 + i*s5 + k*s4;
      const double *h4ijlk = h4e + i*s7 + j*s6 + l*s5 + k*s4;
      const double *h4ilkj = h4e + i*s7 + l*s6 + k*s5 + j*s4;
      const double *h4ikjl = h4e + i*s7 + k*s6 + j*s5 + l*s4;
      const double *h4jikl = h4e + j*s7 + i*s6 + k*s5 + l*s4;
      const double *h4ijkl = h4e + i*s7 + j*s6 + k*s5 + l*s4;
      for (int xx = 0; xx < twon; ++xx)
      for (int yy = 0; yy < twon; ++yy) {
        double temp;
        const int o1 = l*s3 + k*s2 + xx*twon + yy;
        temp = h4jlik[o1];
        temp += h4ijlk[o1];
        temp += h4ilkj[o1];
        temp += h4jikl[o1];

        const int o2 = k*s3 + xx*s2 + l*twon + yy;
        temp += h4ikjl[o2];
        temp += h4jikl[o2];

        const int o3 = xx*s3 + k*s2 + l*twon + yy;
        temp += h4ijkl[o3];
        temp2[xx*twon + yy] += temp;
      }
      const double *h4kijl = h4e + k*s7 + i*s6 + j*s5 + l*s4;
      const double *h4jilk = h4e + j*s7 + i*s6 + l*s5 + k*s4;
      const double *h4iljk = h4e + i*s7 + l*s6 + j*s5 + k*s4;
      for (int xx = 0; xx < twon; ++xx)
      for (int yy = 0; yy < twon; ++yy)
      for (int zz = 0; zz < twon; ++zz) {
        const int o1 = l*s3 + xx*s2 + yy*twon + zz;
        double temp;
        temp = h4kijl[o1];
        temp += h4jilk[o1];
        temp += h4iljk[o1];

        const int o2 = xx*s3 + l*s2 + yy*twon + zz;
        temp += h4ikjl[o2];
        temp += h4ijlk[o2];

        const int o3 = xx*s3 + yy*s2 + l*twon + zz;
        temp += h4ijkl[o3];
        temp3[xx*s2 + yy*twon + zz] += temp;
      }
    }
  }
}

void make_nh123(const int norb,
                const double complex *h4e,
                double complex *nh1e,
                double complex *nh2e,
                double complex *nh3e) {
  const int twon = 2 * norb;
  const int s2 = twon * twon;
  const int s3 = s2 * twon;
  const int s4 = s3 * twon;
  const int s5 = s4 * twon;
  const int s6 = s5 * twon;
  const int s7 = s6 * twon;
  for (int i = 0; i < twon; ++i)
  for (int j = 0; j < twon; ++j)
  for (int k = 0; k < twon; ++k) {
    for (int xx = 0; xx < twon; ++xx)
    for (int yy = 0; yy < twon; ++yy) {
      nh1e[xx*twon + yy] -= h4e[
        xx*s7 + j*s6 + i*s5 + k*s4 + j*s3 + i*s2 + k*twon + yy];
    }
  }

#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < twon; ++i)
  for (int j = 0; j < twon; ++j)
  for (int k = 0; k < twon; ++k) {
    double complex *temp2 = nh2e + i*s3 + j*s2;
    double complex *temp3 = nh3e + i*s5 + j*s4 + k*s3;
    for (int l = 0; l < twon; ++l) {
      const double complex *h4jlik = h4e + j*s7 + l*s6 + i*s5 + k*s4;
      const double complex *h4ijlk = h4e + i*s7 + j*s6 + l*s5 + k*s4;
      const double complex *h4ilkj = h4e + i*s7 + l*s6 + k*s5 + j*s4;
      const double complex *h4ikjl = h4e + i*s7 + k*s6 + j*s5 + l*s4;
      const double complex *h4jikl = h4e + j*s7 + i*s6 + k*s5 + l*s4;
      const double complex *h4ijkl = h4e + i*s7 + j*s6 + k*s5 + l*s4;
      for (int xx = 0; xx < twon; ++xx)
      for (int yy = 0; yy < twon; ++yy) {
        double complex temp;
        const int o1 = l*s3 + k*s2 + xx*twon + yy;
        temp = h4jlik[o1];
        temp += h4ijlk[o1];
        temp += h4ilkj[o1];
        temp += h4jikl[o1];

        const int o2 = k*s3 + xx*s2 + l*twon + yy;
        temp += h4ikjl[o2];
        temp += h4jikl[o2];

        const int o3 = xx*s3 + k*s2 + l*twon + yy;
        temp += h4ijkl[o3];
        temp2[xx*twon + yy] += temp;
      }
      const double complex *h4kijl = h4e + k*s7 + i*s6 + j*s5 + l*s4;
      const double complex *h4jilk = h4e + j*s7 + i*s6 + l*s5 + k*s4;
      const double complex *h4iljk = h4e + i*s7 + l*s6 + j*s5 + k*s4;
      for (int xx = 0; xx < twon; ++xx)
      for (int yy = 0; yy < twon; ++yy)
      for (int zz = 0; zz < twon; ++zz) {
        const int o1 = l*s3 + xx*s2 + yy*twon + zz;
        double complex temp;
        temp = h4kijl[o1];
        temp += h4jilk[o1];
        temp += h4iljk[o1];

        const int o2 = xx*s3 + l*s2 + yy*twon + zz;
        temp += h4ikjl[o2];
        temp += h4ijlk[o2];

        const int o3 = xx*s3 + yy*s2 + l*twon + zz;
        temp += h4ijkl[o3];
        temp3[xx*s2 + yy*twon + zz] += temp;
      }
    }
  }
}
