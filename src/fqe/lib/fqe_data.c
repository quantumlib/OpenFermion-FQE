#include "fqe_data.h"

#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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
    // We have: map[i] = [state, target, parity]
    const double complex * coeff_p = coeff + inc_A * (*cmap)[0];
    double complex * dvec_p = dvec + inc_A * (*cmap)[1];
    const double complex parity =  (*cmap)[2];
    blasfunc->zaxpy(&maxj, &parity, coeff_p, &inc_B, dvec_p, &inc_B);
  }
}

/* 
 * Low memory version for the _apply_array_spatial12_halffilling.
 */
int lm_apply_array12_same_spin(const double complex *coeff,
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
      // *cdexc = [target, orbi, orbj, parity]
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

/* Excpects alpha excitation for ij and beta excitation for kl */
int lm_apply_array12_diff_spin(const double complex *coeff,
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

  // Prepare targetbs and prefactors
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

        // Beta excitations
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

int lm_apply_array1(const double complex *coeff,
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
      // *cdexc = [target, orbj, orbi, parity]
      // The array dexc gives the dexcitation, I want the dedexc
      const int target = (*cdexc)[0];
      const int ijshift = (*cdexc)[1];
      const int parity = (*cdexc)[2];

      const double complex *ccoeff = &coeff[target * inc1];
      const double complex pref = parity * h1e[ijshift];
      // Dedicated code for fixedj?
      blasfunc->zaxpy(&states2, &pref, ccoeff, &inc2, cout, &ONE);
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

  // Quite unbalanced parallelization, I know
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

  // This one is harder to parallelize due to race conditions without
  // duplication of coeff.
  //
  // TODO: omp parallelize through duplication of coeff.
  for (int mapno = 0; mapno < nr_maps; ++mapno) {
    const int (*cmap)[3] = map[mapno];
    const double complex *cdvec = dvec + dvec_inc * mapno;

    for (; cmap < &map[mapno][map_els[mapno]]; ++cmap) {
      // We have: map[i] = [state, target, parity]
      double complex *coeff_p = coeff + inc_A * (*cmap)[0];
      const double complex *dvec_p = cdvec + inc_A * (*cmap)[1];
      const double complex parity = (*cmap)[2];
      blasfunc->zaxpy(&maxj, &parity, dvec_p, &inc_B, coeff_p, &inc_B);
    }
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
    double complex *p_adiag = &output[i];
    const int *c_occ = &occ[i * nel];

    for (int el = 0; el < nel; ++el) {
      *p_adiag += diag[c_occ[el]];
      const double complex *c_array = array + norbs * c_occ[el];
      for (int el2 = 0; el2 < nel; ++el2) {
        *p_adiag += c_array[c_occ[el2]];
      }
    }
  }
  return 0;
}

int zdiagonal_coulomb(const unsigned int *alpha_strings,
                      const unsigned int *beta_strings,
                      const double complex *diag,
                      const double complex *array,
                      double complex *output,
                      const int alpha_states,
                      const int beta_states,
                      const int nalpha,
                      const int nbeta,
                      const int norbs) {
  double complex *adiag = safe_calloc(adiag, alpha_states);
  double complex *bdiag = safe_calloc(bdiag, beta_states);
  int *aocc = safe_malloc(aocc, alpha_states * nalpha);
  int *bocc = safe_malloc(bocc, beta_states * nbeta);

#pragma omp parallel for schedule(static)
  for (int as = 0; as < alpha_states; ++as) {
    get_occupation(&aocc[as * nalpha], alpha_strings[as], nalpha, norbs);
  }
#pragma omp parallel for schedule(static)
  for (int bs = 0; bs < beta_states; ++bs) {
    get_occupation(&bocc[bs * nbeta], beta_strings[bs], nbeta, norbs);
  }

  zdiagonal_coulomb_part(aocc, diag, array, adiag, alpha_states, nalpha, norbs);
  zdiagonal_coulomb_part(bocc, diag, array, bdiag, beta_states, nbeta, norbs);

#pragma omp parallel shared(aocc, bocc, output, array, adiag, bdiag)
  {
    double complex *aarrays = safe_malloc(aarrays, norbs);

#pragma omp for schedule(static)
    for (int as = 0; as < alpha_states; ++as) {
      double complex *c_output = output + as * beta_states;
      const int *caocc = &aocc[as * nalpha];

      for (int i = 0; i < norbs; ++i) { aarrays[i] = 0.0; }
      for (int ela = 0; ela < nalpha; ++ela) {
        const double complex *carray = array + caocc[ela] * norbs;
        for (int i = 0; i < norbs; ++i) {
          aarrays[i] += carray[i];
        }
      }

      for (int bs = 0; bs < beta_states; ++bs) {
        double complex diag_ele = 0.0;
        const int *cbocc = &bocc[bs * nbeta];
        for (int elb = 0; elb < nbeta; ++elb) {
          diag_ele += aarrays[cbocc[elb]];
        }
        diag_ele = diag_ele * 2.0 + adiag[as] + bdiag[bs];
        c_output[bs] *= cexp(diag_ele);
      }
    }
    free(aarrays);
  }

  free(adiag);
  free(bdiag);
  free(aocc);
  free(bocc);
  return 0;
}

int make_dvec_part(const int astates_full,
                   const int bstates_full,
                   const int astates_part_begin,
                   const int bstates_part_begin,
                   const int astates_part_num,
                   const int bstates_part_num,
                   const int (*maps)[4],
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

  // These maps should be ordered as to be as continous as possible for output
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
                   const struct blasfunctions * blasfunc) {
  const int ONE = 1;
  const int totstates = states1_part_num * states2_part_num;
  // These maps should be ordered as to be as continous for output
//#pragma omp parallel for schedule(static)
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
      // *cdexc = [target, orbi, orbj, parity]
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
        //zaxpy_(&states2, &pref, ccoeff, &inc2, cout, &inc2);
      }
    }
    const double complex *xptr = coeff;
    for (int ii = 0; ii < states1; ii++) {
      const double complex ttt = temp[ii];
      //for (int jj = 0; jj < states2; jj++) {
      //    cout[jj] += temp[ii]*xptr[jj];
      //}
      blasfunc->zaxpy(&states2, &ttt, xptr, &inc2, cout, &inc2);
      xptr += inc1;
    }
  }
  free(temp);
  }  // end parallel block
}

/* Excpects alpha excitation for ij and beta excitation for kl */
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

  // This can be replaced, but it's cheap
  int nest = 0;
  for (int s2 = 0; s2 < beta_states; ++s2)
  for (int i = 0; i < nbdexc; ++i) {
    const int orbkl = bdexc[3*(s2*nbdexc + i) + 1];
    if (orbkl == 0) ++nest;
  }
  double complex *vtemp = safe_malloc(vtemp, nest);
  double complex *ctemp = safe_malloc(ctemp, nest * alpha_states);

  // Loop over orbital pairs
  for (int orbid = 0; orbid < norbs2; ++orbid) {
    int nsig = 0;
    // Loop over beta strings and excitations
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
      //gather_kernel(alpha_states, zsign, cptr, beta_states, tptr, nsig);
      //for (int s2 = 0; s2 < alpha_states; ++s2) {
      //  tptr[s2*nsig] = zsign * cptr[s2*beta_states];
      //}
    }

    // contract and scatter
    const complex double *tmperi = h2e + orbid*norbs2;
    for (int s1 = 0; s1 < alpha_states; ++s1) {
      for (int kk = 0; kk < nsig; ++kk) vtemp[kk] = 0.0;
      for (int j = 0; j < nadexc; ++j) {
        int idx2 = adexc[3*(s1*nadexc + j)];
        const int parity = adexc[3*(s1*nadexc + j) + 2];
        const int orbij = adexc[3*(s1*nadexc + j) + 1];
        //const double complex ttt = parity * h2e[orbid * norbs2 + orbij];
        const double complex ttt = parity * tmperi[orbij];
        const double complex *cctmp = ctemp + idx2 * nsig;
        blasfunc->zaxpy(&nsig, &ttt, cctmp, &one, vtemp, &one);
        //for (int isig = 0; isig < nsig; ++isig) {
        //        vtemp[isig] += ttt * cctmp[isig];
        //}
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

/* Excpects alpha excitation for ij and beta excitation for kl */
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

  // This can be replaced, but it's cheap
  int nest = 0;
  for (int s1 = 0; s1 < alpha_states; ++s1)
  for (int i = 0; i < nadexc; ++i) {
    const int orbij = adexc[3*(s1*nadexc + i) + 1];
    if (orbij == 0) ++nest;
  }
  double complex *vtemp = safe_malloc(vtemp, nest);
  double complex *ctemp = safe_malloc(ctemp, nest * beta_states);

  // Loop over orbital pairs
  for (int orbid = 0; orbid < norbs2; ++orbid) {
    int nsig = 0;
    // Loop over alpha strings and excitations
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
      //zaxpy_(&alpha_states, &zsign, cptr, &beta_states, tptr, &nsig);
      blasfunc->zaxpy(&beta_states, &zsign, cptr, &one, tptr, &nsig);
      //gather_kernel(alpha_states, zsign, cptr, beta_states, tptr, nsig);
      //for (int s2 = 0; s2 < alpha_states; ++s2) {
      //  tptr[s2*nsig] = zsign * cptr[s2*beta_states];
      //}
    }

    // contract and scatter
    const complex double *tmperi = h2e + orbid*norbs2;
    for (int s2 = 0; s2 < beta_states; ++s2) {
      for (int kk = 0; kk < nsig; ++kk) vtemp[kk] = 0.0;
      for (int j = 0; j < nbdexc; ++j) {
        int idx2 = bdexc[3*(s2*nbdexc + j)];
        const int parity = bdexc[3*(s2*nbdexc + j) + 2];
        const int orbkl = bdexc[3*(s2*nbdexc + j) + 1];
        //const double complex ttt = parity * h2e[orbid * norbs2 + orbij];
        const double complex ttt = parity * tmperi[orbkl];
        const double complex * cctmp = ctemp + idx2 * nsig;
        blasfunc->zaxpy(&nsig, &ttt, cctmp, &one, vtemp, &one);
        //for (int isig = 0; isig < nsig; ++isig) {
        //  vtemp[isig] += ttt * cctmp[isig];
        //}
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

/* Excpects alpha excitation for ij and beta excitation for kl */
int lm_apply_array12_diff_spin_omp1(const double complex *coeff,
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
  const int nthrds = omp_get_max_threads();
  int *signs = safe_malloc(signs, nadexc_tot);
  int *coff = safe_malloc(coff, nadexc_tot);
  int *boff = safe_malloc(boff, nadexc_tot);

  // This can be replaced, but it's cheap
  int nest = 0;
  for (int s1 = 0; s1 < alpha_states; ++s1)
  for (int i = 0; i < nadexc; ++i) {
    const int orbij = adexc[3*(s1*nadexc + i) + 1];
    if (orbij == 0) ++nest;
  }
  double complex *vtemp = safe_malloc(vtemp, nthrds * nest);
  double complex *ctemp = safe_malloc(ctemp, nest * beta_states);

  // Loop over orbital pairs
  for (int orbid = 0; orbid < norbs2; ++orbid) {
    int nsig = 0;
    // Loop over alpha strings and excitations
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
      //for (int s2 = 0; s2 < alpha_states; ++s2) {
      //  tptr[s2*nsig] = zsign * cptr[s2*beta_states];
      //}
    }

    // contract and scatter
    const complex double *tmperi = h2e + orbid*norbs2;
#pragma omp for schedule(static)
    for (int s2 = 0; s2 < beta_states; ++s2) {
      //for (int kk = 0; kk < nsig; kk++) vtemp[kk] = 0.0;
      const int ithrd = omp_get_thread_num();
      double complex *vpt = vtemp + ithrd*nsig;
      for (int kk = 0; kk < nsig; ++kk) vpt[kk] = 0.0;
      for (int j = 0; j < nbdexc; ++j) {
        int idx2 = bdexc[3*(s2*nbdexc + j)];
        const int parity = bdexc[3*(s2*nbdexc + j) + 2];
        const int orbkl = bdexc[3*(s2*nbdexc + j) + 1];
        const double complex ttt = parity * tmperi[orbkl];
        const double complex * cctmp = ctemp + idx2 * nsig;
        blasfunc->zaxpy(&nsig, &ttt, cctmp, &one, vpt, &one);
        //for (int isig = 0; isig < nsig; ++isig) {
        //        vtemp[isig] += ttt * cctmp[isig];
        //}
      }

      double complex *tmpout = out + s2;
      for (int isig = 0; isig < nsig; ++isig) {
        //tmpout[beta_states*boff[isig]] += vtemp[isig];
        tmpout[beta_states*boff[isig]] += vpt[isig];
      }
    }
    }  // end parallel block
  }

  free(vtemp);
  free(signs);
  free(ctemp);
  free(coff);
  free(boff);
}

/* Excpects alpha excitation for ij and beta excitation for kl */
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
                                   const struct blasfunctions * blasfunc) {
  //return lm_apply_array12_diff_spin_opt1(
  return lm_apply_array12_diff_spin_omp1(
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
