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

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "fci_graph.h"
#include "bitstring.h"
#include "macros.h"
#include "binom.h"

void calculate_Z_matrix(int32_t *out,
                        int norb,
                        int nele) {
#define NB_ 65
  uint64_t* binom = safe_malloc(binom, NB_ * NB_);
  initialize_binom(binom);

#pragma omp parallel for schedule(static) collapse(2)
  for (int km = 0; km < nele - 1; ++km) {
    for (int llm = 0; llm < norb - nele + 1; ++llm) {
      const int k = km + 1;
      const int ll = llm + k;
      int64_t tmp = 0;
      for (int m = norb - ll + 1; m < norb - k + 1; ++m) {
        tmp += binom[nele - k + NB_ * m] - binom[nele - k - 1 + NB_ * (m - 1)];
      }
      out[ll - 1 + norb * (k - 1)] = (int32_t)tmp;
    }
  }
  {
    int k = nele;
    for (int ll = nele; ll < norb + 1; ++ll) {
      out[ll - 1 + norb * (k - 1)] = ll - nele;
    }
  }
  free(binom);
#undef NB_
}


int map_deexc(int32_t *out,
              const int32_t *inp,
              const int lk,
              const int size,
              uint32_t *index,
              const int idx) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    const int target = inp[1 + i*3];
    int32_t *out3k = out + 3 * (index[target] + lk * target);
    out3k[0] = inp[0 + i*3];
    out3k[1] = idx;
    out3k[2] = inp[2 + i*3];
    index[target] += 1;
  }
  return 0;
}


void build_mapping_strings(int (**maps)[3],
                          int32_t *mapl,
                          const int32_t (*exc_deexc)[2],
                          const int nmaps,
                          const uint64_t *strings,
                          const int nstrings,
                          const bool count,
                          const int32_t *Z_matrix,
                          const int norb) {
#pragma omp parallel for schedule(static)
  for (int mapno = 0; mapno < nmaps; ++mapno) {
    const int iorb = exc_deexc[mapno][0];
    const int jorb = exc_deexc[mapno][1];

    int (*cmap)[3] = count ? NULL : maps[mapno];
    int counter = 0;
    for (int stringno = 0; stringno < nstrings; ++stringno) {
      uint64_t cstring = strings[stringno];
      if (CHECK_BIT(cstring, jorb) && !CHECK_BIT(cstring, iorb)) {
        if (!count) {
          (*cmap)[0] = string_to_index(cstring, Z_matrix, norb);
          (*cmap)[1] = string_to_index(
              UNSET_BIT(SET_BIT(cstring, iorb), jorb), Z_matrix, norb);
          (*cmap)[2] = count_bits_between(cstring, iorb, jorb) % 2 == 0 ? 1 : -1;
          ++cmap;
        }
        ++counter;
      } else if (iorb == jorb && CHECK_BIT(cstring, iorb)) {
        if (!count) {
          const int cid = string_to_index(cstring, Z_matrix, norb);
          (*cmap)[0] = cid;
          (*cmap)[1] = cid;
          (*cmap)[2] = 1;
          ++cmap;
        }
        ++counter;
      }
    }

    if (count) {
      mapl[mapno] = counter;
    } else if (counter != mapl[mapno]) {
      fprintf(stderr, "Length of map %d not consistent.\n", mapno);
    }
  }
}

int string_to_index(uint64_t string,
                    const int32_t *Z_matrix,
                    const int norb) {
  int occ = -1;
  int counter = 0;
  int id = 0;
  while (gbit_index(&string, &occ)) {
    id += Z_matrix[counter * norb + occ];
    ++counter;
  }
  return id;
}

void calculate_string_address(uint64_t *out,
                              const uint64_t *strings,
                              const int length,
                              const int32_t *Z_matrix,
                              const int norb) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < length; ++i) {
    const uint64_t string = strings[i];
    const int address = string_to_index(string, Z_matrix, norb);
    out[address] = string;
  }
}

void map_to_deexc_alpha_icol(const int32_t ** mappings,
                             const int32_t * mapl,
                             const uint64_t * strings,
                             const int nstrings,
                             int32_t * exc,
                             int32_t * diag,
                             int32_t * index,
                             const int norb,
                             const int exc0,
                             const int exc1,
                             const int ldiag) {
  int * alpha = safe_malloc(alpha, norb * nstrings);
  int * count = safe_calloc(count, norb);
  int * counter = safe_calloc(counter, norb * exc0);
  for (int i = 0; i < norb * nstrings; ++i) alpha[i] = -1;

  const uint64_t filled_string = ((uint64_t) 1 << norb) - (uint64_t) 1;
  for (int i = 0; i < nstrings; ++i) {
    uint64_t revstring = (~strings[i]) & filled_string;
    int icol = -1;
    while (gbit_index(&revstring, &icol)) {
      assert(icol < norb);
      alpha[icol * nstrings +  i] = count[icol];
      index[icol * exc0 + count[icol]] = i;
      ++count[icol];
    }
  }

  for (int i = 0; i < norb; ++i) {
    assert(count[i] == exc0);
    count[i] = 0;
  }

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < norb; ++i) {
    int32_t * c_exc = &exc[i * exc0 * exc1 * 3];
    int32_t * c_diag = &diag[i * ldiag];
    int * c_alpha = &alpha[i * nstrings];
    int * ccount = &count[i];
    int * counteri = &counter[i * exc0];

    for (int j = 0; j < norb; ++j) {
      const int32_t * mapping = mappings[i * norb + j];
      const int32_t nmaps = mapl[i * norb + j];
      if (i != j) {
        for (int mapid = 0; mapid < nmaps; ++mapid) {
          const int32_t source = mapping[3 * mapid + 0];
          const int32_t target = mapping[3 * mapid + 1];
          const int32_t parity = mapping[3 * mapid + 2];

          const int pos = c_alpha[target];
          assert(pos != -1);
          int * c_counter = &counteri[pos];

          int * cc_exc = &c_exc[(pos * exc1 + (*c_counter)) * 3];
          cc_exc[0] = source;
          cc_exc[1] = j;
          cc_exc[2] = parity;
          ++(*c_counter);
        }
      } else {
        for (int mapid = 0; mapid < nmaps; ++mapid) {
          assert(mapping[3 * mapid + 0] == mapping[3 * mapid + 1]);
          assert(mapping[3 * mapid + 2] == 1);
          c_diag[(*ccount)++] = mapping[3 * mapid + 1];
        }
      }
    }
  }
  free(alpha);
  free(count);
  free(counter);
}

int make_mapping_each(uint64_t *out,
                      const uint64_t *strings,
                      const int length,
                      const int32_t *dag,
                      const int dag_length,
                      const int32_t *undag,
                      const int undag_length) {
  uint64_t dag_mask = 0;
  for (int i = 0; i != dag_length; ++i) {
    dag_mask = SET_BIT(dag_mask, dag[i]);
  }

  uint64_t undag_mask = 0;
  for (int i = 0; i != undag_length; ++i) {
    undag_mask = SET_BIT(undag_mask, undag[i]);
    dag_mask = UNSET_BIT(dag_mask, undag[i]);
  }

  int count = 0;
  for (int i = 0; i < length; ++i) {
    uint64_t current = strings[i];
    const uint64_t dag_masked = current & dag_mask;
    const uint64_t undag_masked = current & undag_mask;
    const bool check = !dag_masked && !(undag_masked ^ undag_mask);
    if (check) {
      int parity = 0;
      for (int j = undag_length-1; j >= 0; --j) {
        parity += count_bits_above(current, undag[j]);
        current = UNSET_BIT(current, undag[j]);
      }
      for (int j = dag_length-1; j >= 0; --j) {
        parity += count_bits_above(current, dag[j]);
        current = SET_BIT(current, dag[j]);
      }
      out[count * 3] = i;
      out[count * 3 + 1] = current;
      out[count * 3 + 2] = parity % 2;
      ++count;
    }
  }
  return count;
}

void make_mapping_each_set(uint64_t *down,
                           uint64_t *up,
                           const uint64_t *istrings,
                           const int length,
                           const int msize,
                           const int nsize,
                           const int dn,
                           const int norb) {

  uint64_t* comb = safe_malloc(comb, msize);
  lexicographic_bitstring_generator(comb, norb, dn);

#pragma omp parallel for schedule(dynamic)
  for (int c = 0; c < msize; ++c) {
    const uint64_t mask = comb[c];
    int occ[16];
    assert(count_bits(mask) == dn && dn < 16);
    get_occupation(occ, mask);

    int count = 0;
    for (int i = 0; i != length; ++i) {
      const uint64_t source = istrings[i];
      if (((source & mask) ^ mask) == 0) {
        int parity = count_bits_above(source, occ[dn - 1]) * dn;
        uint64_t target = UNSET_BIT(source, occ[dn - 1]);
        for (int d = dn - 2; d >= 0; --d) {
          parity += (d + 1) * count_bits_between(source, occ[d], occ[d+1]);
          target = UNSET_BIT(target, occ[d]);
        }
        down[0 + 3 * (count + nsize * c)] = source;
        down[1 + 3 * (count + nsize * c)] = target;
        down[2 + 3 * (count + nsize * c)] = parity;
        up[0 + 3 * (count + nsize * c)] = target;
        up[1 + 3 * (count + nsize * c)] = source;
        up[2 + 3 * (count + nsize * c)] = parity;
        ++count;
      }
    }
  }

  free(comb);
}
