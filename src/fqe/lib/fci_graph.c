#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "fci_graph.h"
#include  "bitstring.h"
#include "macros.h"

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

int calculate_string_address(const int32_t *zmat,
                             const int nele,
                             const int norb,
                             const uint64_t occupation) {
  int array[nele];
  get_occupation(array, occupation, nele, norb);
  int result = 0;
  for (int i = 0; i < nele; ++i)
    result += zmat[array[i] + i * norb];
  return result;
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
    int * counter = safe_calloc(counter, exc0);
    int32_t * c_exc = &exc[i * exc0 * exc1 * 3];
    int32_t * c_diag = &diag[i * ldiag];
    int * c_alpha = &alpha[i * nstrings];
    int * ccount = &count[i];

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
          int * c_counter = &counter[pos];

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
}
