#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <stdbool.h>

#include "fci_graph.h"
#include  "bitstring.h"

int build_mapping_strings(int (**maps)[3],
                          int *mapl,
                          int (*exc_deexc)[2],
                          const int nmaps,
                          const unsigned int * strings,
                          const int nstrings,
                          const bool count,
                          const int *Z_matrix,
                          const int norb) {
  for (int mapno = 0; mapno < nmaps; ++mapno) {
    const int iorb = exc_deexc[mapno][0];
    const int jorb = exc_deexc[mapno][1];

    int (*cmap)[3] = count ? NULL : maps[mapno];
    int counter = 0;
    for (int stringno = 0; stringno < nstrings; ++stringno) {
      unsigned int cstring = strings[stringno];
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

int string_to_index(unsigned int string,
                    const int *Z_matrix,
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
