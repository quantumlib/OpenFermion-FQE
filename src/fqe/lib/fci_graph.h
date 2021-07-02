#pragma once
#include <stdbool.h>

int build_mapping_strings(int (**maps)[3],
                          int *mapl,
                          int (*exc_deexc)[2],
                          const int nmaps,
                          const unsigned int *strings,
                          const int nstrings,
                          const bool count,
                          const int *Z_matrix,
                          const int norb);

int string_to_index(unsigned int string,
                    const int *Z_matrix,
                    const int norb);
