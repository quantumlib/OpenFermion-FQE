#pragma once
#include <stdint.h>
#include <stdbool.h>

int map_deexc(int32_t *out,
              const int32_t *inp,
              const int lk,
              const int size,
              uint32_t *index,
              const int idx);

void build_mapping_strings(int (**maps)[3],
                          int32_t *mapl,
                          const int32_t (*exc_deexc)[2],
                          const int nmaps,
                          const uint64_t *strings,
                          const int nstrings,
                          const bool count,
                          const int32_t *Z_matrix,
                          const int norb);

int string_to_index(uint64_t string,
                    const int32_t *Z_matrix,
                    const int norb);

int calculate_string_address(const int32_t *zmat,
                             const int nele,
                             const int norb,
                             const uint64_t occupation); 

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
                             const int ldiag);
