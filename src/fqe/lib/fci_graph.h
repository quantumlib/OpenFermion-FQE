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

#ifndef SRC_FQE_LIB_FCI_GRAPH_H_
#define SRC_FQE_LIB_FCI_GRAPH_H_
#include <stdint.h>
#include <stdbool.h>

void calculate_Z_matrix(int32_t *out,
                        int norb,
                        int nele);

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

void calculate_string_address(uint64_t *out,
                              const uint64_t *strings,
                              const int length,
                              const int32_t *Z_matrix,
                              const int norb);

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
int make_mapping_each(uint64_t *out,
                      const uint64_t *strings,
                      const int length,
                      const int32_t *dag,
                      const int dag_length,
                      const int32_t *undag,
                      const int undag_length);

void make_mapping_each_opt(uint64_t *down,
                           uint64_t *up,
                           const uint64_t *istrings,
                           const int length,
                           const int msize,
                           const int nsize,
                           const int dn,
                           const int norb);

#endif  // SRC_FQE_LIB_FCI_GRAPH_H_
