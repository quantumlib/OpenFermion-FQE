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

#include "bitstring.h"

int gbit_index(uint64_t *str, int *bit_index);

int count_bits(const uint64_t cstring);

int count_bits_between(uint64_t cstring, const int i, const int j);

int count_bits_above(uint64_t cstring, const int i);

int get_occupation(int *occ, uint64_t str);

void lexicographic_bitstring_generator(uint64_t *out,
                                       int norb,
                                       int nele) {
  if (nele == 0) {
    out[0] = 0;
    return;
  }
  int64_t combo = (1ll << nele) - 1;
  while (combo < (1ll << norb)) {
    *out++ = combo;
    const int64_t x = combo & -combo;
    const int64_t y = combo + x;
    const int64_t z = (combo & ~y);
    combo = z / x;
    combo >>= 1;
    combo |= y;
  }
}
