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

#ifndef FQE_LIB_BITSTRING_H_
#define FQE_LIB_BITSTRING_H_

#include <stdio.h>
#include <stdint.h>

#define CHECK_BIT(b,pos) ((b) & (1ull<<(pos)))
#define SET_BIT(b,pos) ((b) | (1ull<<(pos)))
#define UNSET_BIT(b,pos) ((b) & (~(1ull<<(pos))))

inline int gbit_index(uint64_t *str, int *bit_index) {
#ifdef __GNUC__
  const int pos = __builtin_ffsll(*str);
  *str >>= pos;
  *bit_index += pos;
  return pos;
#else
  if (*bit_index == -1) { *str = *str << 1; }
  while  (*str) {
    *str = *str >> 1;
    *bit_index += 1;
    if (*str & 1) { return 1; }
  }
  return 0;
#endif
}

inline int count_bits(const uint64_t cstring) {
#ifdef __GNUC__
  return __builtin_popcountll(cstring);
#else
  uint64_t v = cstring;
#define T uint64_t
  v = v - ((v >> 1) & (T)~(T)0/3);
  v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);
  v = (v + (v >> 4)) & (T)~(T)0/255*15;
  return (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * 8;
#undef T
#endif
}

inline int count_bits_between(uint64_t cstring, const int i, const int j) {
  cstring &= (((1ull << i) - 1) ^ ((2ull << j) - 1));
  cstring &= (((1ull << j) - 1) ^ ((2ull << i) - 1));
  return count_bits(cstring);
}


inline int count_bits_above(uint64_t cstring, const int i) {
  cstring &= ~((2ull << i) - 1);
  return count_bits(cstring);
}

inline int get_occupation(int *occ,
                          uint64_t str) {
  int id = -1;
  int count = 0;
  while (gbit_index(&str, &id)) {
    occ[count++] = id;
  }
  return count;
}

void lexicographic_bitstring_generator(uint64_t *out,
                                       int norb,
                                       int nele);

#endif  // FQE_LIB_BITSTRING_H_
