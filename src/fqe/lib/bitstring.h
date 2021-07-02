#pragma once

#define CHECK_BIT(b,pos) ((b) & (1<<(pos)))
#define SET_BIT(b,pos) ((b) | (1<<(pos)))
#define UNSET_BIT(b,pos) ((b) & (~(1<<(pos))))

inline int gbit_index(unsigned int *str, int *bit_index) {
  if (*bit_index == -1) { *str = *str << 1; }
  while  (*str) {
    *str = *str >> 1;
    *bit_index += 1;
    if (*str & 1) { return 1; }
  }
  return 0;
}

inline int count_bits_between(unsigned int cstring, const int i, const int j) {
  // Simple implementation. Not necessarily efficient.
  const int minshift = i > j ? j : i;
  const int length = abs(i - j) - 1;
  cstring = cstring >> (minshift + 1);
  int count = 0;
  for (int c = 0; c < length; ++c) {
    count += cstring & 1;
    cstring = cstring >> 1;
  }
  return count;
}

inline int get_occupation(int *occ,
                          unsigned int str,
                          const int nel,
                          const int norb) {
  int id = -1;
  int count = 0;
  while (gbit_index(&str, &id)) { occ[count++] = id; }
  if (nel != count) {
    fprintf(stderr, "Counted electrons is not same as passed electrons.\n");
    return 1;
  }
  return 0;
}
