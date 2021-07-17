#include "bitstring.h"

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
