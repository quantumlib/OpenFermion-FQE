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

#include "wick.h"
#include <stdbool.h>
#include <assert.h>

/* C-kernel for filling custom RDMs using particle RDMs. */
int wickfill(double complex *target,
             const double complex *source,
             const uint32_t *indices,
             const double factor,
             const uint32_t *delta,
             const int norb,
             const int trank,
             const int srank) {
  if (srank == 0 && trank == 1) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < norb; ++i) {
      target[i + norb * i] += factor;
    }
  } else if (srank == 1 && trank == 1) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < norb; ++i) {
      for (int j = 0; j != norb; ++j) {
        const int mat[2] = {i, j};
        target[j + norb * i] += factor * source[mat[indices[1]]
                             + norb * mat[indices[0]]];
      }
    }
  } else if (srank == 0 && trank == 2) {
#pragma omp parallel for schedule(static)
    for (int ij = 0; ij < norb * norb; ++ij) {
      const int i = ij / norb;
      const int j = ij % norb;
      for (int k = 0; k != norb; ++k) {
        for (int l = 0; l != norb; ++l) {
          const int mat[4] = {i, j, k, l};
          if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]] &&
              mat[delta[1 * 2 + 0]] == mat[delta[1 * 2 + 1]]) {
            target[l + norb * (k + norb * (j + norb * i))] += factor;
          }
        }
      }
    }
  } else if (srank == 1 && trank == 2) {
#pragma omp parallel for schedule(static)
    for (int ij = 0; ij < norb * norb; ++ij) {
      const int i = ij / norb;
      const int j = ij % norb;
      for (int k = 0; k != norb; ++k) {
        for (int l = 0; l != norb; ++l) {
          const int mat[4] = {i, j, k, l};
          if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]]) {
            target[l + norb * (k + norb * (j + norb * i))]
              += factor * source[mat[indices[1]]
              + norb * mat[indices[0]]];
          }
        }
      }
    }
  } else if (srank == 2 && trank == 2) {
#pragma omp parallel for schedule(static)
    for (int ij = 0; ij < norb * norb; ++ij) {
      const int i = ij / norb;
      const int j = ij % norb;
      for (int k = 0; k != norb; ++k) {
        for (int l = 0; l != norb; ++l) {
          const int mat[4] = {i, j, k, l};
          target[l + norb * (k + norb * (j + norb * i))]
            += factor * source[mat[indices[3]]
            + norb * (mat[indices[2]]
            + norb * (mat[indices[1]]
            + norb * mat[indices[0]]))];
        }
      }
    }
  } else if (srank == 0 && trank == 3) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < norb; ++i) {
      for (int j = 0; j != norb; ++j) {
        for (int k = 0; k != norb; ++k) {
          for (int l = 0; l != norb; ++l) {
            for (int o = 0; o != norb; ++o) {
              for (int p = 0; p != norb; ++p) {
                const int mat[6] = {i, j, k, l, o, p};
                if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]] &&
                    mat[delta[1 * 2 + 0]] == mat[delta[1 * 2 + 1]] &&
                    mat[delta[2 * 2 + 0]] == mat[delta[2 * 2 + 1]]) {
                  target[p + norb * (o + norb
                    * (l + norb * (k + norb * (j + norb * i))))]
                    += factor;
                }
              }
            }
          }
        }
      }
    }
  } else if (srank == 1 && trank == 3) {
#pragma omp parallel for schedule(static)
    for (int ijk = 0; ijk < norb * norb * norb; ++ijk) {
      const int i = ijk / (norb * norb);
      const int j = (ijk % (norb * norb)) / norb;
      const int k = (ijk % (norb * norb)) % norb;
      for (int l = 0; l != norb; ++l) {
        for (int o = 0; o != norb; ++o) {
          for (int p = 0; p != norb; ++p) {
            const int mat[6] = {i, j, k, l, o, p};
            if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]] &&
                mat[delta[1 * 2 + 0]] == mat[delta[1 * 2 + 1]]) {
              target[p + norb * (o + norb
                * (l + norb * (k + norb * (j + norb * i))))]
                += factor * source[mat[indices[1]] + norb * mat[indices[0]]];
            }
          }
        }
      }
    }
  } else if (srank == 2 && trank == 3) {
#pragma omp parallel for schedule(static)
    for (int ijk = 0; ijk < norb * norb * norb; ++ijk) {
      const int i = ijk / (norb * norb);
      const int j = (ijk % (norb * norb)) / norb;
      const int k = (ijk % (norb * norb)) % norb;
      for (int l = 0; l != norb; ++l) {
        for (int o = 0; o != norb; ++o) {
          for (int p = 0; p != norb; ++p) {
            const int mat[6] = {i, j, k, l, o, p};
            if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]]) {
              target[p + norb * (o + norb
                * (l + norb * (k + norb * (j + norb * i))))]
                += factor * source[mat[indices[3]]
                + norb * (mat[indices[2]]
                + norb * (mat[indices[1]]
                + norb * mat[indices[0]]))];
            }
          }
        }
      }
    }
  } else if (srank == 3 && trank == 3) {
#pragma omp parallel for schedule(static)
    for (int ijk = 0; ijk < norb * norb * norb; ++ijk) {
      const int i = ijk / (norb * norb);
      const int j = (ijk % (norb * norb)) / norb;
      const int k = (ijk % (norb * norb)) % norb;
      for (int l = 0; l != norb; ++l) {
        for (int o = 0; o != norb; ++o) {
          for (int p = 0; p != norb; ++p) {
            const int mat[6] = {i, j, k, l, o, p};
            target[p + norb * (o + norb
              * (l + norb * (k + norb * (j + norb * i))))]
              += factor * source[mat[indices[5]]
              + norb * (mat[indices[4]] + norb * (mat[indices[3]]
              + norb * (mat[indices[2]] + norb * (mat[indices[1]]
              + norb * mat[indices[0]]))))];
          }
        }
      }
    }
  } else if (srank == 0 && trank == 4) {
#pragma omp parallel for schedule(static)
    for (int ijkl = 0; ijkl < norb * norb * norb * norb; ++ijkl) {
      const int i = ijkl / (norb * norb * norb);
      const int jkl = ijkl % (norb * norb * norb);
      const int j = jkl / (norb * norb);
      const int kl = jkl % (norb * norb);
      const int k = kl / norb;
      const int l = kl % norb;
      for (int o = 0; o != norb; ++o) {
        for (int p = 0; p != norb; ++p) {
          for (int q = 0; q != norb; ++q) {
            for (int r = 0; r != norb; ++r) {
              const int mat[8] = {i, j, k, l, o, p, q, r};
              if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]] &&
                  mat[delta[1 * 2 + 0]] == mat[delta[1 * 2 + 1]] &&
                  mat[delta[2 * 2 + 0]] == mat[delta[2 * 2 + 1]] &&
                  mat[delta[3 * 2 + 0]] == mat[delta[3 * 2 + 1]]) {
                target[r + norb * (q + norb * (p + norb * (o + norb
                  * (l + norb * (k + norb * (j + norb * i))))))]
                  += factor;
              }
            }
          }
        }
      }
    }
  } else if (srank == 1 && trank == 4) {
#pragma omp parallel for schedule(static)
    for (int ijkl = 0; ijkl < norb * norb * norb * norb; ++ijkl) {
      const int i = ijkl / (norb * norb * norb);
      const int jkl = ijkl % (norb * norb * norb);
      const int j = jkl / (norb * norb);
      const int kl = jkl % (norb * norb);
      const int k = kl / norb;
      const int l = kl % norb;
      for (int o = 0; o != norb; ++o) {
        for (int p = 0; p != norb; ++p) {
          for (int q = 0; q != norb; ++q) {
            for (int r = 0; r != norb; ++r) {
              const int mat[8] = {i, j, k, l, o, p, q, r};
              if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]] &&
                  mat[delta[1 * 2 + 0]] == mat[delta[1 * 2 + 1]] &&
                  mat[delta[2 * 2 + 0]] == mat[delta[2 * 2 + 1]]) {
                target[r + norb * (q + norb * (p + norb * (o + norb
                  * (l + norb * (k + norb * (j + norb * i))))))]
                  += factor * source[mat[indices[1]] + norb * mat[indices[0]]];
              }
            }
          }
        }
      }
    }
  } else if (srank == 2 && trank == 4) {
#pragma omp parallel for schedule(static)
    for (int ijkl = 0; ijkl < norb * norb * norb * norb; ++ijkl) {
      const int i = ijkl / (norb * norb * norb);
      const int jkl = ijkl % (norb * norb * norb);
      const int j = jkl / (norb * norb);
      const int kl = jkl % (norb * norb);
      const int k = kl / norb;
      const int l = kl % norb;
      for (int o = 0; o != norb; ++o) {
        for (int p = 0; p != norb; ++p) {
          for (int q = 0; q != norb; ++q) {
            for (int r = 0; r != norb; ++r) {
              const int mat[8] = {i, j, k, l, o, p, q, r};
              if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]] &&
                  mat[delta[1 * 2 + 0]] == mat[delta[1 * 2 + 1]]) {
                target[r + norb * (q + norb * (p + norb * (o + norb
                  * (l + norb * (k + norb * (j + norb * i))))))]
                  += factor * source[mat[indices[3]]
                  + norb * (mat[indices[2]]
                  + norb * (mat[indices[1]]
                  + norb * mat[indices[0]]))];
              }
            }
          }
        }
      }
    }
  } else if (srank == 3 && trank == 4) {
#pragma omp parallel for schedule(static)
    for (int ijkl = 0; ijkl < norb * norb * norb * norb; ++ijkl) {
      const int i = ijkl / (norb * norb * norb);
      const int jkl = ijkl % (norb * norb * norb);
      const int j = jkl / (norb * norb);
      const int kl = jkl % (norb * norb);
      const int k = kl / norb;
      const int l = kl % norb;
      for (int o = 0; o != norb; ++o) {
        for (int p = 0; p != norb; ++p) {
          for (int q = 0; q != norb; ++q) {
            for (int r = 0; r != norb; ++r) {
              const int mat[8] = {i, j, k, l, o, p, q, r};
              if (mat[delta[0 * 2 + 0]] == mat[delta[0 * 2 + 1]]) {
                target[r + norb * (q + norb * (p + norb * (o + norb
                  * (l + norb * (k + norb * (j + norb * i))))))]
                  += factor * source[mat[indices[5]]
                  + norb * (mat[indices[4]]
                  + norb * (mat[indices[3]]
                  + norb * (mat[indices[2]]
                  + norb * (mat[indices[1]]
                  + norb * mat[indices[0]]))))];
              }
            }
          }
        }
      }
    }
  } else if (srank == 4 && trank == 4) {
#pragma omp parallel for schedule(static)
    for (int ijkl = 0; ijkl < norb * norb * norb * norb; ++ijkl) {
      const int i = ijkl / (norb * norb * norb);
      const int jkl = ijkl % (norb * norb * norb);
      const int j = jkl / (norb * norb);
      const int kl = jkl % (norb * norb);
      const int k = kl / norb;
      const int l = kl % norb;
      for (int o = 0; o != norb; ++o) {
        for (int p = 0; p != norb; ++p) {
          for (int q = 0; q != norb; ++q) {
            for (int r = 0; r != norb; ++r) {
              const int mat[8] = {i, j, k, l, o, p, q, r};
              target[r + norb * (q + norb * (p + norb * (o + norb
                * (l + norb * (k + norb * (j + norb * i))))))]
                += factor * source[mat[indices[7]]
                + norb * (mat[indices[6]]
                + norb * (mat[indices[5]]
                + norb * (mat[indices[4]]
                + norb * (mat[indices[3]]
                + norb * (mat[indices[2]]
                + norb * (mat[indices[1]]
                + norb * mat[indices[0]]))))))];
            }
          }
        }
      }
    }
  } else {
    assert(false);
  }
  return 0;
}
