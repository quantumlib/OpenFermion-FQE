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

#ifndef SRC_FQE_LIB_MACROS_H_
#define SRC_FQE_LIB_MACROS_H_

#include <stdlib.h>

#define safe_malloc(t, s) safe_malloc_helper((s), sizeof *(t), #t, \
    __FILE__, __LINE__, __func__)
#define safe_calloc(t, s) safe_calloc_helper((s), sizeof *(t), #t, \
    __FILE__, __LINE__, __func__)

void * safe_malloc_helper(long long s, size_t t, const char *typ,
                          const char *file, int line, const char *func);

void * safe_calloc_helper(long long s, size_t t, const char *typ,
                          const char *file, int line, const char *func);

// maximal number of orbitals that the C extension supports
#define MAX_ORBS sizeof(uint64_t)*8

// the number of elements in a batch when computing ZAXPY
#define ZAXPY_STRIDE 450

#endif  // SRC_FQE_LIB_MACROS_H_
