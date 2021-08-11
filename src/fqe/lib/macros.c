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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "macros.h"

#define MY_STRING_LEN 1024

void *safe_malloc_helper(long long s, size_t t, const char *typ,
                          const char *file, int line, const char *func) {
  void *pn = malloc(s * t);
  if (pn == NULL || s < 0) {
    fprintf(stderr,
      "%s:%d @%s :: Failed to allocate %s array of size %lld (%llu bytes)!\n"
      "Maximal size of size_t : %lu\n",
      file, line, func, typ, s, s*t, SIZE_MAX);
    exit(EXIT_FAILURE);
  }
  return pn;
}

void *safe_calloc_helper(long long s, size_t t, const char *typ,
                         const char *file, int line, const char *func) {
  void *pn = calloc(s, t);
  if (pn == NULL || s < 0) {
    fprintf(stderr,
      "%s:%d @%s :: Failed to reallocate %s array of size %lld (%llu bytes)!\n"
      "Maximal size of size_t : %lu\n",
      file, line, func, typ, s, s*t, SIZE_MAX);
    exit(EXIT_FAILURE);
  }
  return pn;
}
