#pragma once

#include <stdlib.h>

#define safe_malloc(t, s) safe_malloc_helper((s), sizeof *(t), #t, __FILE__, __LINE__, __func__)
#define safe_calloc(t, s) safe_calloc_helper((s), sizeof *(t), #t, __FILE__, __LINE__, __func__)

void * safe_malloc_helper(long long s, size_t t, const char *typ,
                          const char *file, int line, const char *func);

void * safe_calloc_helper(long long s, size_t t, const char *typ,
                          const char *file, int line, const char *func);
