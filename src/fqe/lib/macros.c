#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "macros.h"

#define MY_STRING_LEN 1024

static void print_status(void) {
#ifndef NDEBUG
  /* /proc/[pid]/status contains all human readible status, if i want to
   * get the memory later on and am only interested in that (e.g. to
   * write to a file every x seconds), it is better to read from
   * /proc/[pid]/statm (easier to parse) */
  FILE *status;
  char line[MY_STRING_LEN];
  if ((status = fopen("/proc/self/status", "r")) == NULL) {
    fprintf(stderr, "Error in openeing /proc/self/status\n");
    return;
  }
  while (fgets(line, sizeof line, status)) {
    fprintf(stderr, "%s", line);
  }
  fclose(status);
#endif
}

/* ========================================================================== */

void *safe_malloc_helper(long long s, size_t t, const char *typ,
                          const char *file, int line, const char *func) {
  void *pn = malloc(s * t);
  if (pn == NULL || s < 0) {
    const char * slashchar = strrchr(file, '/');
    const char * filname = slashchar == NULL ? file : slashchar + 1;

    fprintf(stderr, "%s:%d @%s :: Failed to allocate %s array of size %lld (%llu bytes)!\n"
      "Maximal size of size_t : %lu\n",
      filname == NULL ? file : filname, line, func,
      typ, s, s*t, SIZE_MAX);
    print_status();
    exit(EXIT_FAILURE);
  }
  return pn;
}

void *safe_calloc_helper(long long s, size_t t, const char *typ,
                         const char *file, int line, const char *func) {
  void *pn = calloc(s, t);
  if (pn == NULL || s < 0) {
    const char *filname = strrchr(file, '/') + 1;
    fprintf(stderr, "%s:%d @%s :: Failed to reallocate %s array of size %lld (%llu bytes)!\n"
      "Maximal size of size_t : %lu\n",
      filname == NULL ? file : filname, line, func,
      typ, s, s*t, SIZE_MAX);
    print_status();
    exit(EXIT_FAILURE);
  }
  return pn;
}
