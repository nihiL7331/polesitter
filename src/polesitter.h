#ifndef POLESITTER_H
#define POLESITTER_H

#ifndef POLESITTER_MALLOC
#include <stdlib.h>
#define POLESITTER_MALLOC(sz) malloc(sz)
#define POLESITTER_FREE(ptr) free(ptr)
#endif

#endif

#ifdef POLESITTER_IMPLEMENTATION

static uint32_t ps__morton_encode(uint32_t x, uint32_t y, uint32_t z) {
    (void)x;
    (void)y;
    (void)z;
    return 0;
}

#endif // POLESITTER_IMPLEMENTATION
