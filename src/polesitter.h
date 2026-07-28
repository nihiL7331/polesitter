#ifndef POLESITTER_H
#define POLESITTER_H

#ifndef POLESITTER_MALLOC
#include <stdlib.h>
#define POLESITTER_MALLOC(sz) malloc(sz)
#define POLESITTER_FREE(ptr) free(ptr)
#endif

#endif

#ifdef POLESITTER_IMPLEMENTATION

// take a 10b num and expand it to 30b by inserting 2 0s between each b.
static uint32_t ps__expand_bits(uint32_t v) {
    v &= 0x000003FF; // only look at 10 ls bits

    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;

    return v;
}

static uint32_t ps__morton_encode(uint32_t x, uint32_t y, uint32_t z) {
    (void)x;
    (void)y;
    (void)z;
    return 0;
}

#endif // POLESITTER_IMPLEMENTATION
