#ifndef POLESITTER_H
#define POLESITTER_H

#include <stdint.h>

#ifndef POLESITTER_MALLOC
#    include <stdlib.h>
#    define POLESITTER_MALLOC(sz) malloc(sz)
#    define POLESITTER_FREE(ptr)  free(ptr)
#endif

// =====================================================================
// arena allocator
// =====================================================================

typedef struct {
    uint8_t* mem;
    size_t   cap;
    size_t   off;
} ps_arena_t;

// init arena with pre-allocated/externally provided buffer
static void ps_arena_init(ps_arena_t* arena, void* buffer, size_t cap);

// alloc raw bytes with a specified align (16 for SIMD)
static void* ps_arena_alloc(ps_arena_t* arena, size_t size, size_t align);

// clear for next frame
static void ps_arena_clear(ps_arena_t* arena);

#endif // POLESITTER_H

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

// final number has 32 bits.
// diving it by 3 dimensions, we can store 10 bits per dimension.
// it interleaves the expanded bits of x, y and z.
static uint32_t ps__morton_encode(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = ps__expand_bits(x);
    uint32_t yy = ps__expand_bits(y);
    uint32_t zz = ps__expand_bits(z);

    // x takes bit 0, y shifts to bit 1, z shifts to bit 2
    return xx | (yy << 1) | (zz << 2);
}

// rounds 'v' up to the nearest multiple of 'align', which must be a power of 2
static inline size_t ps_align_forward(size_t v, size_t align) {
    return (v + (align - 1)) & ~(align - 1);
}

static void ps_arena_init(ps_arena_t* arena, void* buffer, size_t cap) {
    arena->mem = (uint8_t*)buffer;
    arena->cap = cap;
    arena->off = 0;
}

static void* ps_arena_alloc(ps_arena_t* arena, size_t size, size_t align) {
    // current aligned address offset
    size_t curr_addr  = (size_t)(arena->mem + arena->off);
    size_t align_addr = ps_align_forward(curr_addr, align);

    // how much extra padding was added to align the address
    size_t pad = align_addr - curr_addr;

    // check if we have enough space in the arena
    size_t total_req = size + pad;
    if (arena->off + total_req > arena->cap)
        return NULL;

    // advance bump pointer and return aligned address
    void* ptr = (void*)align_addr;
    arena->off += total_req;

    return ptr;
}

static void ps_arena_clear(ps_arena_t* arena) {
    arena->off = 0;
}

#endif // POLESITTER_IMPLEMENTATION
