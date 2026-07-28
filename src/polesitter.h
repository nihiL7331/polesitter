#ifndef POLESITTER_H
#define POLESITTER_H

#include <stddef.h>
#include <stdint.h>

// =====================================================================
// public api
// =====================================================================

typedef enum {
    PS_OK     = 0,
    PS_EOOM   = -1, // out of memory
    PS_EINVAL = -2, // invalid argument
} ps_result_t;

typedef struct ps_context ps_context_t;

typedef struct {
    void*  buff;      // ram provided by host
    size_t buff_size; // total size in B
} ps_config_t;

// init pipeline with a pre-allocated buffer.
ps_result_t ps_init(ps_context_t** out_ctx, const ps_config_t* conf);

#endif // POLESITTER_H

#ifdef POLESITTER_IMPLEMENTATION

// =====================================================================
// internal implementation
// =====================================================================

// ---------------------------------------------------------------------
// arena allocator
// ---------------------------------------------------------------------

typedef struct {
    uint8_t* mem;
    size_t   cap;
    size_t   off;
} ps_arena_t;

struct ps_context {
    ps_arena_t arena;
};

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

// init arena with pre-allocated/externally provided buffer
static void ps_arena_init(ps_arena_t* arena, void* buffer, size_t cap) {
    arena->mem = (uint8_t*)buffer;
    arena->cap = cap;
    arena->off = 0;
}

// alloc raw bytes with a specified align (16 for SIMD)
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

// clear for next frame
static void ps_arena_clear(ps_arena_t* arena) {
    arena->off = 0;
}

// =====================================================================
// public api implementation
// =====================================================================

ps_result_t ps_init(ps_context_t** out_ctx, const ps_config_t* conf) {
    if (!conf || !conf->buff || conf->buff_size < sizeof(ps_context_t))
        return PS_EINVAL;

    // place ctx at beginning of buffer
    ps_context_t* ctx = (ps_context_t*)conf->buff;

    // arena takes the rest
    size_t arena_start = sizeof(ps_context_t);
    ps_arena_init(&ctx->arena, (uint8_t*)conf->buff + arena_start,
                  conf->buff_size - arena_start);

    *out_ctx = ctx;
    return PS_OK;
}

#endif // POLESITTER_IMPLEMENTATION
