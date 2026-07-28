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

typedef struct {
    float* x;
    float* y;
    float* z;
    float* mass;
    float* fx;
    float* fy;
    float* fz;
    size_t cnt;
} ps_particle_arrs_t;

// init pipeline with a pre-allocated buffer.
ps_result_t ps_init(ps_context_t** out_ctx, const ps_config_t* conf);

// compute forces on particles using FMM
ps_result_t ps_calc_forces(ps_context_t* ctx, const ps_particle_arrs_t* arrs,
                           float root_cx, float root_cy, float root_cz,
                           float root_hw);

#endif // POLESITTER_H

#define POLESITTER_IMPLEMENTATION
#ifdef POLESITTER_IMPLEMENTATION

// =====================================================================
// internal implementation
// =====================================================================

// ---------------------------------------------------------------------
// internal structs
// ---------------------------------------------------------------------

typedef struct {
    uint8_t* mem;
    size_t   cap;
    size_t   off;
} ps_arena_t;

// octree node
typedef struct ps_node {
    // physics, 16B
    float x;
    float y;
    float z;
    float half_width; // needed for M2L

    // FMM payload, 32B (monopole + dipole)
    float multipole[4];
    float local[4];

    // structure, 64B
    struct ps_node* children[8];

    // leaf metadata, 16B
    uint32_t is_leaf;
    uint32_t particle_cnt;
    uint32_t first_particle_idx;
    uint32_t _pad;
} ps_node_t;

struct ps_context {
    ps_arena_t arena;
    ps_node_t* root;
};

// ---------------------------------------------------------------------
// internal functions
// ---------------------------------------------------------------------

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
    if (arena->off + total_req > arena->cap) {
        return NULL;
    }

    // advance bump pointer and return aligned address
    void* ptr = (void*)align_addr;
    arena->off += total_req;

    return ptr;
}

// clear for next frame
static void ps_arena_clear(ps_arena_t* arena) {
    arena->off = 0;
}

#define PS_MAX_DEPTH                                                           \
    10 // max depth of octree, 10 levels = 1024 (2^10) leaf nodes

// zero out a newly allocated node
static void ps__node_init(ps_node_t* node) {
    node->x          = 0.0F;
    node->y          = 0.0F;
    node->z          = 0.0F;
    node->half_width = 0.0F;

    for (int i = 0; i < 4; ++i) {
        node->multipole[i] = 0.0F;
        node->local[i]     = 0.0F;
    }

    for (int i = 0; i < 8; ++i) {
        node->children[i] = NULL;
    }

    node->is_leaf            = 0;
    node->particle_cnt       = 0;
    node->first_particle_idx = 0;
}

// walk the morton code and build the tree branches
static ps_result_t ps__tree_insert(ps_arena_t* arena, ps_node_t* root,
                                   uint32_t morton_code,
                                   uint32_t particle_idx) {
    if (!root) {
        return PS_EINVAL;
    }

    ps_node_t* curr = root;

    float curr_x  = root->x;
    float curr_y  = root->y;
    float curr_z  = root->z;
    float curr_hw = root->half_width;

    for (int depth = 0; depth < PS_MAX_DEPTH; ++depth) {
        // shift starts at 27, decreases by 3 each lvl
        int      shift  = 27 - (depth * 3);
        uint32_t octant = (morton_code >> shift) & 0x7; // 3 bits for octant

        curr_hw *= 0.5F;

        curr_x += (octant & 0x1 /* bit 0 */) ? curr_hw : -curr_hw;
        curr_y += (octant & 0x2 /* bit 1 */) ? curr_hw : -curr_hw;
        curr_z += (octant & 0x4 /* bit 2 */) ? curr_hw : -curr_hw;

        if (!curr->children[octant]) {
            ps_node_t* new_node =
                (ps_node_t*)ps_arena_alloc(arena, sizeof(ps_node_t), 16);
            if (!new_node) {
                return PS_EOOM;
            }

            ps__node_init(new_node);

            new_node->x          = curr_x;
            new_node->y          = curr_y;
            new_node->z          = curr_z;
            new_node->half_width = curr_hw;

            curr->children[octant] = new_node;
        }

        curr = curr->children[octant];
    }

    curr->is_leaf = 1;

    if (curr->particle_cnt == 0) {
        curr->first_particle_idx = particle_idx;
    }

    curr->particle_cnt++;

    return PS_OK;
}

// p2m (leaves) and m2m (parents) in post-order traversal
static void ps__fmm_upward_pass(ps_node_t*                node,
                                const ps_particle_arrs_t* arrs) {
    if (!node) {
        return;
    }

    // p2m
    if (node->is_leaf) {
        float m0 = 0.0F; // monopole (total mass)
        float mx = 0.0F; // dipole x (mass moment)
        float my = 0.0F; // dipole y
        float mz = 0.0F; // dipole z

        for (uint32_t i = 0; i < node->particle_cnt; ++i) {
            uint32_t idx  = node->first_particle_idx + i;
            float    mass = arrs->mass[idx];

            // dist from particle pos to center of this voxel
            float dx = arrs->x[idx] - node->x;
            float dy = arrs->y[idx] - node->y;
            float dz = arrs->z[idx] - node->z;

            m0 += mass;
            mx += mass * dx;
            my += mass * dy;
            mz += mass * dz;
        }

        // store expansion payload
        node->multipole[0] = m0;
        node->multipole[1] = mx;
        node->multipole[2] = my;
        node->multipole[3] = mz;
        return;
    }

    // m2m
    float p_m0 = 0.0F;
    float p_mx = 0.0F;
    float p_my = 0.0F;
    float p_mz = 0.0F;

    for (int i = 0; i < 8; ++i) {
        ps_node_t* child = node->children[i];
        if (child) {
            // calc the children first
            ps__fmm_upward_pass(child, arrs);

            // dist vector from the child's center
            // up to the parent's center
            float dx = child->x - node->x;
            float dy = child->y - node->y;
            float dz = child->z - node->z;

            float c_m0 = child->multipole[0];
            float c_mx = child->multipole[1];
            float c_my = child->multipole[2];
            float c_mz = child->multipole[3];

            // shift the childs expansion to the parents center and accumulate
            // dipole requires the monopole * dist
            p_m0 += c_m0;
            p_mx += c_mx + (c_m0 * dx);
            p_my += c_my + (c_m0 * dy);
            p_mz += c_mz + (c_m0 * dz);
        }
    }

    // store aggregated expansion in parent
    node->multipole[0] = p_m0;
    node->multipole[1] = p_mx;
    node->multipole[2] = p_my;
    node->multipole[3] = p_mz;
}

#define PS__SWAP_PTR(type, a, b)                                               \
    do {                                                                       \
        type tmp = a;                                                          \
        (a)      = b;                                                          \
        (b)      = tmp;                                                        \
    } while (0)

static ps_result_t ps__sort_particles(ps_arena_t* arena, uint32_t* morton_codes,
                                      const ps_particle_arrs_t* arrs) {
    size_t cnt = arrs->cnt;
    if (cnt == 0) {
        return PS_OK;
    }

    // borrow temp SoA buffers from the arena
    uint32_t* m_tmp =
        (uint32_t*)ps_arena_alloc(arena, cnt * sizeof(uint32_t), 16);
    float* x_tmp    = (float*)ps_arena_alloc(arena, cnt * sizeof(float), 16);
    float* y_tmp    = (float*)ps_arena_alloc(arena, cnt * sizeof(float), 16);
    float* z_tmp    = (float*)ps_arena_alloc(arena, cnt * sizeof(float), 16);
    float* mass_tmp = (float*)ps_arena_alloc(arena, cnt * sizeof(float), 16);

    if (!m_tmp || !x_tmp || !y_tmp || !z_tmp || !mass_tmp) {
        return PS_EOOM;
    }

    uint32_t* m_src    = morton_codes;
    uint32_t* m_dst    = m_tmp;
    float*    x_src    = arrs->x;
    float*    x_dst    = x_tmp;
    float*    y_src    = arrs->y;
    float*    y_dst    = y_tmp;
    float*    z_src    = arrs->z;
    float*    z_dst    = z_tmp;
    float*    mass_src = arrs->mass;
    float*    mass_dst = mass_tmp;

    // 4 passes of 8b radix sort
    for (int pass = 0; pass < 4; ++pass) {
        uint32_t counts[256]  = {0};
        uint32_t offsets[256] = {0};
        int      shift        = pass * 8;

        // count frequencies of current 8b chunk
        for (size_t i = 0; i < cnt; ++i) {
            uint8_t bucket = (m_src[i] >> shift) & 0xFF;
            counts[bucket]++;
        }

        // calc prefix sums to find the start off per bucket
        offsets[0] = 0;
        for (int i = 0; i < 256; ++i) {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }

        // distribute the data into dest buffers
        for (size_t i = 0; i < cnt; ++i) {
            uint8_t  bucket  = (m_src[i] >> shift) & 0xFF;
            uint32_t dst_idx = offsets[bucket]++;

            m_dst[dst_idx]    = m_src[i];
            x_dst[dst_idx]    = x_src[i];
            y_dst[dst_idx]    = y_src[i];
            z_dst[dst_idx]    = z_src[i];
            mass_dst[dst_idx] = mass_src[i];
        }

        // pingpong pointers for next pass
        PS__SWAP_PTR(uint32_t*, m_src, m_dst);
        PS__SWAP_PTR(float*, x_src, x_dst);
        PS__SWAP_PTR(float*, y_src, y_dst);
        PS__SWAP_PTR(float*, z_src, z_dst);
        PS__SWAP_PTR(float*, mass_src, mass_dst);
    }

    // 4 is an even number, so m_src is guaranteed to be pointing back to the
    // orig morton_codes arr, and x_src back to arrs->x. the sorted data is
    // right where it started. temp arrays will be cleared next frame.

    return PS_OK;
}

// =====================================================================
// public api implementation
// =====================================================================

ps_result_t ps_init(ps_context_t** out_ctx, const ps_config_t* conf) {
    if (!conf || !conf->buff || conf->buff_size < sizeof(ps_context_t)) {
        return PS_EINVAL;
    }

    // place ctx at beginning of buffer
    ps_context_t* ctx = (ps_context_t*)conf->buff;
    ctx->root         = NULL;

    // arena takes the rest
    size_t arena_start = ps_align_forward(sizeof(ps_context_t), 16);
    ps_arena_init(&ctx->arena, (uint8_t*)conf->buff + arena_start,
                  conf->buff_size - arena_start);

    *out_ctx = ctx;
    return PS_OK;
}

ps_result_t ps_calc_forces(ps_context_t* ctx, const ps_particle_arrs_t* arrs,
                           float root_cx, float root_cy, float root_cz,
                           float root_hw) {
    if (!ctx || !arrs) {
        return PS_EINVAL;
    }

    // clear arena for new frame
    ps_arena_clear(&ctx->arena);

    // borrow scratch memory from arena for morton codes
    uint32_t* morton_codes = (uint32_t*)ps_arena_alloc(
        &ctx->arena, arrs->cnt * sizeof(uint32_t), 16);
    if (!morton_codes) {
        return PS_EOOM;
    }

    // precalculate morton codes
    for (size_t i = 0; i < arrs->cnt; ++i) {
        // normalize particle position to [0, 1] based on root node
        float nx = (arrs->x[i] - (root_cx - root_hw)) / (2.0F * root_hw);
        float ny = (arrs->y[i] - (root_cy - root_hw)) / (2.0F * root_hw);
        float nz = (arrs->z[i] - (root_cz - root_hw)) / (2.0F * root_hw);

        // clamp to [0, 1]
        if (nx < 0.0F) {
            nx = 0.0F;
        }
        if (ny < 0.0F) {
            ny = 0.0F;
        }
        if (nz < 0.0F) {
            nz = 0.0F;
        }
        if (nx > 1.0F) {
            nx = 1.0F;
        }
        if (ny > 1.0F) {
            ny = 1.0F;
        }
        if (nz > 1.0F) {
            nz = 1.0F;
        }

        // scale to [0, 1023] for morton encoding
        uint32_t ix = (uint32_t)(nx * 1023.0F);
        uint32_t iy = (uint32_t)(ny * 1023.0F);
        uint32_t iz = (uint32_t)(nz * 1023.0F);

        morton_codes[i] = ps__morton_encode(ix, iy, iz);
    }

    // sort the arrays based on morton_codes as keys
    ps__sort_particles(&ctx->arena, morton_codes, arrs);

    // allocate the root node
    ctx->root = (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    if (!ctx->root) {
        return PS_EOOM;
    }

    ps__node_init(ctx->root);

    // seed geometry
    ctx->root->x          = root_cx;
    ctx->root->y          = root_cy;
    ctx->root->z          = root_cz;
    ctx->root->half_width = root_hw;

    // build the octree
    for (size_t i = 0; i < arrs->cnt; ++i) {
        ps__tree_insert(&ctx->arena, ctx->root, morton_codes[i], (uint32_t)i);
    }

    // upward pass (p2m -> m2m)
    ps__fmm_upward_pass(ctx->root, arrs);

    // TODO: passes come here

    return PS_OK;
}

#endif // POLESITTER_IMPLEMENTATION
