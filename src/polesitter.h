// clang-format off
/*-
    polesitter - v1.0 - Fast Multipole Method (FFM) N-body solver written in C99

    Do this:
        #define POLESITTER_IMPLEMENTATION
    before you include this file in *one* C or C++ file to create the
   implementation.

    // i.e. it should look like this:
    #include ...
    #include ...
    #include ...
    #define POLESITTER_IMPLEMENTATION
    #include "polesitter.h"

    QUICK NOTES:
        Primarily of interest to game devs and graphics programmers.

        Requires bringing your own memory buffer.
        No malloc/free calls are made internally.

    LICENSE
        See end of file for license information.

    QUICKSTART

        ```
        #define POLESITTER_IMPLEMENTATION
        #include "polesitter.h"
        #include <stdint.h>
        #include <stdlib.h>
        #include <stdbool.h>

        #define MEMORY_SIZE 1024ULL * 1024 * 16 // 16 MB
        #define PARTICLE_CNT 1024

        int main(void) {
            // provide a raw memory block

            void* memory_block = malloc(MEMORY_SIZE);

            // initialize the context

            ps_config_t cfg = { memory_block, MEMORY_SIZE };
            ps_context_t* ctx = NULL;
            ps_init(&ctx, &cfg);

            // SoA particle arrays

            float x[PARTICLE_CNT];
            float y[PARTICLE_CNT];
            float z[PARTICLE_CNT];
            float mass[PARTICLE_CNT];
            float fx[PARTICLE_CNT] = {0};
            float fy[PARTICLE_CNT] = {0};
            float fz[PARTICLE_CNT] = {0};

            // external arrays not managed by the solver

            float vx[PARTICLE_CNT] = {0};
            float vy[PARTICLE_CNT] = {0};
            float vz[PARTICLE_CNT] = {0};
            uint32_t morton_codes[PARTICLE_CNT];
            uint32_t ids[PARTICLE_CNT];

            // initialize positions and masses
            // ...

            ps_particle_arrs_t arrs = {
                x, y, z, mass,
                fx, fy, fz,
                ids, PARTICLE_CNT
            };

            float dt = 0.016F; // 60FPS
            bool running = true;
            while (running) {
                // reset arena and force accumulators for the new frame

                ps_arena_clear(&ctx->arena);
                for (int i = 0; i < PARTICLE_CNT; ++i) {
                    ids[i] = i; // reset ids before sorting
                    fx[i] = 0.0F; fy[i] = 0.0F; fz[i] = 0.0F;
                }

                // calculate the global bounding box, sort arrays via morton

                float min_b, max_b, range;
                ps_prepare_particles(
                    &arrs,
                    morton_codes,
                    &min_b,
                    &max_b,
                    &range
                );

                // run the physics tick
                // it builds the octree, computes mps and evals forces

                float root_c  = min_b + (range / 2.0F);
                float root_hw = range / 2.0F;
                ps_calc_forces(
                    ctx,
                    &arrs,
                    morton_codes,
                    root_c,
                    root_c,
                    root_c,
                    root_hw
                );

                // shuffle extern vel arrs to match the newly sorted pos

                float tmp_vx[PARTICLE_CNT];
                float tmp_vy[PARTICLE_CNT];
                float tmp_vz[PARTICLE_CNT];
                for (int i = 0; i < PARTICLE_CNT; ++i) {
                    uint32_t old_idx = ids[i];
                    tmp_vx[i] = vx[old_idx];
                    tmp_vy[i] = vy[old_idx];
                    tmp_vz[i] = vz[old_idx];
                }
                for (int i = 0; i < PARTICLE_CNT; ++i) {
                    vx[i] = tmp_vx[i];
                    vy[i] = tmp_vy[i];
                    vz[i] = tmp_vz[i];
                }

                // integrate velocities/positions externally,
                // polesitter doesn't handle that
                // e.g. via semi-implicit euler:

                for (int i = 0; i < PARTICLE_CNT; ++i) {
                    if (mass[i] <= 0.0F) {
                        continue;
                    }

                    vx[i] += (fx[i] / mass[i]) * dt;
                    vy[i] += (fy[i] / mass[i]) * dt;
                    vz[i] += (fz[i] / mass[i]) * dt;

                    x[i] += vx[i] * dt;
                    y[i] += vy[i] * dt;
                    z[i] += vz[i] * dt;
                }
            }

            // cleanup
            free(memory_block);
            return 0;
        }
        ```
*/
// clang-format on

#ifndef POLESITTER_H
#define POLESITTER_H

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__AVX2__)

#include <immintrin.h>
#define PS_USE_AVX2

#elif defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#define PS_USE_NEON

#endif // __aarch64__ || _M_ARM64

#ifndef PS_RESTRICT

#if defined(__cplusplus) || defined(_MSC_VER)
#define PS_RESTRICT __restrict

#else

#define PS_RESTRICT restrict

#endif

#endif // PS_RESTRICT

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
    float*    x;
    float*    y;
    float*    z;
    float*    mass;
    float*    fx;
    float*    fy;
    float*    fz;
    uint32_t* id;
    size_t    cnt;
} ps_particle_arrs_t;

// init pipeline with a pre-allocated buffer.
ps_result_t ps_init(ps_context_t** out_ctx, const ps_config_t* conf);

// compute forces on particles using FMM
ps_result_t ps_calc_forces(ps_context_t* ctx, const ps_particle_arrs_t* arrs,
                           uint32_t* morton_codes, float root_cx, float root_cy,
                           float root_cz, float root_hw);

// calculates the global bounding box and generates morton codes for all
// particles
ps_result_t ps_prepare_particles(ps_particle_arrs_t* arrs,
                                 uint32_t* out_morton_codes, float* out_min_b,
                                 float* out_max_b, float* out_range);

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
static uint32_t ps_impl_expand_bits(uint32_t v) {
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
static uint32_t ps_impl_morton_encode(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = ps_impl_expand_bits(x);
    uint32_t yy = ps_impl_expand_bits(y);
    uint32_t zz = ps_impl_expand_bits(z);

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
static void ps_impl_node_init(ps_node_t* node) {
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
static ps_result_t ps_impl_tree_insert(ps_arena_t* arena, ps_node_t* root,
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
                (ps_node_t*)ps_arena_alloc(arena, sizeof(ps_node_t), 32);
            if (!new_node) {
                return PS_EOOM;
            }

            ps_impl_node_init(new_node);

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

// pass 1
// p2m (leaves) and m2m (parents) in post-order traversal
static void ps_impl_fmm_upward_pass(ps_node_t*                node,
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
            ps_impl_fmm_upward_pass(child, arrs);

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

// pass 2
// m2l dual-tree traversal to find well-separated nodes and translate expansions
static void ps_impl_fmm_interaction_pass(ps_node_t* target, ps_node_t* src) {
    if (!target || !src) {
        return;
    }

    float dx      = src->x - target->x;
    float dy      = src->y - target->y;
    float dz      = src->z - target->z;
    float dist_sq = (dx * dx) + (dy * dy) + (dz * dz);

    // multipole acceptance criterion
    // if dist > sum of half-widths, they are well-separated
    // theta dictates accuracy vs speed
    float theta  = 1.0F;
    float hw_sum = target->half_width + src->half_width;

    // well-separated
    if (dist_sq > (theta * theta) * (hw_sum * hw_sum)) {
        float dist   = sqrtf(dist_sq);
        float inv_r3 = 1.0F / (dist * dist_sq);
        float inv_r5 = inv_r3 / dist_sq;

        float m0 = src->multipole[0];
        float mx = src->multipole[1];
        float my = src->multipole[2];
        float mz = src->multipole[3];

        // monopole contribution to target's local field
        float force_m0_x = m0 * dx * inv_r3;
        float force_m0_y = m0 * dy * inv_r3;
        float force_m0_z = m0 * dz * inv_r3;

        // dipole contribution to target's local field
        float m_dot_r      = (mx * dx) + (my * dy) + (mz * dz);
        float dipole_coeff = 3.0F * m_dot_r * inv_r5;

        float force_dip_x = (dx * dipole_coeff) - (mx * inv_r3);
        float force_dip_y = (dy * dipole_coeff) - (my * inv_r3);
        float force_dip_z = (dz * dipole_coeff) - (mz * inv_r3);

        // accumulate into target's local expansion
        target->local[1] += force_m0_x + force_dip_x;
        target->local[2] += force_m0_y + force_dip_y;
        target->local[3] += force_m0_z + force_dip_z;

        return;
    }

    // not well-separated
    if (target->is_leaf && src->is_leaf) {
        // both are leaves, near-field p2p pass will handle exact dists
        return;
    }

    if (target->is_leaf) {
        // target is as small as possible, open the src
        for (int i = 0; i < 8; ++i) {
            ps_impl_fmm_interaction_pass(target, src->children[i]);
        }
    } else if (src->is_leaf) {
        // src is as small as possible, open the target
        for (int i = 0; i < 8; ++i) {
            ps_impl_fmm_interaction_pass(target->children[i], src);
        }
    } else {
        // subdivide both and pair all 64 permutations
        for (int i = 0; i < 8; ++i) {
            if (!target->children[i]) {
                continue;
            }

            for (int j = 0; j < 8; ++j) {
                if (!src->children[j]) {
                    continue;
                }

                ps_impl_fmm_interaction_pass(target->children[i],
                                             src->children[j]);
            }
        }
    }
}

// pass 3
// l2l, pushes the accumulated background field from parents down to their
// children
static void ps_impl_fmm_downward_pass(ps_node_t* node) {
    if (!node || node->is_leaf) {
        return;
    }

    for (int i = 0; i < 8; ++i) {
        ps_node_t* child = node->children[i];
        if (child) {
            // 1-st order local expansion,
            // shifting it is adding the parent's field to the child's field
            child->local[1] += node->local[1];
            child->local[2] += node->local[2];
            child->local[3] += node->local[3];

            ps_impl_fmm_downward_pass(child);
        }
    }
}

// pass 4
// l2p, applies the accumulated bg field to the particles inside the leaf
static void ps_impl_fmm_l2p_pass(ps_node_t*                node,
                                 const ps_particle_arrs_t* arrs) {
    if (!node) {
        return;
    }

    if (node->is_leaf) {
        float field_x = node->local[1];
        float field_y = node->local[2];
        float field_z = node->local[3];

        uint32_t i = 0;

#if defined(PS_USE_AVX2)
        // broadcast local bg field to all 8 lanes
        __m256 f_x_vec = _mm256_set1_ps(field_x);
        __m256 f_y_vec = _mm256_set1_ps(field_y);
        __m256 f_z_vec = _mm256_set1_ps(field_z);

        // process in chunks of 8
        for (; i + 7 < node->particle_cnt; i += 8) {
            uint32_t idx = node->first_particle_idx + i;

            // load 8 masses
            __m256 m_vec = _mm256_loadu_ps(&arrs->mass[idx]);

            // load 8 current forces
            __m256 cur_fx = _mm256_loadu_ps(&arrs->fx[idx]);
            __m256 cur_fy = _mm256_loadu_ps(&arrs->fy[idx]);
            __m256 cur_fz = _mm256_loadu_ps(&arrs->fz[idx]);

            // F_xyz += mass * field_xyz
            cur_fx = _mm256_add_ps(cur_fx, _mm256_mul_ps(m_vec, f_x_vec));
            cur_fy = _mm256_add_ps(cur_fy, _mm256_mul_ps(m_vec, f_y_vec));
            cur_fz = _mm256_add_ps(cur_fz, _mm256_mul_ps(m_vec, f_z_vec));

            // store 8 updated forces back to mem
            _mm256_store_ps(&arrs->fx[idx], cur_fx);
            _mm256_store_ps(&arrs->fy[idx], cur_fy);
            _mm256_store_ps(&arrs->fz[idx], cur_fz);
        }
#elif defined(PS_USE_NEON)
        // broadcast local bg field to all 4 lanes
        float32x4_t f_x_vec = vdupq_n_f32(field_x);
        float32x4_t f_y_vec = vdupq_n_f32(field_y);
        float32x4_t f_z_vec = vdupq_n_f32(field_z);

        // process in chunks of 4
        for (; i + 3 < node->particle_cnt; i += 4) {
            uint32_t idx = node->first_particle_idx + i;

            // load 4 masses
            float32x4_t m_vec = vld1q_f32(&arrs->mass[idx]);

            // load 4 current forces
            float32x4_t cur_fx = vld1q_f32(&arrs->fx[idx]);
            float32x4_t cur_fy = vld1q_f32(&arrs->fy[idx]);
            float32x4_t cur_fz = vld1q_f32(&arrs->fz[idx]);

            // F_xyz += mass * field_xyz
            cur_fx = vmlaq_f32(cur_fx, m_vec, f_x_vec);
            cur_fy = vmlaq_f32(cur_fy, m_vec, f_y_vec);
            cur_fz = vmlaq_f32(cur_fz, m_vec, f_z_vec);

            // store 4 updated forces back to mem
            vst1q_f32(&arrs->fx[idx], cur_fx);
            vst1q_f32(&arrs->fy[idx], cur_fy);
            vst1q_f32(&arrs->fz[idx], cur_fz);
        }
#endif

        // fallback for the remainder
        for (; i < node->particle_cnt; ++i) {
            uint32_t idx  = node->first_particle_idx + i;
            float    mass = arrs->mass[idx];

            // F = m * a
            arrs->fx[idx] += mass * field_x;
            arrs->fy[idx] += mass * field_y;
            arrs->fz[idx] += mass * field_z;
        }

        return;
    }

    // cascade
    for (int i = 0; i < 8; ++i) {
        ps_impl_fmm_l2p_pass(node->children[i], arrs);
    }
}

// pass 5
// p2p, dual-tree traversal to calc N-body forces for near-field neighbors
static void ps_impl_fmm_p2p_pass(ps_node_t* target, ps_node_t* src,
                                 const ps_particle_arrs_t* arrs) {
    if (!target || !src) {
        return;
    }

    float dx      = src->x - target->x;
    float dy      = src->y - target->y;
    float dz      = src->z - target->z;
    float dist_sq = (dx * dx) + (dy * dy) + (dz * dz);

    float theta  = 1.0F;
    float hw_sum = target->half_width + src->half_width;

    // if well-separated M2L already handled it
    if (dist_sq > (theta * theta) * (hw_sum * hw_sum)) {
        return;
    }

    // if both are leaves and too close
    // do direct N-body force
    if (target->is_leaf && src->is_leaf) {
        for (uint32_t i = 0; i < target->particle_cnt; ++i) {
            uint32_t t_idx  = target->first_particle_idx + i;
            float    t_x    = arrs->x[t_idx];
            float    t_y    = arrs->y[t_idx];
            float    t_z    = arrs->z[t_idx];
            float    t_mass = arrs->mass[t_idx];

            float f_x = 0.0F;
            float f_y = 0.0F;
            float f_z = 0.0F;

            const float* PS_RESTRICT sx = arrs->x;
            const float* PS_RESTRICT sy = arrs->y;
            const float* PS_RESTRICT sz = arrs->z;
            const float* PS_RESTRICT sm = arrs->mass;

            uint32_t j = 0;
#if defined(PS_USE_AVX2)
            // broadcast target coords and soft param
            __m256 t_x_vec = _mm256_set1_ps(t_x);
            __m256 t_y_vec = _mm256_set1_ps(t_y);
            __m256 t_z_vec = _mm256_set1_ps(t_z);
            __m256 eps_vec = _mm256_set1_ps(0.1F);

            // accumulators for target particles forces
            __m256 f_x_vec = _mm256_setzero_ps();
            __m256 f_y_vec = _mm256_setzero_ps();
            __m256 f_z_vec = _mm256_setzero_ps();

            // process in chunks of 8
            for (; j + 7 < src->particle_cnt; j += 8) {
                uint32_t s_idx = src->first_particle_idx + j;

                // load 8 source coordinates and masses
                __m256 s_x_vec = _mm256_loadu_ps(&sx[s_idx]);
                __m256 s_y_vec = _mm256_loadu_ps(&sy[s_idx]);
                __m256 s_z_vec = _mm256_loadu_ps(&sz[s_idx]);
                __m256 s_m_vec = _mm256_loadu_ps(&sm[s_idx]);

                // calculate distance vectors
                __m256 p_dx = _mm256_sub_ps(s_x_vec, t_x_vec);
                __m256 p_dy = _mm256_sub_ps(s_y_vec, t_y_vec);
                __m256 p_dz = _mm256_sub_ps(s_z_vec, t_z_vec);

                // p_dist_sq = dx*dx + dy*dy + dz*dz + eps
                __m256 dist_sq = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(p_dx, p_dx),
                                  _mm256_mul_ps(p_dy, p_dy)),
                    _mm256_add_ps(_mm256_mul_ps(p_dz, p_dz), eps_vec));

                // inv sqrt 1.0F / sqrt(dist_sq)
                __m256 inv_dist = _mm256_rsqrt_ps(dist_sq);

                // inv_dist3 = inv_dist^3
                __m256 inv_dist3 =
                    _mm256_mul_ps(_mm256_mul_ps(inv_dist, inv_dist), inv_dist);

                // F_m = mass * inv_dist3
                __m256 force = _mm256_mul_ps(s_m_vec, inv_dist3);

                // accumulate force components
                f_x_vec = _mm256_add_ps(f_x_vec, _mm256_mul_ps(p_dx, force));
                f_y_vec = _mm256_add_ps(f_y_vec, _mm256_mul_ps(p_dy, force));
                f_z_vec = _mm256_add_ps(f_z_vec, _mm256_mul_ps(p_dz, force));
            }

            // dump the 8 lanes back to memory and sum them up
            float temp_fx[8], temp_fy[8], temp_fz[8];
            _mm256_storeu_ps(temp_fx, f_x_vec);
            _mm256_storeu_ps(temp_fy, f_y_vec);
            _mm256_storeu_ps(temp_fz, f_z_vec);

            for (int lane = 0; lane < 8; ++lane) {
                f_x += temp_fx[lane];
                f_y += temp_fy[lane];
                f_z += temp_fz[lane];
            }

#elif defined(PS_USE_NEON)
            // broadcast target coords and soft param
            float32x4_t t_x_vec = vdupq_n_f32(t_x);
            float32x4_t t_y_vec = vdupq_n_f32(t_y);
            float32x4_t t_z_vec = vdupq_n_f32(t_z);
            float32x4_t eps_vec = vdupq_n_f32(2.0F);

            // accumulators for target particles forces
            float32x4_t f_x_vec = vdupq_n_f32(0.0F);
            float32x4_t f_y_vec = vdupq_n_f32(0.0F);
            float32x4_t f_z_vec = vdupq_n_f32(0.0F);

            // process in chunks of 4
            for (; j + 3 < src->particle_cnt; j += 4) {
                uint32_t s_idx = src->first_particle_idx + j;

                // load 4 source coordinates and masses
                float32x4_t s_x_vec = vld1q_f32(&sx[s_idx]);
                float32x4_t s_y_vec = vld1q_f32(&sy[s_idx]);
                float32x4_t s_z_vec = vld1q_f32(&sz[s_idx]);
                float32x4_t s_m_vec = vld1q_f32(&sm[s_idx]);

                // calculate distance vectors
                float32x4_t p_dx = vsubq_f32(s_x_vec, t_x_vec);
                float32x4_t p_dy = vsubq_f32(s_y_vec, t_y_vec);
                float32x4_t p_dz = vsubq_f32(s_z_vec, t_z_vec);

                // p_dist_sq = dx*dx + dy*dy + dz*dz + eps
                float32x4_t dist_sq = vmlaq_f32(eps_vec, p_dx, p_dx);
                dist_sq             = vmlaq_f32(dist_sq, p_dy, p_dy);
                dist_sq             = vmlaq_f32(dist_sq, p_dz, p_dz);

                // inv sqrt 1.0F / sqrt(dist_sq)
                // estimate + newton iteration
                float32x4_t inv_dist_est = vrsqrteq_f32(dist_sq);
                float32x4_t nr_step      = vrsqrtsq_f32(
                    dist_sq, vmulq_f32(inv_dist_est, inv_dist_est));
                float32x4_t inv_dist = vmulq_f32(inv_dist_est, nr_step);

                // inv_dist3 = inv_dist^3
                float32x4_t inv_dist3 =
                    vmulq_f32(inv_dist, vmulq_f32(inv_dist, inv_dist));

                // F_m = mass * inv_dist3
                float32x4_t force = vmulq_f32(s_m_vec, inv_dist3);

                // accumulate force components
                f_x_vec = vmlaq_f32(f_x_vec, p_dx, force);
                f_y_vec = vmlaq_f32(f_y_vec, p_dy, force);
                f_z_vec = vmlaq_f32(f_z_vec, p_dz, force);
            }

            // dump the 4 lanes back to memory and sum them up
            float temp_fx[4], temp_fy[4], temp_fz[4];
            vst1q_f32(temp_fx, f_x_vec);
            vst1q_f32(temp_fy, f_y_vec);
            vst1q_f32(temp_fz, f_z_vec);

            for (int lane = 0; lane < 4; ++lane) {
                f_x += temp_fx[lane];
                f_y += temp_fy[lane];
                f_z += temp_fz[lane];
            }
#endif
            // fallback for the remainder (previous operations left n in mod 8
            // particles)
            for (; j < src->particle_cnt; ++j) {
                uint32_t s_idx = src->first_particle_idx + j;

                float p_dx = sx[s_idx] - t_x;
                float p_dy = sy[s_idx] - t_y;
                float p_dz = sz[s_idx] - t_z;

                // add negligible value to prevent div by 0
                float p_dist_sq =
                    (p_dx * p_dx) + (p_dy * p_dy) + (p_dz * p_dz) + 0.1F;

                float inv_dist  = 1.0F / sqrtf(p_dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;

                // gravity force magnitude
                float force = sm[s_idx] * inv_dist3;

                f_x += p_dx * force;
                f_y += p_dy * force;
                f_z += p_dz * force;
            }

            arrs->fx[t_idx] += t_mass * f_x;
            arrs->fy[t_idx] += t_mass * f_y;
            arrs->fz[t_idx] += t_mass * f_z;
        }

        return;
    }

    // otherwise subdivide and recurse (like in m2l)
    if (target->is_leaf) {
        for (int i = 0; i < 8; ++i) {
            ps_impl_fmm_p2p_pass(target, src->children[i], arrs);
        }
    } else if (src->is_leaf) {
        for (int i = 0; i < 8; ++i) {
            ps_impl_fmm_p2p_pass(target->children[i], src, arrs);
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            if (!target->children[i]) {
                continue;
            }

            for (int j = 0; j < 8; ++j) {
                if (!src->children[j]) {
                    continue;
                }

                ps_impl_fmm_p2p_pass(target->children[i], src->children[j],
                                     arrs);
            }
        }
    }
}

#define PS_IMPL_SWAP_PTR(type, a, b)                                           \
    do {                                                                       \
        type tmp = a;                                                          \
        (a)      = b;                                                          \
        (b)      = tmp;                                                        \
    } while (0)

static ps_result_t ps_impl_sort_particles(ps_arena_t* arena,
                                          uint32_t*   morton_codes,
                                          const ps_particle_arrs_t* arrs) {
    size_t cnt = arrs->cnt;
    if (cnt == 0) {
        return PS_OK;
    }

    // used later to reclaim temp memory
    size_t tmp_off = arena->off;

    // borrow temp SoA buffers from the arena
    uint32_t* m_tmp =
        (uint32_t*)ps_arena_alloc(arena, cnt * sizeof(uint32_t), 32);
    uint32_t* id_tmp =
        (uint32_t*)ps_arena_alloc(arena, cnt * sizeof(uint32_t), 32);
    float* x_tmp    = (float*)ps_arena_alloc(arena, cnt * sizeof(float), 32);
    float* y_tmp    = (float*)ps_arena_alloc(arena, cnt * sizeof(float), 32);
    float* z_tmp    = (float*)ps_arena_alloc(arena, cnt * sizeof(float), 32);
    float* mass_tmp = (float*)ps_arena_alloc(arena, cnt * sizeof(float), 32);

    if (!m_tmp || !x_tmp || !y_tmp || !z_tmp || !mass_tmp) {
        return PS_EOOM;
    }

    uint32_t* m_src    = morton_codes;
    uint32_t* m_dst    = m_tmp;
    uint32_t* id_src   = arrs->id;
    uint32_t* id_dst   = id_tmp;
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
        for (int i = 1; i < 256; ++i) {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }

        // distribute the data into dest buffers
        for (size_t i = 0; i < cnt; ++i) {
            uint8_t  bucket  = (m_src[i] >> shift) & 0xFF;
            uint32_t dst_idx = offsets[bucket]++;

            m_dst[dst_idx]    = m_src[i];
            id_dst[dst_idx]   = id_src[i];
            x_dst[dst_idx]    = x_src[i];
            y_dst[dst_idx]    = y_src[i];
            z_dst[dst_idx]    = z_src[i];
            mass_dst[dst_idx] = mass_src[i];
        }

        // pingpong pointers for next pass
        PS_IMPL_SWAP_PTR(uint32_t*, m_src, m_dst);
        PS_IMPL_SWAP_PTR(uint32_t*, id_src, id_dst);
        PS_IMPL_SWAP_PTR(float*, x_src, x_dst);
        PS_IMPL_SWAP_PTR(float*, y_src, y_dst);
        PS_IMPL_SWAP_PTR(float*, z_src, z_dst);
        PS_IMPL_SWAP_PTR(float*, mass_src, mass_dst);
    }

    // reclaim memory
    arena->off = tmp_off;

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
                           uint32_t* morton_codes, float root_cx, float root_cy,
                           float root_cz, float root_hw) {
    if (!ctx || !arrs) {
        return PS_EINVAL;
    }

    // clear arena for new frame
    ps_arena_clear(&ctx->arena);

    // sort the arrays based on morton_codes as keys
    ps_impl_sort_particles(&ctx->arena, morton_codes, arrs);

    // allocate the root node
    ctx->root = (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 32);
    if (!ctx->root) {
        return PS_EOOM;
    }

    ps_impl_node_init(ctx->root);

    // seed geometry
    ctx->root->x          = root_cx;
    ctx->root->y          = root_cy;
    ctx->root->z          = root_cz;
    ctx->root->half_width = root_hw;

    // build the octree
    for (size_t i = 0; i < arrs->cnt; ++i) {
        ps_impl_tree_insert(&ctx->arena, ctx->root, morton_codes[i],
                            (uint32_t)i);

        arrs->fx[i] = 0.0F;
        arrs->fy[i] = 0.0F;
        arrs->fz[i] = 0.0F;
    }

    // upward pass (p2m -> m2m)
    ps_impl_fmm_upward_pass(ctx->root, arrs);

    // interaction pass (m2l)
    ps_impl_fmm_interaction_pass(ctx->root, ctx->root);

    // downward pass (l2l)
    ps_impl_fmm_downward_pass(ctx->root);

    // evaluation pass (l2p->p2p)
    ps_impl_fmm_l2p_pass(ctx->root, arrs);
    ps_impl_fmm_p2p_pass(ctx->root, ctx->root, arrs);

    return PS_OK;
}

ps_result_t ps_prepare_particles(ps_particle_arrs_t* arrs,
                                 uint32_t* out_morton_codes, float* out_min_b,
                                 float* out_max_b, float* out_range) {
    if (!arrs || !out_morton_codes || arrs->cnt == 0) {
        return PS_EINVAL;
    }

    float  min_b = FLT_MAX;
    float  max_b = -FLT_MAX; // FLT_MIN is minimum normalized positive float
    size_t i     = 0;

#if defined(PS_USE_AVX2)
    __m256 v_min = _mm256_set1_ps(FLT_MAX);
    __m256 v_max = _mm256_set1_ps(-FLT_MAX);

    for (; i + 7 < arrs->cnt; i += 8) {
        __m256 vx = _mm256_loadu_ps(&arrs->x[i]);
        __m256 vy = _mm256_loadu_ps(&arrs->y[i]);
        __m256 vz = _mm256_loadu_ps(&arrs->z[i]);

        // find the local min/max for x, y, and z within this chunk
        __m256 v_cmin = _mm256_min_ps(vx, _mm256_min_ps(vy, vz));
        __m256 v_cmax = _mm256_max_ps(vx, _mm256_max_ps(vy, vz));

        // accumulate to global min/max
        v_min = _mm256_min_ps(v_min, v_cmin);
        v_max = _mm256_max_ps(v_max, v_cmax);
    }

    // extract the 8 lanes and find abs min/max
    float temp_min[8], temp_max[8];
    _mm256_storeu_ps(temp_min, v_min);
    _mm256_storeu_ps(temp_max, v_max);
    for (int j = 0; j < 8; ++j) {
        if (temp_min[j] < min_b) {
            min_b = temp_min[j];
        }
        if (temp_max[j] > max_b) {
            max_b = temp_max[j];
        }
    }
#elif defined(PS_USE_NEON)
    float32x4_t v_min = vdupq_n_f32(FLT_MAX);
    float32x4_t v_max = vdupq_n_f32(-FLT_MAX);

    for (; i + 3 < arrs->cnt; i += 4) {
        float32x4_t vx = vld1q_f32(&arrs->x[i]);
        float32x4_t vy = vld1q_f32(&arrs->y[i]);
        float32x4_t vz = vld1q_f32(&arrs->z[i]);

        // find the local min/max for x, y, and z within this chunk
        float32x4_t v_cmin = vminq_f32(vx, vminq_f32(vy, vz));
        float32x4_t v_cmax = vmaxq_f32(vx, vmaxq_f32(vy, vz));

        // accumulate to global min/max
        v_min = vminq_f32(v_min, v_cmin);
        v_max = vmaxq_f32(v_max, v_cmax);
    }

    // extract the 4 lanes and find abs min/max
    float temp_min[4], temp_max[4];
    vst1q_f32(temp_min, v_min);
    vst1q_f32(temp_max, v_max);
    for (int j = 0; j < 4; ++j) {
        if (temp_min[j] < min_b) {
            min_b = temp_min[j];
        }
        if (temp_max[j] > max_b) {
            max_b = temp_max[j];
        }
    }
#endif

    // fallback
    // find cubic bb bounds
    for (; i < arrs->cnt; i++) {
        if (arrs->x[i] < min_b) {
            min_b = arrs->x[i];
        }
        if (arrs->y[i] < min_b) {
            min_b = arrs->y[i];
        }
        if (arrs->z[i] < min_b) {
            min_b = arrs->z[i];
        }

        if (arrs->x[i] > max_b) {
            max_b = arrs->x[i];
        }
        if (arrs->y[i] > max_b) {
            max_b = arrs->y[i];
        }
        if (arrs->z[i] > max_b) {
            max_b = arrs->z[i];
        }
    }

    float range = max_b - min_b;
    if (range < 0.001F) {
        range = 0.001F;
    }

    i = 0;

#if defined(PS_USE_AVX2)
    __m256 v_min_b = _mm256_set1_ps(min_b);
    __m256 v_scale = _mm256_set1_ps(1023.0F / range);
    __m256 v_zero  = _mm256_setzero_ps();
    __m256 v_1023  = _mm256_set1_ps(1023.0F);

    // consts for morton expansion
    __m256i m_000003FF = _mm256_set1_epi32(0x000003FF);
    __m256i m_030000FF = _mm256_set1_epi32(0x030000FF);
    __m256i m_0300F00F = _mm256_set1_epi32(0x0300F00F);
    __m256i m_030C30C3 = _mm256_set1_epi32(0x030C30C3);
    __m256i m_09249249 = _mm256_set1_epi32(0x09249249);

// bit exp macro
#define PS_EXPAND_AXV2(v)                                                      \
    (v) = _mm256_and_si256(v, m_000003FF);                                     \
    (v) = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 16)),       \
                           m_030000FF);                                        \
    (v) = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 8)),        \
                           m_0300F00F);                                        \
    (v) = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 4)),        \
                           m_030C30C3);                                        \
    (v) = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 2)),        \
                           m_09249249);

    for (; i + 7 < arrs->cnt; i += 8) {
        __m256 vx = _mm256_loadu_ps(&arrs->x[i]);
        __m256 vy = _mm256_loadu_ps(&arrs->y[i]);
        __m256 vz = _mm256_loadu_ps(&arrs->z[i]);

        // map to 0-1023
        vx = _mm256_mul_ps(_mm256_sub_ps(vx, v_min_b), v_scale);
        vy = _mm256_mul_ps(_mm256_sub_ps(vy, v_min_b), v_scale);
        vz = _mm256_mul_ps(_mm256_sub_ps(vz, v_min_b), v_scale);

        // clamp to 0-1023
        vx = _mm256_max_ps(v_zero, _mm256_min_ps(vx, v_1023));
        vy = _mm256_max_ps(v_zero, _mm256_min_ps(vy, v_1023));
        vz = _mm256_max_ps(v_zero, _mm256_min_ps(vz, v_1023));

        // convert to int
        __m256i mx = _mm256_cvtps_epi32(vx);
        __m256i my = _mm256_cvtps_epi32(vy);
        __m256i mz = _mm256_cvtps_epi32(vz);

        // expand bits for morton encoding
        PS_EXPAND_AXV2(mx);
        PS_EXPAND_AXV2(my);
        PS_EXPAND_AXV2(mz);

        // interleave bits and store morton codes
        // ix | (iy << 1) | (iz << 2)
        __m256i morton_codes_vec =
            _mm256_or_si256(_mm256_or_si256(mx, _mm256_slli_epi32(my, 1)),
                            _mm256_slli_epi32(mz, 2));

        _mm256_storeu_si256((__m256i*)&out_morton_codes[i], morton_codes_vec);
    }
#undef PS_EXPAND_AXV2
#elif defined(PS_USE_NEON)
    float32x4_t v_min_b = vdupq_n_f32(min_b);
    float32x4_t v_scale = vdupq_n_f32(1023.0F / range);
    float32x4_t v_zero  = vdupq_n_f32(0.0F);
    float32x4_t v_1023  = vdupq_n_f32(1023.0F);

    // consts for morton expansion
    int32x4_t m_000003FF = vdupq_n_s32(0x000003FF);
    int32x4_t m_030000FF = vdupq_n_s32(0x030000FF);
    int32x4_t m_0300F00F = vdupq_n_s32(0x0300F00F);
    int32x4_t m_030C30C3 = vdupq_n_s32(0x030C30C3);
    int32x4_t m_09249249 = vdupq_n_s32(0x09249249);

// bit exp macro
#define PS_EXPAND_NEON(v)                                                      \
    (v) = vandq_s32(v, m_000003FF);                                            \
    (v) = vandq_s32(vorrq_s32(v, vshlq_n_s32(v, 16)), m_030000FF);             \
    (v) = vandq_s32(vorrq_s32(v, vshlq_n_s32(v, 8)), m_0300F00F);              \
    (v) = vandq_s32(vorrq_s32(v, vshlq_n_s32(v, 4)), m_030C30C3);              \
    (v) = vandq_s32(vorrq_s32(v, vshlq_n_s32(v, 2)), m_09249249);

    for (; i + 3 < arrs->cnt; i += 4) {
        float32x4_t vx = vld1q_f32(&arrs->x[i]);
        float32x4_t vy = vld1q_f32(&arrs->y[i]);
        float32x4_t vz = vld1q_f32(&arrs->z[i]);

        // map to 0-1023
        vx = vmulq_f32(vsubq_f32(vx, v_min_b), v_scale);
        vy = vmulq_f32(vsubq_f32(vy, v_min_b), v_scale);
        vz = vmulq_f32(vsubq_f32(vz, v_min_b), v_scale);

        // clamp to 0-1023
        vx = vmaxq_f32(v_zero, vminq_f32(vx, v_1023));
        vy = vmaxq_f32(v_zero, vminq_f32(vy, v_1023));
        vz = vmaxq_f32(v_zero, vminq_f32(vz, v_1023));

        // convert to int
        int32x4_t mx = vcvtq_s32_f32(vx);
        int32x4_t my = vcvtq_s32_f32(vy);
        int32x4_t mz = vcvtq_s32_f32(vz);

        // expand bits for morton encoding
        PS_EXPAND_NEON(mx);
        PS_EXPAND_NEON(my);
        PS_EXPAND_NEON(mz);

        // interleave bits and store morton codes
        // ix | (iy << 1) | (iz << 2)
        int32x4_t morton_codes_vec =
            vorrq_s32(mx, vorrq_s32(vshlq_n_s32(my, 1), vshlq_n_s32(mz, 2)));

        vst1q_s32((int32_t*)&out_morton_codes[i], morton_codes_vec);
    }
#undef PS_EXPAND_NEON
#endif

    // fallback
    // gen morton codes mapped to 0-1023
    for (; i < arrs->cnt; i++) {
        int mx = (int)(((arrs->x[i] - min_b) / range) * 1023.0F);
        int my = (int)(((arrs->y[i] - min_b) / range) * 1023.0F);
        int mz = (int)(((arrs->z[i] - min_b) / range) * 1023.0F);

        if (mx < 0) {
            mx = 0;
        }
        if (mx > 1023) {
            mx = 1023;
        }
        if (my < 0) {
            my = 0;
        }
        if (my > 1023) {
            my = 1023;
        }
        if (mz < 0) {
            mz = 0;
        }
        if (mz > 1023) {
            mz = 1023;
        }

        out_morton_codes[i] = ps_impl_morton_encode(mx, my, mz);
    }

    if (out_min_b) {
        *out_min_b = min_b;
    }
    if (out_max_b) {
        *out_max_b = max_b;
    }
    if (out_range) {
        *out_range = range;
    }

    return PS_OK;
}

#endif // POLESITTER_IMPLEMENTATION

/*
    This software is available under the MIT license:

    Copyright (c) 2026 Patryk Pujanek

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/
