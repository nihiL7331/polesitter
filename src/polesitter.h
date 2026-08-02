/*
    polesitter - Fast Multipole Method (FFM) N-body solver written in C99

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
#ifndef POLESITTER_H
#define POLESITTER_H

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__AVX2__)
#    include <immintrin.h>
#    define PS_USE_AVX2
#elif defined(__aarch64__) || defined(_M_ARM64)
#    include <arm_neon.h>
#    define PS_USE_NEON
#else
#    define PS_USE_SCALAR
#endif

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

// pass 1
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

// pass 2
// m2l dual-tree traversal to find well-separated nodes and translate expansions
static void ps__fmm_interaction_pass(ps_node_t* target, ps_node_t* src) {
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
            ps__fmm_interaction_pass(target, src->children[i]);
        }
    } else if (src->is_leaf) {
        // src is as small as possible, open the target
        for (int i = 0; i < 8; ++i) {
            ps__fmm_interaction_pass(target->children[i], src);
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

                ps__fmm_interaction_pass(target->children[i], src->children[j]);
            }
        }
    }
}

// pass 3
// l2l, pushes the accumulated background field from parents down to their
// children
static void ps__fmm_downward_pass(ps_node_t* node) {
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

            ps__fmm_downward_pass(child);
        }
    }
}

// pass 4
// l2p, applies the accumulated bg field to the particles inside the leaf
static void ps__fmm_l2p_pass(ps_node_t* node, const ps_particle_arrs_t* arrs) {
    if (!node) {
        return;
    }

    if (node->is_leaf) {
        float field_x = node->local[1];
        float field_y = node->local[2];
        float field_z = node->local[3];

        for (uint32_t i = 0; i < node->particle_cnt; ++i) {
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
        ps__fmm_l2p_pass(node->children[i], arrs);
    }
}

// pass 5
// p2p, dual-tree traversal to calc N-body forces for near-field neighbors
static void ps__fmm_p2p_pass(ps_node_t* target, ps_node_t* src,
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

            const float* restrict sx = arrs->x;
            const float* restrict sy = arrs->y;
            const float* restrict sz = arrs->z;
            const float* restrict sm = arrs->mass;

            uint32_t j = 0;
#if defined(PS_USE_AVX2)
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
                __m256 s_x_vec = _mm256_load_ps(&sx[s_idx]);
                __m256 s_y_vec = _mm256_load_ps(&sy[s_idx]);
                __m256 s_z_vec = _mm256_load_ps(&sz[s_idx]);
                __m256 s_m_vec = _mm256_load_ps(&sm[s_idx]);

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
            ps__fmm_p2p_pass(target, src->children[i], arrs);
        }
    } else if (src->is_leaf) {
        for (int i = 0; i < 8; ++i) {
            ps__fmm_p2p_pass(target->children[i], src, arrs);
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

                ps__fmm_p2p_pass(target->children[i], src->children[j], arrs);
            }
        }
    }
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

    // used later to reclaim temp memory
    size_t tmp_off = arena->off;

    // borrow temp SoA buffers from the arena
    uint32_t* m_tmp =
        (uint32_t*)ps_arena_alloc(arena, cnt * sizeof(uint32_t), 16);
    uint32_t* id_tmp =
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
        PS__SWAP_PTR(uint32_t*, m_src, m_dst);
        PS__SWAP_PTR(uint32_t*, id_src, id_dst);
        PS__SWAP_PTR(float*, x_src, x_dst);
        PS__SWAP_PTR(float*, y_src, y_dst);
        PS__SWAP_PTR(float*, z_src, z_dst);
        PS__SWAP_PTR(float*, mass_src, mass_dst);
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

        arrs->fx[i] = 0.0F;
        arrs->fy[i] = 0.0F;
        arrs->fz[i] = 0.0F;
    }

    // upward pass (p2m -> m2m)
    ps__fmm_upward_pass(ctx->root, arrs);

    // interaction pass (m2l)
    ps__fmm_interaction_pass(ctx->root, ctx->root);

    // downward pass (l2l)
    ps__fmm_downward_pass(ctx->root);

    // evaluation pass (l2p->p2p)
    ps__fmm_l2p_pass(ctx->root, arrs);
    ps__fmm_p2p_pass(ctx->root, ctx->root, arrs);

    return PS_OK;
}

ps_result_t ps_prepare_particles(ps_particle_arrs_t* arrs,
                                 uint32_t* out_morton_codes, float* out_min_b,
                                 float* out_max_b, float* out_range) {
    if (!arrs || !out_morton_codes || arrs->cnt == 0) {
        return PS_EINVAL;
    }

    float min_b = FLT_MAX;
    float max_b = -FLT_MAX; // FLT_MIN is minimum normalized positive float

    // find cubic bb bounds
    for (size_t i = 0; i < arrs->cnt; i++) {
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

    // gen morton codes mapped to 0-1023
    for (size_t i = 0; i < arrs->cnt; i++) {
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

        out_morton_codes[i] = ps__morton_encode(mx, my, mz);
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
