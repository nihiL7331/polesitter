// clang-format off
/*-
    polesitter - v2.0 - Fast Multipole Method (FFM) N-body solver written in C99

    Do this:
        #define POLESITTER_IMPLEMENTATION
    before you include this file in *one* C or C++ file to create the
   implementation.

   Use #define PS_MULTITHREADING to enable multithreading capabilities.

    // i.e. it should look like this:
    #include ...
    #include ...
    #include ...
    #define PS_MULTITHREADING
    #define POLESITTER_IMPLEMENTATION
    #include "polesitter.h"

    QUICK NOTES:
        Primarily of interest to game devs and graphics programmers.

        Requires bringing your own memory buffer.
        No malloc/free calls are made internally.

    LICENSE
        See end of file for license information.

    RESOURCES
        The Fastest Gravity Algorithm You've Never Heard Of, Keyframe Codes: https://youtu.be/FhMftauQZqU?si=E3nmNp6FuSqhn2OD
        Fast multipole method, Wikipedia: https://en.wikipedia.org/wiki/Fast_multipole_method
        Introduction to FFM, Long Chen: https://www.math.uci.edu/~chenlong/226/FMMsimple.pdf
        A short course on fast multipole methods, Rick Beatson; Leslie Greengard: https://math.nyu.edu/~greengar/shortcourse_fmm.pdf

    QUICKSTART

        ```
        #define PS_MULTITHREADING
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
            ps_config_t cfg = {
                .buff = memory_block,
                .buff_size = MEMORY_SIZE,
                .max_particles = PARTICLE_CNT,
                .theta = 2.0F,
                .thrd_cnt = 4
            };
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
            ps_destroy(ctx);
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
#include <string.h>

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

#ifdef PS_MULTITHREADING

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef HANDLE             ps_thrd_t;
typedef CRITICAL_SECTION   ps_mtx_t;
typedef CONDITION_VARIABLE ps_cond_t;
typedef DWORD(WINAPI* ps_thrd_func_t)(void*);

#define PS_THRD_RET_TYPE DWORD WINAPI
#define PS_THRD_RET_VAL  0

#if defined(_WIN64)

// clang-format off

#define ps_atomic_fetch_add_size_t(ptr, val)                                   \
    (size_t)InterlockedExchangeAdd64((volatile LONG64*)(ptr), (LONG64)(val))
#else
#define ps_atomic_fetch_add_size_t(ptr, val)                                   \
    (size_t)InterlockedExchangeAdd((volatile LONG*)(ptr), (LONG)(val))
#endif

// clang-format on

#else // POSIX

#include <pthread.h>

typedef pthread_t       ps_thrd_t;
typedef pthread_mutex_t ps_mtx_t;
typedef pthread_cond_t  ps_cond_t;
typedef void* (*ps_thrd_func_t)(void*);

#define PS_THRD_RET_TYPE void*
#define PS_THRD_RET_VAL  NULL

#define ps_atomic_fetch_add_size_t(ptr, val) __sync_fetch_and_add((ptr), (val))

#endif // POSIX

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
    defined(_M_IX86)

#ifdef _WIN32
#define PS_YIELD() YieldProcessor()
#else
#define PS_YIELD() __asm__ volatile("pause" ::: "memory")
#endif
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
#define PS_YIELD() __asm__ volatile("yield" ::: "memory")
#else
#define PS_YIELD()
#endif

typedef volatile int ps_spinlock_t;

#ifdef _WIN32

#define ps_spin_lock(lock)                                                     \
    while (InterlockedExchange((volatile LONG*)(lock), 1)) {                   \
        while (*(lock))                                                        \
            PS_YIELD();                                                        \
    }
#define ps_spin_unlock(lock) InterlockedExchange((volatile LONG*)(lock), 0)

#else

#define ps_spin_lock(lock)                                                     \
    while (__sync_lock_test_and_set((lock), 1)) {                              \
        while (*(lock))                                                        \
            PS_YIELD();                                                        \
    }

#define ps_spin_unlock(lock) __sync_lock_release(lock)

#endif

#endif // PS_MULTITHREADING

// =====================================================================
// public api
// =====================================================================

typedef enum {
    PS_OK     = 0,
    PS_EOOM   = -1, // out of memory
    PS_EINVAL = -2, // invalid argument
    PS_ETHRD  = -3, // thread creation failed
    PS_EALLOC = -4, // allocation failed
} ps_result_t;

typedef struct ps_context ps_context_t;

typedef struct {
    void*  buff;          // ram provided by host
    size_t buff_size;     // total size in B
    size_t max_particles; // maximum capacity for the solver
    float  theta;         // MAC threshold. 0.0F defaults to 1.0F
    size_t thrd_cnt;      // 0 defaults to 1 (single-threaded impl)
} ps_config_t;

typedef struct {
    float* x;
    float* y;
#ifndef PS_2D
    float* z;
#endif // PS_3D
    float* mass;

    float* fx;
    float* fy;
#ifndef PS_2D
    float* fz;
#endif // PS_3D

    uint32_t* id;
    size_t    cnt;
} ps_particle_arrs_t;

// init pipeline with a pre-allocated buffer.
ps_result_t ps_init(ps_context_t** out_ctx, const ps_config_t* cfg);

// compute forces on particles using FMM
ps_result_t ps_calc_forces(ps_context_t* ctx, const ps_particle_arrs_t* arrs,
                           uint32_t* morton_codes, float root_cx, float root_cy,
#ifndef PS_2D
                           float root_cz,
#endif // PS_3D
                           float root_hw);

// calculates the global bounding box and generates morton codes for all
// particles
ps_result_t ps_prepare_particles(ps_particle_arrs_t* arrs,
                                 uint32_t* out_morton_codes, float* out_min_b,
                                 float* out_max_b, float* out_range);

// shuts down bg threads safely before freeing memory
ps_result_t ps_destroy(ps_context_t* ctx);

#endif // POLESITTER_H

#define POLESITTER_IMPLEMENTATION
#ifdef POLESITTER_IMPLEMENTATION

// =====================================================================
// internal implementation
// =====================================================================

// ---------------------------------------------------------------------
// internal structs
// ---------------------------------------------------------------------

// represents one threads memory slice
typedef struct {
    uint8_t*        mem;
    size_t          cap;
    volatile size_t off;
} ps_arena_t;

#ifdef PS_2D

// quadtree node
// exactly 32 bytes (2/cache line)
typedef struct ps_node {
    // physics, 16B
    float    x;
    float    y;
    float    half_width; // needed for M2L
    uint32_t _pad;

    // FMM payload, 16B
    union {
        uint32_t children_offs[4];

        struct {
            uint32_t is_leaf;
            uint32_t particle_cnt;
            uint32_t first_particle_idx;
            uint32_t _pad2;
        } leaf;
    } data;
} ps_node_t;

#else // PS_3D

// octree node
// exactly 64 bytes (1 cache line)
typedef struct ps_node {
    // physics, 16B
    float x;
    float y;
    float z;
    float half_width; // needed for M2L

    // FMM payload, 48B
    uint32_t _pad[4];
    union {
        uint32_t children_offs[8];

        struct {
            // leaf metadata, 16B
            uint32_t is_leaf;
            uint32_t particle_cnt;
            uint32_t first_particle_idx;
            uint32_t _pad2[5];
        } leaf;
    } data;
} ps_node_t;

#endif // PS_3D

#ifdef PS_2D

#define PS_OCTANTS         4
#define PS_EXPANSION_TERMS 3 // 2p + 1, p = 1

#else // PS_3D

#define PS_OCTANTS         8
#define PS_EXPANSION_TERMS 4 // (p + 1)^2, p = 1

#endif // PS_3D

#ifndef PS_MAX_THRDS
#define PS_MAX_THRDS 32
#endif // PS_MAX_THRDS

#ifndef PS_MAX_JOBS
#define PS_MAX_JOBS 512
#endif // PS_MAX_JOBS

typedef enum {
    PS_JOB_EXIT = 0,
    PS_JOB_UPWARD,
    PS_JOB_INTERACTION,
    PS_JOB_DOWNWARD,
    PS_JOB_P2P,
    PS_JOB_RADIX_MAP,
    PS_JOB_RADIX_SCATTER,
    PS_JOB_TREE_BUILD,
} ps_job_type_t;

typedef struct {
    ps_job_type_t type;
    uint32_t      thrd_id; // which worker is executing this

    union {
        // for fmm passes
        struct {
            ps_node_t*                target;
            ps_node_t*                src;
            const ps_particle_arrs_t* arrs;
        } fmm;

        // for radix
        struct {
            uint32_t start_idx;
            uint32_t end_idx;
            uint32_t chunk_id;
        } array;

        // for tree
        struct {
            ps_node_t* node;
            uint32_t   start_idx;
            uint32_t   end_idx;
            uint32_t   depth;
        } tree;
    } data;
} ps_job_t;

#ifdef PS_MULTITHREADING

typedef struct {
    ps_thrd_t thrds[PS_MAX_THRDS];
    uint32_t  thrd_cnt;

    ps_job_t queue[PS_MAX_JOBS];
    int      hd;
    int      tl;
    int      cnt;
    int      active_jobs;
    int      shutdown_flag;

    ps_spinlock_t lock;
} ps_thrd_pool_t;

// payload passed to each worker thread when it spawns
typedef struct {
    ps_thrd_pool_t* pool;
    ps_context_t*   ctx;
    uint32_t        thrd_id; // 0 is main thrd
} ps_worker_arg_t;

#endif // PS_MULTITHREADING

#define PS_RADIX_BUF_SIZE 16

typedef struct {
    uint32_t m[256][PS_RADIX_BUF_SIZE];
    uint32_t id[256][PS_RADIX_BUF_SIZE];
    float    x[256][PS_RADIX_BUF_SIZE];
    float    y[256][PS_RADIX_BUF_SIZE];
    float    z[256][PS_RADIX_BUF_SIZE];
    float    mass[256][PS_RADIX_BUF_SIZE];
    uint8_t  cnt[256];
} ps_scatter_buf_t;

// holds the shared state for the radix pipeline
typedef struct {
    uint32_t* m_src;
    uint32_t* m_dst;
    uint32_t* id_src;
    uint32_t* id_dst;
    float*    x_src;
    float*    x_dst;
    float*    y_src;
    float*    y_dst;
    float*    z_src;
    float*    z_dst;
    float*    mass_src;
    float*    mass_dst;

    ps_scatter_buf_t* bufs;
    uint8_t           pass; // 0-3

    // each thrd gets its own 256-bin histogram and off array
    uint32_t hists[PS_MAX_THRDS + 1][256];
    uint32_t offs[PS_MAX_THRDS + 1][256];
} ps_radix_state_t;

struct ps_context {
    ps_node_t* root;
    float      theta;
    ps_arena_t arena;
    float (*local_exp)[PS_EXPANSION_TERMS];
    float (*multipole_exp)[PS_EXPANSION_TERMS];
    ps_radix_state_t radix_state;

#ifdef PS_MULTITHREADING

    ps_thrd_pool_t  pool;
    ps_worker_arg_t worker_args[PS_MAX_THRDS];

#endif // PS_MULTITHREADING
};

// ---------------------------------------------------------------------
// internal functions
// ---------------------------------------------------------------------

// separated from the above for clarity purposes
#ifdef PS_MULTITHREADING

#ifdef _WIN32

static inline void ps_mtx_init(ps_mtx_t* m) {
    InitializeCriticalSection(m);
}
static inline void ps_mtx_destroy(ps_mtx_t* m) {
    DeleteCriticalSection(m);
}
static inline void ps_mtx_lock(ps_mtx_t* m) {
    EnterCriticalSection(m);
}
static inline void ps_mtx_unlock(ps_mtx_t* m) {
    LeaveCriticalSection(m);
}

static inline void ps_cond_init(ps_cond_t* c) {
    InitializeConditionVariable(c);
}
static inline void ps_cond_destroy(ps_cond_t* c) {
    (void)c; /* no-op */
}
static inline void ps_cond_wait(ps_cond_t* c, ps_mtx_t* m) {
    SleepConditionVariableCS(c, m, INFINITE);
}
static inline void ps_cond_signal(ps_cond_t* c) {
    WakeConditionVariable(c);
}
static inline void ps_cond_bcast(ps_cond_t* c) {
    WakeAllConditionVariable(c);
}

static inline int ps_thrd_create(ps_thrd_t* t, ps_thrd_func_t f, void* a) {
    *t = CreateThread(NULL, 0, f, a, 0, NULL);
    return (*t != NULL) ? 0 : -1;
}
static inline void ps_thrd_join(ps_thrd_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}

#else // POSIX

static inline void ps_mtx_init(ps_mtx_t* m) {
    pthread_mutex_init(m, NULL);
}
static inline void ps_mtx_destroy(ps_mtx_t* m) {
    pthread_mutex_destroy(m);
}
static inline void ps_mtx_lock(ps_mtx_t* m) {
    pthread_mutex_lock(m);
}
static inline void ps_mtx_unlock(ps_mtx_t* m) {
    pthread_mutex_unlock(m);
}

static inline void ps_cond_init(ps_cond_t* c) {
    pthread_cond_init(c, NULL);
}
static inline void ps_cond_destroy(ps_cond_t* c) {
    pthread_cond_destroy(c);
}
static inline void ps_cond_wait(ps_cond_t* c, ps_mtx_t* m) {
    pthread_cond_wait(c, m);
}
static inline void ps_cond_signal(ps_cond_t* c) {
    pthread_cond_signal(c);
}
static inline void ps_cond_bcast(ps_cond_t* c) {
    pthread_cond_broadcast(c);
}

static inline int ps_thrd_create(ps_thrd_t* t, ps_thrd_func_t f, void* a) {
    return pthread_create(t, NULL, f, a);
}
static inline void ps_thrd_join(ps_thrd_t t) {
    pthread_join(t, NULL);
}

#endif

#endif // PS_MULTITHREADING

// forward declare passes for router
static void ps_impl_fmm_upward_pass(ps_context_t* ctx, ps_node_t* node,
                                    const ps_particle_arrs_t* arrs);
static void ps_impl_fmm_interaction_pass(ps_context_t* ctx, ps_node_t* target,
                                         ps_node_t* src, float theta);
static void ps_impl_fmm_downward_pass(ps_context_t* ctx, ps_node_t* node,
                                      const ps_particle_arrs_t* arrs);
static void ps_impl_fmm_p2p_pass(ps_context_t* ctx, ps_node_t* target,
                                 ps_node_t* src, const ps_particle_arrs_t* arrs,
                                 float theta);
static void ps_impl_radix_map(ps_context_t* ctx, uint32_t chunk_id,
                              size_t start_idx, size_t end_idx);
static void ps_impl_radix_scatter(ps_context_t* ctx, uint32_t chunk_id,
                                  size_t start_idx, size_t end_idx);
static void ps_impl_build_tree(ps_context_t* ctx, uint32_t thrd_id,
                               ps_node_t* node, uint32_t depth,
                               size_t start_idx, size_t end_idx,
                               const uint32_t* morton_codes);

static inline void ps_impl_exec_job(ps_context_t* ctx, ps_job_t* job) {
    if (job->type == PS_JOB_UPWARD) {
        ps_impl_fmm_upward_pass(ctx, job->data.fmm.src, job->data.fmm.arrs);
    } else if (job->type == PS_JOB_INTERACTION) {
        ps_impl_fmm_interaction_pass(ctx, job->data.fmm.target,
                                     job->data.fmm.src, ctx->theta);
    } else if (job->type == PS_JOB_DOWNWARD) {
        ps_impl_fmm_downward_pass(ctx, job->data.fmm.src, job->data.fmm.arrs);
    } else if (job->type == PS_JOB_P2P) {
        ps_impl_fmm_p2p_pass(ctx, job->data.fmm.target, job->data.fmm.src,
                             job->data.fmm.arrs, ctx->theta);
    } else if (job->type == PS_JOB_RADIX_MAP) {
        ps_impl_radix_map(ctx, job->data.array.chunk_id,
                          job->data.array.start_idx, job->data.array.end_idx);
    } else if (job->type == PS_JOB_RADIX_SCATTER) {
        ps_impl_radix_scatter(ctx, job->data.array.chunk_id,
                              job->data.array.start_idx,
                              job->data.array.end_idx);
    } else if (job->type == PS_JOB_TREE_BUILD) {
        ps_impl_build_tree(ctx, job->thrd_id, job->data.tree.node,
                           job->data.tree.depth, job->data.tree.start_idx,
                           job->data.tree.end_idx, ctx->radix_state.m_src);
    }
}

static void ps_impl_pool_submit(ps_context_t* ctx, ps_job_t job) {
#ifdef PS_MULTITHREADING

    if (ctx->pool.thrd_cnt > 0) {
        ps_thrd_pool_t* pool = &ctx->pool;

        // Queue state is shared with workers; inspect and update it only
        // while holding the spinlock. volatile does not make these accesses
        // atomic or establish inter-thread ordering in C.
        while (1) {
            ps_spin_lock(&pool->lock);

            if (pool->shutdown_flag) {
                ps_spin_unlock(&pool->lock);
                return;
            }

            if (pool->cnt < PS_MAX_JOBS) {
                pool->queue[pool->tl] = job;
                pool->tl              = (pool->tl + 1) % PS_MAX_JOBS;
                pool->cnt++;
                ps_spin_unlock(&pool->lock);
                return;
            }

            ps_spin_unlock(&pool->lock);
            PS_YIELD();
        }
    }

#endif
    // single-threaded

    job.thrd_id = 0;
    ps_impl_exec_job(ctx, &job);
}

// no-op if single-threaded
static void ps_impl_pool_wait(ps_context_t* ctx) {
    (void)ctx;
#ifdef PS_MULTITHREADING
    ps_thrd_pool_t* pool = &ctx->pool;

    while (1) {
        ps_spin_lock(&pool->lock);
        int done = (pool->cnt == 0 && pool->active_jobs == 0);
        ps_spin_unlock(&pool->lock);

        if (done) {
            return;
        }

        PS_YIELD();
    }
#endif
}

#ifdef PS_MULTITHREADING

static PS_THRD_RET_TYPE ps_impl_worker_loop(void* arg) {
    ps_worker_arg_t* w_arg = (ps_worker_arg_t*)arg;
    ps_thrd_pool_t*  pool  = w_arg->pool;

    while (1) {
        ps_spin_lock(&pool->lock);

        if (pool->cnt == 0) {
            int shutdown = pool->shutdown_flag;
            ps_spin_unlock(&pool->lock);

            if (shutdown) {
                break;
            }

            PS_YIELD();
            continue;
        }

        // dequeue
        ps_job_t job = pool->queue[pool->hd];
        pool->hd     = (pool->hd + 1) % PS_MAX_JOBS;

        pool->active_jobs++;
        pool->cnt--;
        ps_spin_unlock(&pool->lock);

        // execute the job
        job.thrd_id = w_arg->thrd_id;
        ps_impl_exec_job(w_arg->ctx, &job);

        // mark done
        ps_spin_lock(&pool->lock);
        pool->active_jobs--;
        ps_spin_unlock(&pool->lock);
    }

    return PS_THRD_RET_VAL;
}

static int ps_impl_pool_init(ps_context_t* ctx, uint32_t num_thrds) {
    ps_thrd_pool_t*  pool        = &ctx->pool;
    ps_worker_arg_t* worker_args = ctx->worker_args;

    pool->hd            = 0;
    pool->tl            = 0;
    pool->cnt           = 0;
    pool->active_jobs   = 0;
    pool->shutdown_flag = 0;
    pool->thrd_cnt      = num_thrds;

    if (num_thrds == 0) {
        return PS_OK;
    }

    if (pool->thrd_cnt > PS_MAX_THRDS) {
        pool->thrd_cnt = PS_MAX_THRDS;
    }

    pool->lock = 0;

    for (uint32_t i = 0; i < pool->thrd_cnt; ++i) {
        worker_args[i].pool    = pool;
        worker_args[i].ctx     = ctx;
        worker_args[i].thrd_id = i + 1;

        if (ps_thrd_create(&pool->thrds[i], ps_impl_worker_loop,
                           &worker_args[i]) != 0) {

            return PS_ETHRD;
        }
    }

    return PS_OK;
}

#endif // PS_MULTITHREADING

#ifdef PS_2D

// take a 16b num and expand it to 32b by inserting 1 0 between each b.
static uint32_t ps_impl_expand_bits(uint32_t v) {
    v &= 0x0000FFFF; // only look at 16 ls bits

    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;

    return v;
}

// final number has 32 bits.
// dividing it by 2 dimensions, we can store 16 bits per dimension.
// it interleaves the expanded bits of x and y.
static uint32_t ps_impl_morton_encode(uint32_t x, uint32_t y) {
    uint32_t xx = ps_impl_expand_bits(x);
    uint32_t yy = ps_impl_expand_bits(y);

    // x takes bit 0, y shifts to bit 1 ...
    return xx | (yy << 1);
}

#else // PS_3D

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

#endif // PS_3D

// rounds 'v' up to the nearest multiple of 'align', which must be a power of 2
static inline size_t ps_impl_align_forward(size_t v, size_t align) {
    return (v + (align - 1)) & ~(align - 1);
}

// alloc raw bytes
static void* ps_impl_arena_alloc(ps_context_t* ctx, size_t size) {
    if (!ctx) {
        return NULL;
    }

    size = (size + 63) & ~63;

#ifdef PS_MULTITHREADING
    size_t old_off = ps_atomic_fetch_add_size_t(&ctx->arena.off, size);
#else
    size_t old_off = ctx->arena.off;
    ctx->arena.off += size;
#endif

    // check if we have enough space in the arena
    if (old_off + size > ctx->arena.cap) {
        return NULL;
    }

    // advance bump pointer and return aligned address
    return ctx->arena.mem + old_off;
}

// helper to grab a new node
static inline ps_node_t* ps_impl_alloc_node(ps_context_t* ctx) {
    if (!ctx) {
        return NULL;
    }

    size_t size = 64;

#ifdef PS_MULTITHREADING
    size_t old_off = ps_atomic_fetch_add_size_t(&ctx->arena.off, size);
#else
    size_t old_off = ctx->arena.off;
    ctx->arena.off += size;
#endif

    // check if we have enough space in the arena
    if (old_off + size > ctx->arena.cap) {
        return NULL;
    }

    ps_node_t* node = (ps_node_t*)(ctx->arena.mem + old_off);

    memset(node, 0, 64);

    uint32_t idx = (uint32_t)(old_off / 64);

    for (int i = 0; i < PS_EXPANSION_TERMS; ++i) {
        ctx->local_exp[idx][i]     = 0.0F;
        ctx->multipole_exp[idx][i] = 0.0F;
    }

    return node;
}

static inline ps_node_t* ps_impl_get_node(ps_context_t* ctx, uint32_t off) {
    if (off == 0) {
        return NULL;
    }

    return (ps_node_t*)(ctx->arena.mem + off);
}

static inline float* ps_impl_get_local(ps_context_t* ctx, ps_node_t* node) {
    uint32_t idx = (uint32_t)(((uint8_t*)node - ctx->arena.mem) / 64);

    return ctx->local_exp[idx];
}

static inline float* ps_impl_get_multipole(ps_context_t* ctx, ps_node_t* node) {
    uint32_t idx = (uint32_t)(((uint8_t*)node - ctx->arena.mem) / 64);

    return ctx->multipole_exp[idx];
}

// clear for next frame
static void ps_impl_arena_clear(ps_arena_t* arena) {
    arena->off = 0;
}

#define PS_MAX_DEPTH                                                           \
    10 // max depth of octree, 10 levels = 1024 (2^10) leaf nodes

// bin search to find the idx where target oct begins
static inline size_t ps_impl_find_split(const uint32_t* codes, size_t start,
                                        size_t end, int shift,
                                        uint32_t target_oct) {
    size_t left  = start;
    size_t right = end;
    while (left < right) {
        size_t   mid = left + ((right - left) / 2);
        uint32_t oct = (codes[mid] >> shift) & 0x7;

        if (oct < target_oct) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

// walk the morton code and build the tree branches
static void ps_impl_build_tree(ps_context_t* ctx, uint32_t thrd_id,
                               ps_node_t* node, uint32_t depth,
                               size_t start_idx, size_t end_idx,
                               const uint32_t* morton_codes) {
    // max depth reached or only 1 particle left
    if (depth == PS_MAX_DEPTH || end_idx - start_idx <= 1) {
        node->data.leaf.is_leaf            = 1;
        node->data.leaf.first_particle_idx = (uint32_t)start_idx;
        node->data.leaf.particle_cnt       = (uint32_t)(end_idx - start_idx);
        return;
    }

    // shift starts at 27, decreases by 3 each lvl
    uint8_t shift      = (uint8_t)(27 - (depth * 3));
    size_t  curr_start = start_idx;

    // subdivide into 8 octants
    for (uint32_t oct = 0; oct < PS_OCTANTS; ++oct) {
        // find where this octnat ends in the sorted arr
        size_t oct_end = ps_impl_find_split(morton_codes, curr_start, end_idx,
                                            shift, oct + 1);

        // if this octant has particles build a branch
        if (oct_end > curr_start) {
            ps_node_t* child = ps_impl_alloc_node(ctx);
            if (!child) {
                return;
            }

            float hw          = node->half_width * 0.5F;
            child->x          = node->x + ((oct & 1) ? hw : -hw);
            child->y          = node->y + ((oct & 2) ? hw : -hw);
            child->z          = node->z + ((oct & 4) ? hw : -hw);
            child->half_width = hw;

            node->data.children_offs[oct] =
                (uint32_t)((uint8_t*)child - ctx->arena.mem);

// spawn job for top layers
#ifdef PS_MULTITHREADING
            if (depth == 0) {
                ps_job_t job;
                job.type                = PS_JOB_TREE_BUILD;
                job.data.tree.node      = child;
                job.data.tree.start_idx = (uint32_t)curr_start;
                job.data.tree.end_idx   = (uint32_t)oct_end;
                job.data.tree.depth     = depth + 1;
                ps_impl_pool_submit(ctx, job);
            } else
#endif // PS_MULTITHREADING
            {
                ps_impl_build_tree(ctx, thrd_id, child, depth + 1, curr_start,
                                   oct_end, morton_codes);
            }
        }

        curr_start = oct_end;
        if (curr_start == end_idx) {
            break; // all particles handled
        }
    }
}

// pass 1
// p2m (leaves) and m2m (parents) in post-order traversal
static void ps_impl_fmm_upward_pass(ps_context_t* ctx, ps_node_t* node,
                                    const ps_particle_arrs_t* arrs) {
    if (!node) {
        return;
    }

    // p2m
    if (node->data.leaf.is_leaf & 1) {
        float m0 = 0.0F; // monopole (total mass)
        float mx = 0.0F; // dipole x (mass moment)
        float my = 0.0F; // dipole y
        float mz = 0.0F; // dipole z

        for (uint32_t i = 0; i < node->data.leaf.particle_cnt; ++i) {
            uint32_t idx  = node->data.leaf.first_particle_idx + i;
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
        float* n_multi = ps_impl_get_multipole(ctx, node);
        n_multi[0]     = m0;
        n_multi[1]     = mx;
        n_multi[2]     = my;
        n_multi[3]     = mz;
        return;
    }

    // m2m
    float p_m0 = 0.0F;
    float p_mx = 0.0F;
    float p_my = 0.0F;
    float p_mz = 0.0F;

    for (int i = 0; i < PS_OCTANTS; ++i) {
        ps_node_t* child = ps_impl_get_node(ctx, node->data.children_offs[i]);
        if (!child) {
            continue;
        }

        // calc the children first
        ps_impl_fmm_upward_pass(ctx, child, arrs);

        // dist vector from the child's center
        // up to the parent's center
        float dx = child->x - node->x;
        float dy = child->y - node->y;
        float dz = child->z - node->z;

        float* c_multi = ps_impl_get_multipole(ctx, child);
        float  c_m0    = c_multi[0];
        float  c_mx    = c_multi[1];
        float  c_my    = c_multi[2];
        float  c_mz    = c_multi[3];

        // shift the childs expansion to the parents center and accumulate
        // dipole requires the monopole * dist
        p_m0 += c_m0;
        p_mx += c_mx + (c_m0 * dx);
        p_my += c_my + (c_m0 * dy);
        p_mz += c_mz + (c_m0 * dz);
    }

    // store aggregated expansion in parent
    float* n_multi = ps_impl_get_multipole(ctx, node);
    n_multi[0]     = p_m0;
    n_multi[1]     = p_mx;
    n_multi[2]     = p_my;
    n_multi[3]     = p_mz;
}

// pass 2
// m2l dual-tree traversal to find well-separated nodes and translate expansions
static void ps_impl_fmm_interaction_pass(ps_context_t* ctx, ps_node_t* target,
                                         ps_node_t* src, float theta) {
    if (!target || !src) {
        return;
    }

    float dx      = src->x - target->x;
    float dy      = src->y - target->y;
    float dz      = src->z - target->z;
    float dist_sq = (dx * dx) + (dy * dy) + (dz * dz);

    float hw_sum = target->half_width + src->half_width;

    // well-separated
    if (dist_sq > (theta * theta) * (hw_sum * hw_sum)) {
        float soft_dist_sq = dist_sq + 0.1F;
        float dist         = sqrtf(soft_dist_sq);
        float inv_r3       = 1.0F / (dist * soft_dist_sq);
        float inv_r5       = inv_r3 / soft_dist_sq;

        float* s_multi = ps_impl_get_multipole(ctx, src);
        float  m0      = s_multi[0];
        float  mx      = s_multi[1];
        float  my      = s_multi[2];
        float  mz      = s_multi[3];

        // monopole contribution to target's local field
        float force_m0_x = m0 * dx * inv_r3;
        float force_m0_y = m0 * dy * inv_r3;
        float force_m0_z = m0 * dz * inv_r3;

        // dipole contribution to target's local field
        float m_dot_r      = (mx * dx) + (my * dy) + (mz * dz);
        float dipole_coeff = 3.0F * m_dot_r * inv_r5;

        float force_dip_x = (mx * inv_r3) - (dx * dipole_coeff);
        float force_dip_y = (my * inv_r3) - (dy * dipole_coeff);
        float force_dip_z = (mz * inv_r3) - (dz * dipole_coeff);

        // accumulate into target's local expansion
        float* t_local = ps_impl_get_local(ctx, target);
        t_local[1] += force_m0_x + force_dip_x;
        t_local[2] += force_m0_y + force_dip_y;
        t_local[3] += force_m0_z + force_dip_z;

        return;
    }

    // not well-separated
    if ((target->data.leaf.is_leaf & 1) && (src->data.leaf.is_leaf & 1)) {
        // both are leaves, near-field p2p pass will handle exact dists
        return;
    }

    if (target->data.leaf.is_leaf & 1) {
        // target is as small as possible, open the src
        for (int i = 0; i < PS_OCTANTS; ++i) {
            ps_node_t* src_child =
                ps_impl_get_node(ctx, src->data.children_offs[i]);
            if (!src_child) {
                continue;
            }

            ps_impl_fmm_interaction_pass(ctx, target, src_child, theta);
        }
    } else if (src->data.leaf.is_leaf & 1) {
        // src is as small as possible, open the target
        for (int i = 0; i < PS_OCTANTS; ++i) {
            ps_node_t* target_child =
                ps_impl_get_node(ctx, target->data.children_offs[i]);
            if (!target_child) {
                continue;
            }

            ps_impl_fmm_interaction_pass(ctx, target_child, src, theta);
        }
    } else {
        // subdivide larger to ensure symmetric depth traversal
        if (src->half_width > target->half_width * 1.01F) {
            // source is larger, subdivide source only
            for (int i = 0; i < PS_OCTANTS; ++i) {
                ps_node_t* src_child =
                    ps_impl_get_node(ctx, src->data.children_offs[i]);
                if (!src_child) {
                    continue;
                }

                ps_impl_fmm_interaction_pass(ctx, target, src_child, theta);
            }
        } else if (target->half_width > src->half_width * 1.01F) {
            // target is larger, subdivide target only
            for (int i = 0; i < PS_OCTANTS; ++i) {
                ps_node_t* target_child =
                    ps_impl_get_node(ctx, target->data.children_offs[i]);
                if (!target_child) {
                    continue;
                }

                ps_impl_fmm_interaction_pass(ctx, target_child, src, theta);
            }
        } else {
            // same size, subdivide both and pair permutations
            for (int i = 0; i < PS_OCTANTS; ++i) {
                ps_node_t* target_child =
                    ps_impl_get_node(ctx, target->data.children_offs[i]);
                if (!target_child) {
                    continue;
                }

                for (int j = 0; j < PS_OCTANTS; ++j) {
                    ps_node_t* src_child =
                        ps_impl_get_node(ctx, src->data.children_offs[j]);
                    if (!src_child) {
                        continue;
                    }

                    ps_impl_fmm_interaction_pass(ctx, target_child, src_child,
                                                 theta);
                }
            }
        }
    }
}

// pass 3/4
// l2l, pushes the accumulated background field from parents down to their
// children
// l2p, applies the accumulated bg field to the particles inside the leaf
static void ps_impl_fmm_downward_pass(ps_context_t* ctx, ps_node_t* node,
                                      const ps_particle_arrs_t* arrs) {
    if (!node) {
        return;
    }

    if (node->data.leaf.is_leaf & 1) {
        float* t_local = ps_impl_get_local(ctx, node);
        float  field_x = t_local[1];
        float  field_y = t_local[2];
        float  field_z = t_local[3];

        uint32_t i = 0;

#if defined(PS_USE_AVX2)
        // broadcast local bg field to all 8 lanes
        __m256 f_x_vec = _mm256_set1_ps(field_x);
        __m256 f_y_vec = _mm256_set1_ps(field_y);
        __m256 f_z_vec = _mm256_set1_ps(field_z);

        // process in chunks of 8
        for (; i + 7 < node->data.leaf.particle_cnt; i += 8) {
            uint32_t idx = node->data.leaf.first_particle_idx + i;

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
            _mm256_storeu_ps(&arrs->fx[idx], cur_fx);
            _mm256_storeu_ps(&arrs->fy[idx], cur_fy);
            _mm256_storeu_ps(&arrs->fz[idx], cur_fz);
        }
#elif defined(PS_USE_NEON)
        // broadcast local bg field to all 4 lanes
        float32x4_t f_x_vec = vdupq_n_f32(field_x);
        float32x4_t f_y_vec = vdupq_n_f32(field_y);
        float32x4_t f_z_vec = vdupq_n_f32(field_z);

        // process in chunks of 4
        for (; i + 3 < node->data.leaf.particle_cnt; i += 4) {
            uint32_t idx = node->data.leaf.first_particle_idx + i;

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
        for (; i < node->data.leaf.particle_cnt; ++i) {
            uint32_t idx  = node->data.leaf.first_particle_idx + i;
            float    mass = arrs->mass[idx];

            // F = m * a
            arrs->fx[idx] += mass * field_x;
            arrs->fy[idx] += mass * field_y;
            arrs->fz[idx] += mass * field_z;
        }

        return;
    }

    for (int i = 0; i < PS_OCTANTS; ++i) {
        ps_node_t* child = ps_impl_get_node(ctx, node->data.children_offs[i]);
        if (!child) {
            continue;
        }

        // 1-st order local expansion,
        // shifting it is adding the parent's field to the child's field
        float*       child_local = ps_impl_get_local(ctx, child);
        const float* node_local  = ps_impl_get_local(ctx, node);
        child_local[1] += node_local[1];
        child_local[2] += node_local[2];
        child_local[3] += node_local[3];

        ps_impl_fmm_downward_pass(ctx, child, arrs);
    }
}

// pass 5
// p2p, dual-tree traversal to calc N-body forces for near-field neighbors
static void ps_impl_fmm_p2p_pass(ps_context_t* ctx, ps_node_t* target,
                                 ps_node_t* src, const ps_particle_arrs_t* arrs,
                                 float theta) {
    if (!target || !src) {
        return;
    }

    float dx      = src->x - target->x;
    float dy      = src->y - target->y;
    float dz      = src->z - target->z;
    float dist_sq = (dx * dx) + (dy * dy) + (dz * dz);

    float hw_sum = target->half_width + src->half_width;

    // if well-separated M2L already handled it
    if (dist_sq > (theta * theta) * (hw_sum * hw_sum)) {
        return;
    }

    // if both are leaves and too close
    // do direct N-body force
    if ((target->data.leaf.is_leaf & 1) && (src->data.leaf.is_leaf & 1)) {
        for (uint32_t i = 0; i < target->data.leaf.particle_cnt; ++i) {
            uint32_t t_idx  = target->data.leaf.first_particle_idx + i;
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
            for (; j + 7 < src->data.leaf.particle_cnt; j += 8) {
                uint32_t s_idx = src->data.leaf.first_particle_idx + j;

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
            float32x4_t eps_vec = vdupq_n_f32(0.1F);

            // accumulators for target particles forces
            float32x4_t f_x_vec = vdupq_n_f32(0.0F);
            float32x4_t f_y_vec = vdupq_n_f32(0.0F);
            float32x4_t f_z_vec = vdupq_n_f32(0.0F);

            // process in chunks of 4
            for (; j + 3 < src->data.leaf.particle_cnt; j += 4) {
                uint32_t s_idx = src->data.leaf.first_particle_idx + j;

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
            for (; j < src->data.leaf.particle_cnt; ++j) {
                uint32_t s_idx = src->data.leaf.first_particle_idx + j;

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
    if (target->data.leaf.is_leaf & 1) {
        for (int i = 0; i < PS_OCTANTS; ++i) {
            ps_node_t* src_child =
                ps_impl_get_node(ctx, src->data.children_offs[i]);
            if (!src_child) {
                continue;
            }

            ps_impl_fmm_p2p_pass(ctx, target, src_child, arrs, theta);
        }
    } else if (src->data.leaf.is_leaf & 1) {
        for (int i = 0; i < PS_OCTANTS; ++i) {
            ps_node_t* target_child =
                ps_impl_get_node(ctx, target->data.children_offs[i]);
            if (!target_child) {
                continue;
            }

            ps_impl_fmm_p2p_pass(ctx, target_child, src, arrs, theta);
        }
    } else {
        if (src->half_width > target->half_width * 1.01F) {
            // source is larger, subdivice source only
            for (int i = 0; i < PS_OCTANTS; ++i) {
                ps_node_t* src_child =
                    ps_impl_get_node(ctx, src->data.children_offs[i]);
                if (!src_child) {
                    continue;
                }

                ps_impl_fmm_p2p_pass(ctx, target, src_child, arrs, theta);
            }
        } else if (target->half_width > src->half_width * 1.01F) {
            // target is larger, subdivide target only
            for (int i = 0; i < PS_OCTANTS; ++i) {
                ps_node_t* target_child =
                    ps_impl_get_node(ctx, target->data.children_offs[i]);
                if (!target_child) {
                    continue;
                }

                ps_impl_fmm_p2p_pass(ctx, target_child, src, arrs, theta);
            }
        } else {
            // same size, subdivide both
            for (int i = 0; i < PS_OCTANTS; ++i) {
                ps_node_t* target_child =
                    ps_impl_get_node(ctx, target->data.children_offs[i]);
                if (!target_child) {
                    continue;
                }

                for (int j = 0; j < PS_OCTANTS; ++j) {
                    ps_node_t* src_child =
                        ps_impl_get_node(ctx, src->data.children_offs[j]);
                    if (!src_child) {
                        continue;
                    }

                    ps_impl_fmm_p2p_pass(ctx, target_child, src_child, arrs,
                                         theta);
                }
            }
        }
    }
}

// generates a histogram for a threads chunk of sorted memory
static void ps_impl_radix_map(ps_context_t* ctx, uint32_t chunk_id,
                              size_t start_idx, size_t end_idx) {
    int             shift = ctx->radix_state.pass * 8;
    const uint32_t* m_src = ctx->radix_state.m_src;
    uint32_t*       hist  = ctx->radix_state.hists[chunk_id];

    // count frequencies for this threads chunk
    for (size_t i = start_idx; i < end_idx; ++i) {
        uint8_t bucket = (uint8_t)((m_src[i] >> shift) & 0xFF);
        hist[bucket]++;
    }
}

static void ps_impl_radix_scatter(ps_context_t* ctx, uint32_t chunk_id,
                                  size_t start_idx, size_t end_idx) {
    int               shift = ctx->radix_state.pass * 8;
    ps_radix_state_t* rs    = &ctx->radix_state;
    ps_scatter_buf_t* buf   = &rs->bufs[chunk_id];

    // reset buffer counts for this pass
    for (int i = 0; i < 256; ++i) {
        buf->cnt[i] = 0;
    }

    // make a local copy of this threads start offsets
    // so we can increment them safely in regs
    uint32_t local_offs[256];
    for (int i = 0; i < 256; ++i) {
        local_offs[i] = rs->offs[chunk_id][i];
    }

    for (size_t i = start_idx; i < end_idx; ++i) {
        uint8_t bucket = (uint8_t)((rs->m_src[i] >> shift) & 0xFF);
        uint8_t c      = buf->cnt[bucket];

        buf->m[bucket][c]    = rs->m_src[i];
        buf->id[bucket][c]   = rs->id_src[i];
        buf->x[bucket][c]    = rs->x_src[i];
        buf->y[bucket][c]    = rs->y_src[i];
        buf->z[bucket][c]    = rs->z_src[i];
        buf->mass[bucket][c] = rs->mass_src[i];

        c++;

        // if buffer is full, flush to main memory in one
        if (c == PS_RADIX_BUF_SIZE) {
            uint32_t dst_idx = local_offs[bucket];

            for (int j = 0; j < PS_RADIX_BUF_SIZE; ++j) {
                rs->m_dst[dst_idx + j]    = buf->m[bucket][j];
                rs->id_dst[dst_idx + j]   = buf->id[bucket][j];
                rs->x_dst[dst_idx + j]    = buf->x[bucket][j];
                rs->y_dst[dst_idx + j]    = buf->y[bucket][j];
                rs->z_dst[dst_idx + j]    = buf->z[bucket][j];
                rs->mass_dst[dst_idx + j] = buf->mass[bucket][j];
            }

            local_offs[bucket] += PS_RADIX_BUF_SIZE;
            buf->cnt[bucket] = 0;
        } else {
            buf->cnt[bucket] = c;
        }
    }

    for (int bucket = 0; bucket < 256; ++bucket) {
        uint8_t rem = buf->cnt[bucket];

        if (rem > 0) {
            uint32_t dst_idx = local_offs[bucket];

            for (int j = 0; j < rem; ++j) {
                rs->m_dst[dst_idx + j]    = buf->m[bucket][j];
                rs->id_dst[dst_idx + j]   = buf->id[bucket][j];
                rs->x_dst[dst_idx + j]    = buf->x[bucket][j];
                rs->y_dst[dst_idx + j]    = buf->y[bucket][j];
                rs->z_dst[dst_idx + j]    = buf->z[bucket][j];
                rs->mass_dst[dst_idx + j] = buf->mass[bucket][j];
            }

            local_offs[bucket] += rem;
        }
    }
}

#define PS_IMPL_SWAP_PTR(type, a, b)                                           \
    do {                                                                       \
        type tmp = a;                                                          \
        (a)      = b;                                                          \
        (b)      = tmp;                                                        \
    } while (0)

static ps_result_t ps_impl_sort_particles(ps_context_t* ctx,
                                          uint32_t*     morton_codes,
                                          const ps_particle_arrs_t* arrs) {
    size_t cnt = arrs->cnt;
    if (cnt == 0) {
        return PS_OK;
    }

    ps_radix_state_t* rs = &ctx->radix_state;
    rs->m_src            = morton_codes;
    rs->id_src           = arrs->id;
    rs->x_src            = arrs->x;
    rs->y_src            = arrs->y;
    rs->z_src            = arrs->z;
    rs->mass_src         = arrs->mass;

    // how many chunks split into
#ifdef PS_MULTITHREADING
    uint32_t num_chunks = (ctx->pool.thrd_cnt > 0) ? ctx->pool.thrd_cnt : 1;
#else
    uint32_t num_chunks = 1;
#endif

#define DEPLOY_JOB(job_type)                                                   \
    for (uint32_t i = 0; i < num_chunks; ++i) {                                \
        size_t start = i * chunk_size;                                         \
        size_t end   = start + chunk_size;                                     \
        if (end > cnt) {                                                       \
            end = cnt;                                                         \
        }                                                                      \
        if (start >= cnt) {                                                    \
            break;                                                             \
        }                                                                      \
        ps_job_t job;                                                          \
        job.type                 = job_type;                                   \
        job.data.array.start_idx = (uint32_t)start;                            \
        job.data.array.end_idx   = (uint32_t)end;                              \
        job.data.array.chunk_id  = i;                                          \
        ps_impl_pool_submit(ctx, job);                                         \
    }

    // ceiling div so last chunk takes remainder
    size_t chunk_size = (cnt + num_chunks - 1) / num_chunks;

    size_t pre_sort_off = ctx->arena.off;
    rs->bufs            = (ps_scatter_buf_t*)ps_impl_arena_alloc(
        ctx, num_chunks * sizeof(ps_scatter_buf_t));

    if (!rs->bufs) {
        ctx->arena.off = pre_sort_off;
        return PS_EOOM;
    }

    // 4 passes of 8b radix sort
    for (int pass = 0; pass < 4; ++pass) {
        rs->pass = (uint8_t)pass;

        // clear histograms on main thread
        for (int bucket = 0; bucket < 256; ++bucket) {
            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                rs->hists[chunk][bucket] = 0;
            }
        }

        // local histograms/map
        DEPLOY_JOB(PS_JOB_RADIX_MAP);

        // wait for all threads to finish mapping
        ps_impl_pool_wait(ctx);

        // reduce
        // calculates exactly where each thread should start writing for each
        // bucket
        uint32_t run_off = 0;
        for (int bucket = 0; bucket < 256; ++bucket) {
            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                rs->offs[chunk][bucket] = run_off;
                run_off += rs->hists[chunk][bucket];
            }
        }

        // scatter
        DEPLOY_JOB(PS_JOB_RADIX_SCATTER);

        // wait for all data to be scattered
        ps_impl_pool_wait(ctx);

        // ping-pong pointers
        PS_IMPL_SWAP_PTR(uint32_t*, rs->m_src, rs->m_dst);
        PS_IMPL_SWAP_PTR(uint32_t*, rs->id_src, rs->id_dst);
        PS_IMPL_SWAP_PTR(float*, rs->x_src, rs->x_dst);
        PS_IMPL_SWAP_PTR(float*, rs->y_src, rs->y_dst);
        PS_IMPL_SWAP_PTR(float*, rs->z_src, rs->z_dst);
        PS_IMPL_SWAP_PTR(float*, rs->mass_src, rs->mass_dst);
    }

#undef DEPLOY_JOB

    ctx->arena.off = pre_sort_off;

    // 4 is an even number, so m_src is guaranteed to be pointing back to the
    // orig morton_codes arr, and x_src back to arrs->x. the sorted data is
    // right where it started. temp arrays will be cleared next frame.

    return PS_OK;
}

// =====================================================================
// public api implementation
// =====================================================================

ps_result_t ps_init(ps_context_t** out_ctx, const ps_config_t* cfg) {
    if (!cfg || !cfg->buff || cfg->buff_size < sizeof(ps_context_t)) {
        return PS_EINVAL;
    }

    // place ctx at beginning of buffer
    ps_context_t* ctx = (ps_context_t*)cfg->buff;
    ctx->root         = NULL;
    ctx->theta        = (cfg->theta > 0.0F) ? cfg->theta : 1.0F;

    uint32_t total_thrds = (uint32_t)cfg->thrd_cnt;
    if (total_thrds > PS_MAX_THRDS + 1) {
        total_thrds = PS_MAX_THRDS + 1;
    } else if (total_thrds < 1) {
        total_thrds = 1;
    }

    // calculate space needed for temp radix arrs
    size_t radix_tmp_size = 6 * cfg->max_particles * sizeof(float);

    // ensure we have enough mem
    size_t arena_start = ps_impl_align_forward(sizeof(ps_context_t), 64);
    if (arena_start + radix_tmp_size > cfg->buff_size) {
        return PS_EOOM;
    }

    // temp arrs go at end of the buffer
    uint8_t* radix_mem = (uint8_t*)cfg->buff + cfg->buff_size - radix_tmp_size;

    // wire permanent ptrs to radix_state
    ctx->radix_state.m_dst =
        (uint32_t*)(radix_mem + (0 * cfg->max_particles * sizeof(float)));
    ctx->radix_state.id_dst =
        (uint32_t*)(radix_mem + (1 * cfg->max_particles * sizeof(float)));
    ctx->radix_state.x_dst =
        (float*)(radix_mem + (2 * cfg->max_particles * sizeof(float)));
    ctx->radix_state.y_dst =
        (float*)(radix_mem + (3 * cfg->max_particles * sizeof(float)));
    ctx->radix_state.z_dst =
        (float*)(radix_mem + (4 * cfg->max_particles * sizeof(float)));
    ctx->radix_state.mass_dst =
        (float*)(radix_mem + (5 * cfg->max_particles * sizeof(float)));

    size_t par_arrs_size = 2 * (PS_EXPANSION_TERMS * sizeof(float));
    size_t footprint     = sizeof(ps_node_t) + par_arrs_size;
    size_t usable        = cfg->buff_size - arena_start - radix_tmp_size;
    size_t max_nodes     = usable / footprint;

    // global arena gets remaining space
    ctx->arena.mem = (uint8_t*)cfg->buff + arena_start;
    ctx->arena.cap = max_nodes * sizeof(ps_node_t);
    ctx->arena.off = 0;

    // clang-format off

    ctx->local_exp = (float (*)[PS_EXPANSION_TERMS])(ctx->arena.mem + ctx->arena.cap);
    ctx->multipole_exp = (float (*)[PS_EXPANSION_TERMS])((uint8_t*)ctx->local_exp + (max_nodes * PS_EXPANSION_TERMS * sizeof(float)));

    // clang-format on

#ifdef PS_MULTITHREADING

    // initialize the pool
    if (ps_impl_pool_init(ctx, total_thrds - 1) != 0) {
        return PS_EALLOC;
    }

#endif // PS_MULTITHREADING

    *out_ctx = ctx;
    return PS_OK;
}

#ifdef PS_MULTITHREADING

// dispatches target nodes for M2L/P2P passes
static void ps_impl_dispatch_fmm(ps_context_t* ctx, ps_job_type_t job_type,
                                 ps_node_t* target, ps_node_t* src,
                                 const ps_particle_arrs_t* arrs, int curr_depth,
                                 int target_depth) {
    if (!target) {
        return;
    }

    if (curr_depth == target_depth || (target->data.leaf.is_leaf & 1)) {
        ps_job_t job;
        job.type            = job_type;
        job.data.fmm.target = target;
        job.data.fmm.src    = src;
        job.data.fmm.arrs   = arrs;
        ps_impl_pool_submit(ctx, job);
        return;
    }

    // otherwise keep traversing down
    for (int i = 0; i < PS_OCTANTS; ++i) {
        ps_node_t* target_child =
            ps_impl_get_node(ctx, target->data.children_offs[i]);
        if (!target_child) {
            continue;
        }

        ps_impl_dispatch_fmm(ctx, job_type, target_child, src, arrs,
                             curr_depth + 1, target_depth);
    }
}

// dispatches nodes for downward pass
static void ps_impl_dispatch_downward(ps_context_t* ctx, ps_node_t* target,
                                      const ps_particle_arrs_t* arrs,
                                      int curr_depth, int target_depth) {
    if (!target) {
        return;
    }

    if (curr_depth == target_depth || (target->data.leaf.is_leaf & 1)) {
        ps_job_t job;
        job.type          = PS_JOB_DOWNWARD;
        job.data.fmm.src  = target;
        job.data.fmm.arrs = arrs;
        ps_impl_pool_submit(ctx, job);
        return;
    }

    // above target depth, manually push local field down before dispatch
    for (int i = 0; i < PS_OCTANTS; ++i) {
        ps_node_t* child = ps_impl_get_node(ctx, target->data.children_offs[i]);
        if (!child) {
            continue;
        }

        float* child_local  = ps_impl_get_local(ctx, child);
        float* target_local = ps_impl_get_local(ctx, target);

        child_local[1] += target_local[1];
        child_local[2] += target_local[2];
        child_local[3] += target_local[3];
        ps_impl_dispatch_downward(ctx, child, arrs, curr_depth + 1,
                                  target_depth);
    }
}

#endif // PS_MULTITHREADING

ps_result_t ps_calc_forces(ps_context_t* ctx, const ps_particle_arrs_t* arrs,
                           uint32_t* morton_codes, float root_cx, float root_cy,
#ifndef PS_2D
                           float root_cz,
#endif // PS_3D
                           float root_hw) {
    if (!ctx || !arrs) {
        return PS_EINVAL;
    }

    // clear arena for new frame
    ps_impl_arena_clear(&ctx->arena);

    // sort the arrays based on morton_codes as keys
    ps_result_t res = ps_impl_sort_particles(ctx, morton_codes, arrs);
    if (res != PS_OK) {
        return res;
    }

    // allocate the root node
    ctx->root = ps_impl_alloc_node(ctx);

    // seed geometry
    ctx->root->x = root_cx;
    ctx->root->y = root_cy;
#ifndef PS_2D
    ctx->root->z = root_cz;
#endif // PS_3D
    ctx->root->half_width = root_hw;

    // seed the radix state so the workers can read the morton codes
    ctx->radix_state.m_src = morton_codes;

    for (size_t i = 0; i < arrs->cnt; ++i) {
        arrs->fx[i] = 0.0F;
        arrs->fy[i] = 0.0F;
#ifndef PS_2D
        arrs->fz[i] = 0.0F;
#endif // PS_3D
    }

    // build the octree
    ps_impl_build_tree(ctx, 0, ctx->root, 0, 0, arrs->cnt, morton_codes);
    ps_impl_pool_wait(ctx);

    // upward pass (p2m -> m2m)
    ps_impl_fmm_upward_pass(ctx, ctx->root, arrs);

// interaction pass (m2l)
#ifdef PS_MULTITHREADING
    int dispatch_depth = 2;
    ps_impl_dispatch_fmm(ctx, PS_JOB_INTERACTION, ctx->root, ctx->root, arrs, 0,
                         dispatch_depth);
    ps_impl_pool_wait(ctx);
#else
    ps_impl_fmm_interaction_pass(ctx, ctx->root, ctx->root, ctx->theta);
#endif

    // downward pass (l2l -> l2p)
#ifdef PS_MULTITHREADING
    ps_impl_dispatch_downward(ctx, ctx->root, arrs, 0, dispatch_depth);
    ps_impl_pool_wait(ctx);
#else
    ps_impl_fmm_downward_pass(ctx, ctx->root, arrs);
#endif

// evaluation pass (p2p)
#ifdef PS_MULTITHREADING
    ps_impl_dispatch_fmm(ctx, PS_JOB_P2P, ctx->root, ctx->root, arrs, 0,
                         dispatch_depth);
    ps_impl_pool_wait(ctx);
#else
    ps_impl_fmm_p2p_pass(ctx, ctx->root, ctx->root, arrs, ctx->theta);
#endif

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

#ifdef PS_2D

#if defined(PS_USE_AVX2)

    __m256 v_min = _mm256_set1_ps(FLT_MAX);
    __m256 v_max = _mm256_set1_ps(-FLT_MAX);

    for (; i + 7 < arrs->cnt; i += 8) {
        __m256 vx = _mm256_loadu_ps(&arrs->x[i]);
        __m256 vy = _mm256_loadu_ps(&arrs->y[i]);

        // find the local min/max for x and y within this chunk
        __m256 v_cmin = _mm256_min_ps(vx, vy);
        __m256 v_cmax = _mm256_max_ps(vx, vy);

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

        // find the local min/max for x and y within this chunk
        float32x4_t v_cmin = vminq_f32(vx, vy);
        float32x4_t v_cmax = vmaxq_f32(vx, vy);

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

#endif // PS_USE_NEON

    // fallback
    // find rect bb bounds
    for (; i < arrs->cnt; i++) {
        if (arrs->x[i] < min_b) {
            min_b = arrs->x[i];
        }
        if (arrs->y[i] < min_b) {
            min_b = arrs->y[i];
        }

        if (arrs->x[i] > max_b) {
            max_b = arrs->x[i];
        }
        if (arrs->y[i] > max_b) {
            max_b = arrs->y[i];
        }
    }

#else // PS_3D

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

#endif // PS_USE_NEON

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

#endif // PS_3D

    float range = max_b - min_b;
    if (range < 0.001F) {
        range = 0.001F;
    }

    i = 0;

#ifdef PS_2D

#if defined(PS_USE_AVX2)

    __m256 v_min_b = _mm256_set1_ps(min_b);
    __m256 v_scale = _mm256_set1_ps(65535.0F / range);
    __m256 v_zero  = _mm256_setzero_ps();
    __m256 v_65535 = _mm256_set1_ps(65535.0F);

    // consts for morton expansion
    __m256i m_0000FFFF = _mm256_set1_epi32(0x0000FFFF);
    __m256i m_00FF00FF = _mm256_set1_epi32(0x00FF00FF);
    __m256i m_0F0F0F0F = _mm256_set1_epi32(0x0F0F0F0F);
    __m256i m_33333333 = _mm256_set1_epi32(0x33333333);
    __m256i m_55555555 = _mm256_set1_epi32(0x55555555);

// bit exp macro
#define PS_EXPAND_AXV2(v)                                                      \
    (v) = _mm256_and_si256(v, m_0000FFFF);                                     \
    (v) = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 8)),        \
                           m_00FF00FF);                                        \
    (v) = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 4)),        \
                           m_0F0F0F0F);                                        \
    (v) = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 2)),        \
                           m_33333333);                                        \
    (v) = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 1)),        \
                           m_55555555);

    for (; i + 7 < arrs->cnt; i += 8) {
        __m256 vx = _mm256_loadu_ps(&arrs->x[i]);
        __m256 vy = _mm256_loadu_ps(&arrs->y[i]);

        // map to 0-65535
        vx = _mm256_mul_ps(_mm256_sub_ps(vx, v_min_b), v_scale);
        vy = _mm256_mul_ps(_mm256_sub_ps(vy, v_min_b), v_scale);

        // clamp to 0-65535
        vx = _mm256_max_ps(v_zero, _mm256_min_ps(vx, v_65535));
        vy = _mm256_max_ps(v_zero, _mm256_min_ps(vy, v_65535));

        // convert to int
        __m256i mx = _mm256_cvtps_epi32(vx);
        __m256i my = _mm256_cvtps_epi32(vy);

        // expand bits for morton encoding
        PS_EXPAND_AXV2(mx);
        PS_EXPAND_AXV2(my);

        // interleave bits and store morton codes
        // ix | (iy << 1)
        __m256i morton_codes_vec =
            _mm256_or_si256(mx, _mm256_slli_epi32(my, 1));

        _mm256_storeu_si256((__m256i*)&out_morton_codes[i], morton_codes_vec);
    }
#undef PS_EXPAND_AXV2

#elif defined(PS_USE_NEON)

    float32x4_t v_min_b = vdupq_n_f32(min_b);
    float32x4_t v_scale = vdupq_n_f32(65535.0F / range);
    float32x4_t v_zero  = vdupq_n_f32(0.0F);
    float32x4_t v_65535 = vdupq_n_f32(65535.0F);

    // consts for morton expansion
    int32x4_t m_0000FFFF = vdupq_n_s32(0x0000FFFF);
    int32x4_t m_00FF00FF = vdupq_n_s32(0x00FF00FF);
    int32x4_t m_0F0F0F0F = vdupq_n_s32(0x0F0F0F0F);
    int32x4_t m_33333333 = vdupq_n_s32(0x33333333);
    int32x4_t m_55555555 = vdupq_n_s32(0x55555555);

// bit exp macro
#define PS_EXPAND_NEON(v)                                                      \
    (v) = vandq_s32(v, m_0000FFFF);                                            \
    (v) = vandq_s32(vorrq_s32(v, vshlq_n_s32(v, 8)), m_00FF00FF);              \
    (v) = vandq_s32(vorrq_s32(v, vshlq_n_s32(v, 4)), m_0F0F0F0F);              \
    (v) = vandq_s32(vorrq_s32(v, vshlq_n_s32(v, 2)), m_33333333);              \
    (v) = vandq_s32(vorrq_s32(v, vshlq_n_s32(v, 1)), m_55555555);

    for (; i + 3 < arrs->cnt; i += 4) {
        float32x4_t vx = vld1q_f32(&arrs->x[i]);
        float32x4_t vy = vld1q_f32(&arrs->y[i]);

        // map to 0-65535
        vx = vmulq_f32(vsubq_f32(vx, v_min_b), v_scale);
        vy = vmulq_f32(vsubq_f32(vy, v_min_b), v_scale);

        // clamp to 0-65535
        vx = vmaxq_f32(v_zero, vminq_f32(vx, v_65535));
        vy = vmaxq_f32(v_zero, vminq_f32(vy, v_65535));

        // convert to int
        int32x4_t mx = vcvtq_s32_f32(vx);
        int32x4_t my = vcvtq_s32_f32(vy);

        // expand bits for morton encoding
        PS_EXPAND_NEON(mx);
        PS_EXPAND_NEON(my);

        // interleave bits and store morton codes
        // ix | (iy << 1)
        int32x4_t morton_codes_vec = vorrq_s32(mx, vshlq_n_s32(my, 1));

        vst1q_s32((int32_t*)&out_morton_codes[i], morton_codes_vec);
    }
#undef PS_EXPAND_NEON

#endif // PS_USE_NEON

    // fallback
    // gen morton codes mapped to 0-65535
    for (; i < arrs->cnt; i++) {
        int mx = (int)(((arrs->x[i] - min_b) / range) * 65535.0F);
        int my = (int)(((arrs->y[i] - min_b) / range) * 65535.0F);
        int mz = (int)(((arrs->z[i] - min_b) / range) * 65535.0F);

        if (mx < 0) {
            mx = 0;
        }
        if (mx > 65535) {
            mx = 65535;
        }
        if (my < 0) {
            my = 0;
        }
        if (my > 65535) {
            my = 65535;
        }

        out_morton_codes[i] = ps_impl_morton_encode(mx, my);
    }

#else // PS_3D

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

#endif // PS_USE_NEON

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

#endif // PS_3D

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

ps_result_t ps_destroy(ps_context_t* ctx) {
    if (!ctx) {
        return PS_EINVAL;
    }

#ifdef PS_MULTITHREADING

    ps_thrd_pool_t* pool = &ctx->pool;

    ps_spin_lock(&pool->lock);
    pool->shutdown_flag = 1;
    ps_spin_unlock(&pool->lock);

    for (uint32_t i = 0; i < pool->thrd_cnt; ++i) {
        ps_thrd_join(pool->thrds[i]);
    }

#endif // PS_MULTITHREADING

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
