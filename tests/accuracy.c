#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PS_MULTITHREADING
#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#define ARENA_SIZE   (1024ULL * 1024ULL * 32ULL)
#define SOFTENING_SQ 0.1F
#define THETA        3.4641F

static int g_failures = 0;

static void failf(const char* name, double actual, double expected,
                  double tolerance) {
    (void)fprintf(stderr,
                  "[FAIL] %s: actual=%.9g expected=%.9g tolerance=%.9g\n", name,
                  actual, expected, tolerance);
    ++g_failures;
}

static void check_close(const char* name, double actual, double expected,
                        double tolerance) {
    if (fabs(actual - expected) > tolerance) {
        failf(name, actual, expected, tolerance);
    } else {
        (void)printf("[PASS] %s: %.9g\n", name, actual);
    }
}

static uint32_t lcg_next(uint32_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state;
}

static float rand_unit(uint32_t* state) {
    return (float)(lcg_next(state) >> 8) / 16777215.0F;
}

static void direct_reference(const float* x, const float* y, const float* z,
                             const float* mass, size_t cnt, double* fx,
                             double* fy, double* fz) {
    for (size_t i = 0; i < cnt; ++i) {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;

        for (size_t j = 0; j < cnt; ++j) {
            if (i == j) {
                continue;
            }

            double dx     = (double)x[j] - (double)x[i];
            double dy     = (double)y[j] - (double)y[i];
            double dz     = (double)z[j] - (double)z[i];
            double r2     = (dx * dx) + (dy * dy) + (dz * dz) + SOFTENING_SQ;
            double inv_r  = 1.0 / sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double scale  = (double)mass[i] * (double)mass[j] * inv_r3;

            ax += dx * scale;
            ay += dy * scale;
            az += dz * scale;
        }

        fx[i] = ax;
        fy[i] = ay;
        fz[i] = az;
    }
}

static void run_dipole_sign_test(void) {
    void*         arena = malloc(ARENA_SIZE);
    ps_context_t* ctx   = NULL;
    ps_config_t   cfg   = {arena, ARENA_SIZE, 2, THETA, 1};
    ps_init(&ctx, &cfg);

    ps_node_t* target = ps_impl_alloc_node(ctx);
    ps_node_t* source = ps_impl_alloc_node(ctx);

    target->x          = 0.0F;
    target->y          = 0.0F;
    target->z          = 0.0F;
    target->half_width = 1.0F;

    source->x          = 100.0F;
    source->y          = 0.0F;
    source->z          = 0.0F;
    source->half_width = 1.0F;

    /* One unit mass displaced +1 from the source-cell center. */
    float* s_multi = ps_impl_get_multipole(ctx, source);
    s_multi[0]     = 1.0F;
    s_multi[1]     = 1.0F;

    ps_impl_fmm_interaction_pass(ctx, target, source, THETA);

    /*
     * First-order softened expansion for R=100 and delta=+1 is
     * approximately 9.799856e-05. The large separation keeps this
     * analytic check independent of the production MAC threshold.
     */
    float* target_local = ps_impl_get_local(ctx, target);
    check_close("M2L dipole orientation", target_local[1], 0.00009799856, 5e-9);

    ps_destroy(ctx);
    free(arena);
}

static void fill_uniform(float* x, float* y, float* z, float* mass,
                         uint32_t* id, size_t cnt) {
    uint32_t state = 7331U;
    for (size_t i = 0; i < cnt; ++i) {
        x[i]    = (rand_unit(&state) * 100.0F) - 50.0F;
        y[i]    = (rand_unit(&state) * 100.0F) - 50.0F;
        z[i]    = (rand_unit(&state) * 100.0F) - 50.0F;
        mass[i] = 0.5F + (rand_unit(&state) * 1.5F);
        id[i]   = (uint32_t)i;
    }
}

static void fill_two_clusters(float* x, float* y, float* z, float* mass,
                              uint32_t* id, size_t cnt) {
    uint32_t state = 20260803U;
    size_t   half  = cnt / 2;

    for (size_t i = 0; i < cnt; ++i) {
        float center = i < half ? -12.0F : 12.0F;
        float skew   = i < half ? -0.75F : 0.75F;
        x[i]         = center + skew + ((rand_unit(&state) - 0.5F) * 2.0F);
        y[i]         = (rand_unit(&state) - 0.5F) * 2.0F;
        z[i]         = (rand_unit(&state) - 0.5F) * 2.0F;
        mass[i]      = 0.5F + (rand_unit(&state) * 2.0F);
        id[i]        = (uint32_t)i;
    }
}

static int comp_arrs(const char* phase, size_t cnt, ps_particle_arrs_t* st,
                     ps_particle_arrs_t* mt, const uint32_t* morton_st,
                     const uint32_t* morton_mt) {
    int mismatches = 0;
    for (size_t i = 0; i < cnt; ++i) {
        if (st->id[i] != mt->id[i] || morton_st[i] != morton_mt[i]) {
            printf(
                "[%s] Order mismatch at array index %zu: ST_ID=%u MT_ID=%u\n",
                phase, i, st->id[i], mt->id[i]);
            mismatches++;
        }
        if (fabsf(st->fx[i] - mt->fx[i]) > 1e-4F ||
            fabsf(st->fy[i] - mt->fy[i]) > 1e-4F ||
            fabsf(st->fz[i] - mt->fz[i]) > 1e-4F) {
            printf("[%s] Force mismatch at array index %zu (ID %u): "
                   "ST(%.2f,%.2f,%.2f) MT(%.2f,%.2f,%.2f)",
                   phase, i, st->id[i], st->fx[i], st->fy[i], st->fz[i],
                   mt->fx[i], mt->fy[i], mt->fz[i]);
            mismatches++;

            if (mismatches >= 5) {
                break;
            }
        }
    }

    if (mismatches > 0) {
        return 0;
    }

    printf("[PASS] %s\n", phase);

    return 1;
}

static int comp_trees_rec(ps_context_t* ctx_st, ps_context_t* ctx_mt,
                          ps_node_t* a, ps_node_t* b, int depth,
                          const char* phase) {
    if (!a && !b) {
        return 1;
    }

    if (!a || !b) {
        printf("[%s] Topology mismatch at depth %d: one node is NULL\n", phase,
               depth);
        return 0;
    }

    if (a->data.leaf.is_leaf & 1) {
        if (a->data.leaf.particle_cnt != b->data.leaf.particle_cnt ||
            a->data.leaf.first_particle_idx !=
                b->data.leaf.first_particle_idx) {
            printf("[%s] Leaf data mismatch at depth %d. ST(cnt=%u, idx=%u) "
                   "MT(cnt=%u, idx=%u)\n",
                   phase, depth, a->data.leaf.particle_cnt,
                   a->data.leaf.first_particle_idx, b->data.leaf.particle_cnt,
                   b->data.leaf.first_particle_idx);
            return 0;
        }
    }

    float* a_multi = ps_impl_get_multipole(ctx_st, a);
    float* b_multi = ps_impl_get_multipole(ctx_mt, b);
    float* a_local = ps_impl_get_local(ctx_st, a);
    float* b_local = ps_impl_get_local(ctx_mt, b);
    for (int i = 0; i < 4; ++i) {
        if (fabsf(a_multi[i] - b_multi[i]) > 1e-4F) {
            printf(
                "[%s] Multipole mismatch at depth %d, idx %d: ST=%f, MT=%f\n",
                phase, depth, i, a_multi[i], b_multi[i]);
            return 0;
        }

        if (fabsf(a_local[i] - b_local[i]) > 1e-4F) {
            printf(
                "[%s] Local field mismatch at depth %d, idx %d: ST=%f, MT=%f\n",
                phase, depth, i, a_local[i], b_local[i]);
            return 0;
        }
    }

    if (!(a->data.leaf.is_leaf & 1)) {
        for (int i = 0; i < 8; ++i) {
            ps_node_t* a_child =
                ps_impl_get_node(ctx_st, a->data.children_offs[i]);
            ps_node_t* b_child =
                ps_impl_get_node(ctx_mt, b->data.children_offs[i]);
            if (!comp_trees_rec(ctx_st, ctx_mt, a_child, b_child, depth + 1,
                                phase)) {
                return 0;
            }
        }
    }

    return 1;
}

static void run_st_vs_mt_test(void) {
    size_t cnt = 512;

    float* x_st = (float*)malloc(cnt * sizeof(float));
    float* y_st = (float*)malloc(cnt * sizeof(float));
    float* z_st = (float*)malloc(cnt * sizeof(float));
    float* m_st = (float*)malloc(cnt * sizeof(float));

    float* x_mt = (float*)malloc(cnt * sizeof(float));
    float* y_mt = (float*)malloc(cnt * sizeof(float));
    float* z_mt = (float*)malloc(cnt * sizeof(float));
    float* m_mt = (float*)malloc(cnt * sizeof(float));

    float*    fx_st = (float*)calloc(cnt, sizeof(float));
    float*    fy_st = (float*)calloc(cnt, sizeof(float));
    float*    fz_st = (float*)calloc(cnt, sizeof(float));
    uint32_t* id_st = (uint32_t*)malloc(cnt * sizeof(uint32_t));
    uint32_t* mo_st = (uint32_t*)malloc(cnt * sizeof(uint32_t));

    float*    fx_mt = (float*)calloc(cnt, sizeof(float));
    float*    fy_mt = (float*)calloc(cnt, sizeof(float));
    float*    fz_mt = (float*)calloc(cnt, sizeof(float));
    uint32_t* id_mt = (uint32_t*)malloc(cnt * sizeof(uint32_t));
    uint32_t* mo_mt = (uint32_t*)malloc(cnt * sizeof(uint32_t));

    fill_two_clusters(x_st, y_st, z_st, m_st, id_st, cnt);
    for (size_t i = 0; i < cnt; ++i) {
        x_mt[i]  = x_st[i];
        y_mt[i]  = y_st[i];
        z_mt[i]  = z_st[i];
        m_mt[i]  = m_st[i];
        id_mt[i] = id_st[i];
    }

    ps_particle_arrs_t arrs_st = {x_st,  y_st,  z_st,  m_st, fx_st,
                                  fy_st, fz_st, id_st, cnt};
    ps_particle_arrs_t arrs_mt = {x_mt,  y_mt,  z_mt,  m_mt, fx_mt,
                                  fy_mt, fz_mt, id_mt, cnt};

    void* arena_st = malloc(ARENA_SIZE);
    void* arena_mt = malloc(ARENA_SIZE);

    ps_context_t* ctx_st = NULL;
    ps_context_t* ctx_mt = NULL;
    ps_config_t   cfg_st = {arena_st, ARENA_SIZE, cnt, THETA, 1};
    ps_config_t   cfg_mt = {arena_mt, ARENA_SIZE, cnt, THETA, 4};

    ps_init(&ctx_st, &cfg_st);
    ps_init(&ctx_mt, &cfg_mt);

    float min_b, max_b, range;
    ps_prepare_particles(&arrs_st, mo_st, &min_b, &max_b, &range);
    ps_prepare_particles(&arrs_mt, mo_mt, &min_b, &max_b, &range);
    float rc = min_b + (range * 0.5F);
    float hw = range * 0.5F;

    ps_impl_arena_clear(&ctx_st->arena);
    ps_impl_arena_clear(&ctx_mt->arena);

    ps_impl_sort_particles(ctx_st, mo_st, &arrs_st);
    ps_impl_sort_particles(ctx_mt, mo_mt, &arrs_mt);
    if (!comp_arrs("RADIX SORT", cnt, &arrs_st, &arrs_mt, mo_st, mo_mt)) {
        return;
    }

    // allocate the root node
    ctx_st->root = ps_impl_alloc_node(ctx_st);
    ctx_mt->root = ps_impl_alloc_node(ctx_mt);

    // seed geometry
    ctx_st->root->x          = rc;
    ctx_st->root->y          = rc;
    ctx_st->root->z          = rc;
    ctx_st->root->half_width = hw;

    ctx_mt->root->x          = rc;
    ctx_mt->root->y          = rc;
    ctx_mt->root->z          = rc;
    ctx_mt->root->half_width = hw;

    // seed the radix state so the workers can read the morton codes
    ctx_st->radix_state.m_src = mo_st;
    ctx_mt->radix_state.m_src = mo_mt;

    for (size_t i = 0; i < cnt; ++i) {
        arrs_st.fx[i] = 0.0F;
        arrs_st.fy[i] = 0.0F;
        arrs_st.fz[i] = 0.0F;
    }
    for (size_t i = 0; i < cnt; ++i) {
        arrs_mt.fx[i] = 0.0F;
        arrs_mt.fy[i] = 0.0F;
        arrs_mt.fz[i] = 0.0F;
    }

    // build the octree
    ps_impl_build_tree(ctx_st, 0, ctx_st->root, 0, 0, cnt, mo_st);
    ps_impl_pool_wait(ctx_st);
    ps_impl_build_tree(ctx_mt, 0, ctx_mt->root, 0, 0, cnt, mo_mt);
    ps_impl_pool_wait(ctx_mt);

    if (!comp_trees_rec(ctx_st, ctx_mt, ctx_st->root, ctx_mt->root, 0,
                        "TREE BUILD")) {
        return;
    }

    // upward pass (p2m -> m2m)
    ps_impl_fmm_upward_pass(ctx_st, ctx_st->root, &arrs_st);
    ps_impl_fmm_upward_pass(ctx_mt, ctx_mt->root, &arrs_mt);
    if (!comp_trees_rec(ctx_st, ctx_mt, ctx_st->root, ctx_mt->root, 0,
                        "UPWARD PASS")) {
        return;
    }

    // interaction pass (m2l)
    ps_impl_fmm_interaction_pass(ctx_st, ctx_st->root, ctx_st->root, THETA);
    ps_impl_dispatch_fmm(ctx_mt, PS_JOB_INTERACTION, ctx_mt->root, ctx_mt->root,
                         &arrs_mt, 0, 2);
    ps_impl_pool_wait(ctx_mt);

    // downward pass (l2l -> l2p)
    ps_impl_fmm_downward_pass(ctx_st, ctx_st->root, &arrs_st);
    ps_impl_dispatch_downward(ctx_mt, ctx_mt->root, &arrs_mt, 0, 2);
    ps_impl_pool_wait(ctx_mt);
    if (!comp_arrs("DOWNWARD PASS ARRAYS", cnt, &arrs_st, &arrs_mt, mo_st,
                   mo_mt)) {
        return;
    }

    // evaluation pass (p2p)
    ps_impl_fmm_p2p_pass(ctx_st, ctx_st->root, ctx_st->root, &arrs_st, THETA);
    ps_impl_dispatch_fmm(ctx_mt, PS_JOB_P2P, ctx_mt->root, ctx_mt->root,
                         &arrs_mt, 0, 2);
    ps_impl_pool_wait(ctx_mt);
    if (!comp_arrs("P2P PASS", cnt, &arrs_st, &arrs_mt, mo_st, mo_mt)) {
        return;
    }

    ps_destroy(ctx_st);
    ps_destroy(ctx_mt);
    printf("[SUCCESS] ST/MT pipeline match.\n");
}

static double run_accuracy_case(const char* name, size_t cnt,
                                void (*fill)(float*, float*, float*, float*,
                                             uint32_t*, size_t)) {
    float*    x      = (float*)malloc(cnt * sizeof(float));
    float*    y      = (float*)malloc(cnt * sizeof(float));
    float*    z      = (float*)malloc(cnt * sizeof(float));
    float*    mass   = (float*)malloc(cnt * sizeof(float));
    float*    fx     = (float*)calloc(cnt, sizeof(float));
    float*    fy     = (float*)calloc(cnt, sizeof(float));
    float*    fz     = (float*)calloc(cnt, sizeof(float));
    uint32_t* id     = (uint32_t*)malloc(cnt * sizeof(uint32_t));
    uint32_t* morton = (uint32_t*)malloc(cnt * sizeof(uint32_t));
    double*   ref_fx = (double*)malloc(cnt * sizeof(double));
    double*   ref_fy = (double*)malloc(cnt * sizeof(double));
    double*   ref_fz = (double*)malloc(cnt * sizeof(double));
    void*     arena  = malloc(ARENA_SIZE);

    if (!x || !y || !z || !mass || !fx || !fy || !fz || !id || !morton ||
        !ref_fx || !ref_fy || !ref_fz || !arena) {
        (void)fprintf(stderr, "[FAIL] %s: allocation failure\n", name);
        ++g_failures;
        free(x);
        free(y);
        free(z);
        free(mass);
        free(fx);
        free(fy);
        free(fz);
        free(id);
        free(morton);
        free(ref_fx);
        free(ref_fy);
        free(ref_fz);
        free(arena);
        return -1.0;
    }

    fill(x, y, z, mass, id, cnt);
    direct_reference(x, y, z, mass, cnt, ref_fx, ref_fy, ref_fz);

    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, id, cnt};
    ps_context_t*      ctx  = NULL;
    ps_config_t        cfg  = {arena, ARENA_SIZE, cnt, THETA, 4};

    if (ps_init(&ctx, &cfg) != PS_OK) {
        (void)fprintf(stderr, "[FAIL] %s: ps_init failed\n", name);
        ++g_failures;
        free(x);
        free(y);
        free(z);
        free(mass);
        free(fx);
        free(fy);
        free(fz);
        free(id);
        free(morton);
        free(ref_fx);
        free(ref_fy);
        free(ref_fz);
        free(arena);
        return -1.0;
    }

    float min_b = 0.0F;
    float max_b = 0.0F;
    float range = 0.0F;
    if (ps_prepare_particles(&arrs, morton, &min_b, &max_b, &range) != PS_OK) {
        (void)fprintf(stderr, "[FAIL] %s: ps_prepare_particles failed\n", name);
        ++g_failures;
    } else {
        float root_c = min_b + (range * 0.5F);
        if (ps_calc_forces(ctx, &arrs, morton, root_c, root_c, root_c,
                           range * 0.5F) != PS_OK) {
            (void)fprintf(stderr, "[FAIL] %s: ps_calc_forces failed\n", name);
            ++g_failures;
        }
    }

    double err2    = 0.0;
    double ref2    = 0.0;
    double max_rel = 0.0;

    for (size_t i = 0; i < cnt; ++i) {
        uint32_t original = id[i];
        double   dx       = (double)fx[i] - ref_fx[original];
        double   dy       = (double)fy[i] - ref_fy[original];
        double   dz       = (double)fz[i] - ref_fz[original];
        double   e2       = (dx * dx) + (dy * dy) + (dz * dz);
        double   r2       = (ref_fx[original] * ref_fx[original]) +
                    (ref_fy[original] * ref_fy[original]) +
                    (ref_fz[original] * ref_fz[original]);
        err2 += e2;
        ref2 += r2;
        if (r2 > 1e-24) {
            double rel = sqrt(e2 / r2);
            if (rel > max_rel) {
                max_rel = rel;
            }
        }
    }

    double rms_rel = ref2 > 0.0 ? sqrt(err2 / ref2) : 0.0;
    (void)printf(
        "[METRIC] %s: rms_relative_error=%.6f max_relative_error=%.6f\n", name,
        rms_rel, max_rel);

    ps_destroy(ctx);

    free(x);
    free(y);
    free(z);
    free(mass);
    free(fx);
    free(fy);
    free(fz);
    free(id);
    free(morton);
    free(ref_fx);
    free(ref_fy);
    free(ref_fz);
    free(arena);

    return rms_rel;
}

int main(void) {
    run_dipole_sign_test();
    run_st_vs_mt_test();

    double uniform = run_accuracy_case("uniform-512", 512, fill_uniform);
    double clusters =
        run_accuracy_case("two-clusters-512", 512, fill_two_clusters);

    /* Intentionally generous initial gates: these are validation smoke
    tests,
       not production accuracy targets. */
    if (uniform > 0.10) {
        failf("uniform RMS relative error", uniform, 0.10, 0.0);
    }
    if (clusters > 0.10) {
        failf("clustered RMS relative error", clusters, 0.10, 0.0);
    }

    if (g_failures != 0) {
        (void)fprintf(stderr, "\n%d accuracy validation(s) failed.\n",
                      g_failures);
        return EXIT_FAILURE;
    }

    (void)printf("\nAll accuracy validations passed.\n");
    return EXIT_SUCCESS;
}
