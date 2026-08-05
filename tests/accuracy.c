#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#define ARENA_SIZE   (1024ULL * 1024ULL * 32ULL)
#define SOFTENING_SQ 0.1F
#define THETA        2.0F

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
                             const float* mass, size_t count, double* fx,
                             double* fy, double* fz) {
    for (size_t i = 0; i < count; ++i) {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;

        for (size_t j = 0; j < count; ++j) {
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
    ps_node_t target;
    ps_node_t source;
    ps_impl_node_init(&target);
    ps_impl_node_init(&source);

    target.x          = 0.0F;
    target.y          = 0.0F;
    target.z          = 0.0F;
    target.half_width = 1.0F;

    source.x          = 100.0F;
    source.y          = 0.0F;
    source.z          = 0.0F;
    source.half_width = 1.0F;

    /* One unit mass displaced +1 from the source-cell center. */
    source.multipole[0] = 1.0F;
    source.multipole[1] = 1.0F;

    ps_impl_fmm_interaction_pass(&target, &source, THETA);

    /*
     * First-order softened expansion for R=100 and delta=+1 is
     * approximately 9.799856e-05. The large separation keeps this
     * analytic check independent of the production MAC threshold.
     */
    check_close("M2L dipole orientation", target.local[1], 0.00009799856, 5e-9);
}

static void fill_uniform(float* x, float* y, float* z, float* mass,
                         uint32_t* id, size_t count) {
    uint32_t state = 7331U;
    for (size_t i = 0; i < count; ++i) {
        x[i]    = (rand_unit(&state) * 100.0F) - 50.0F;
        y[i]    = (rand_unit(&state) * 100.0F) - 50.0F;
        z[i]    = (rand_unit(&state) * 100.0F) - 50.0F;
        mass[i] = 0.5F + (rand_unit(&state) * 1.5F);
        id[i]   = (uint32_t)i;
    }
}

static void fill_two_clusters(float* x, float* y, float* z, float* mass,
                              uint32_t* id, size_t count) {
    uint32_t state = 20260803U;
    size_t   half  = count / 2;

    for (size_t i = 0; i < count; ++i) {
        float center = i < half ? -12.0F : 12.0F;
        float skew   = i < half ? -0.75F : 0.75F;
        x[i]         = center + skew + ((rand_unit(&state) - 0.5F) * 2.0F);
        y[i]         = (rand_unit(&state) - 0.5F) * 2.0F;
        z[i]         = (rand_unit(&state) - 0.5F) * 2.0F;
        mass[i]      = 0.5F + (rand_unit(&state) * 2.0F);
        id[i]        = (uint32_t)i;
    }
}

static double run_accuracy_case(const char* name, size_t count,
                                void (*fill)(float*, float*, float*, float*,
                                             uint32_t*, size_t)) {
    float*    x      = (float*)malloc(count * sizeof(float));
    float*    y      = (float*)malloc(count * sizeof(float));
    float*    z      = (float*)malloc(count * sizeof(float));
    float*    mass   = (float*)malloc(count * sizeof(float));
    float*    fx     = (float*)calloc(count, sizeof(float));
    float*    fy     = (float*)calloc(count, sizeof(float));
    float*    fz     = (float*)calloc(count, sizeof(float));
    uint32_t* id     = (uint32_t*)malloc(count * sizeof(uint32_t));
    uint32_t* morton = (uint32_t*)malloc(count * sizeof(uint32_t));
    double*   ref_fx = (double*)malloc(count * sizeof(double));
    double*   ref_fy = (double*)malloc(count * sizeof(double));
    double*   ref_fz = (double*)malloc(count * sizeof(double));
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

    fill(x, y, z, mass, id, count);
    direct_reference(x, y, z, mass, count, ref_fx, ref_fy, ref_fz);

    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, id, count};
    ps_context_t*      ctx  = NULL;
    ps_config_t        cfg  = {arena, ARENA_SIZE, THETA, 1};

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

    for (size_t i = 0; i < count; ++i) {
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

    double uniform = run_accuracy_case("uniform-512", 512, fill_uniform);
    double clusters =
        run_accuracy_case("two-clusters-512", 512, fill_two_clusters);

    /* Intentionally generous initial gates: these are validation smoke tests,
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
