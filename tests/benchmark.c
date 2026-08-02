#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#if defined(_MSC_VER)
#    include <malloc.h>
#    define ALIGNED_MALLOC(size, align) _aligned_malloc((size), (align))
#    define ALIGNED_FREE(ptr)           _aligned_free((ptr))
#else
#    include <mm_malloc.h>
#    define ALIGNED_MALLOC(size, align) _mm_malloc((size), (align))
#    define ALIGNED_FREE(ptr)           _mm_free((ptr))
#endif

#define ARENA_SIZE (1024ULL * 1024ULL * 128ULL)

void direct_nbody_baseline(const ps_particle_arrs_t* arrs) {
    size_t cnt = arrs->cnt;

    for (size_t i = 0; i < cnt; ++i) {
        arrs->fx[i] = 0.0F;
        arrs->fy[i] = 0.0F;
        arrs->fz[i] = 0.0F;
    }

    for (size_t i = 0; i < cnt; ++i) {
        float t_x    = arrs->x[i];
        float t_y    = arrs->y[i];
        float t_z    = arrs->z[i];
        float t_mass = arrs->mass[i];

        float f_x = 0.0F;
        float f_y = 0.0F;
        float f_z = 0.0F;

        for (size_t j = 0; j < cnt; ++j) {
            float mask = (float)(i != j);
            float p_dx = arrs->x[j] - t_x;
            float p_dy = arrs->y[j] - t_y;
            float p_dz = arrs->z[j] - t_z;

            float p_dist_sq =
                (p_dx * p_dx) + (p_dy * p_dy) + (p_dz * p_dz) + 2.0F;
            float inv_dist  = 1.0F / sqrtf(p_dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;

            float force = arrs->mass[j] * inv_dist3 * mask;

            f_x += p_dx * force;
            f_y += p_dy * force;
            f_z += p_dz * force;
        }

        arrs->fx[i] += t_mass * f_x;
        arrs->fy[i] += t_mass * f_y;
        arrs->fz[i] += t_mass * f_z;
    }
}

int main(void) {
    int test_sizes[] = {1000,  2500,  5000,  10000,  15000,  20000,
                        25000, 50000, 75000, 100000, 150000, 200000};
    int num_tests    = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("N,Direct_Time,FMM_Time\n");

    for (int t = 0; t < num_tests; ++t) {
        int count = test_sizes[t];

        float*    px           = ALIGNED_MALLOC(count * sizeof(float), 32);
        float*    py           = ALIGNED_MALLOC(count * sizeof(float), 32);
        float*    pz           = ALIGNED_MALLOC(count * sizeof(float), 32);
        float*    mass         = ALIGNED_MALLOC(count * sizeof(float), 32);
        float*    fx           = ALIGNED_MALLOC(count * sizeof(float), 32);
        float*    fy           = ALIGNED_MALLOC(count * sizeof(float), 32);
        float*    fz           = ALIGNED_MALLOC(count * sizeof(float), 32);
        uint32_t* id           = ALIGNED_MALLOC(count * sizeof(uint32_t), 32);
        uint32_t* morton_codes = ALIGNED_MALLOC(count * sizeof(uint32_t), 32);
        memset(fx, 0, count * sizeof(float));
        memset(fy, 0, count * sizeof(float));
        memset(fz, 0, count * sizeof(float));

        srand(7331);
        for (int i = 0; i < count; i++) {
            px[i]   = ((float)rand() / RAND_MAX) * 100.0F;
            py[i]   = ((float)rand() / RAND_MAX) * 100.0F;
            pz[i]   = ((float)rand() / RAND_MAX) * 100.0F;
            mass[i] = 1.0F;
            id[i]   = i;
        }

        ps_particle_arrs_t arrs = {px, py, pz, mass, fx, fy, fz, id, count};

        clock_t start_direct = clock();
        direct_nbody_baseline(&arrs);
        clock_t end_direct = clock();
        double  time_direct =
            (double)(end_direct - start_direct) / CLOCKS_PER_SEC;

        for (int i = 0; i < count; ++i) {
            fx[i] = 0;
            fy[i] = 0;
            fz[i] = 0;
        }

        void*         buffer = malloc(ARENA_SIZE);
        ps_context_t* ctx    = NULL;
        ps_config_t   cfg    = {buffer, ARENA_SIZE};
        ps_init(&ctx, &cfg);

        clock_t start_fmm = clock();

        float min_b, max_b, range;
        ps_prepare_particles(&arrs, morton_codes, &min_b, &max_b, &range);

        float root_c  = min_b + (range / 2.0F);
        float root_hw = range / 2.0F;
        ps_calc_forces(ctx, &arrs, morton_codes, root_c, root_c, root_c,
                       root_hw);

        clock_t end_fmm  = clock();
        double  time_fmm = (double)(end_fmm - start_fmm) / CLOCKS_PER_SEC;

        printf("%d,%f,%f\n", count, time_direct, time_fmm);

        ALIGNED_FREE(buffer);
        ALIGNED_FREE(px);
        ALIGNED_FREE(py);
        ALIGNED_FREE(pz);
        ALIGNED_FREE(mass);
        ALIGNED_FREE(fx);
        ALIGNED_FREE(fy);
        ALIGNED_FREE(fz);
        ALIGNED_FREE(id);
        ALIGNED_FREE(morton_codes);
    }

    return 0;
}
