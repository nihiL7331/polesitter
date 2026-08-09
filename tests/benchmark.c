// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dc136-c,cert-dc151-cpp)
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PS_MULTITHREADING
#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#if defined(_MSC_VER)

#include <malloc.h>
#define ALIGNED_MALLOC(size, align) _aligned_malloc((size), (align))
#define ALIGNED_FREE(ptr)           _aligned_free((ptr))

#else

#include <mm_malloc.h>
#define ALIGNED_MALLOC(size, align) _mm_malloc((size), (align))
#define ALIGNED_FREE(ptr)           _mm_free((ptr))

#endif

#define ARENA_SIZE (1024ULL * 1024ULL * 512ULL)
#define THETA      1.0F

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

double get_wall_time(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }

    LARGE_INTEGER time;
    QueryPerformanceCounter(&time);

    return (double)time.QuadPart / (double)freq.QuadPart;
}

#else

#include <time.h>

double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ((double)ts.tv_nsec / 1e9);
}

#endif

double get_baseline_time(int count) {
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

    // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
    srand(7331);
    for (int i = 0; i < count; i++) {
        // NOLINTNEXTLINE(cert-msc30-c,cert-msc50-cpp)
        px[i] = ((float)rand() / RAND_MAX) * 100.0F;
        // NOLINTNEXTLINE(cert-msc30-c,cert-msc50-cpp)
        py[i] = ((float)rand() / RAND_MAX) * 100.0F;
        // NOLINTNEXTLINE(cert-msc30-c,cert-msc50-cpp)
        pz[i]   = ((float)rand() / RAND_MAX) * 100.0F;
        mass[i] = 1.0F;
        id[i]   = i;
    }

    ps_particle_arrs_t arrs = {px, py, pz, mass, fx, fy, fz, id, count};

    for (int i = 0; i < count; ++i) {
        arrs.fx[i] = 0.0F;
        arrs.fy[i] = 0.0F;
        arrs.fz[i] = 0.0F;
    }

    double start = get_wall_time();
    for (int i = 0; i < count; ++i) {
        float t_x    = arrs.x[i];
        float t_y    = arrs.y[i];
        float t_z    = arrs.z[i];
        float t_mass = arrs.mass[i];

        float f_x = 0.0F;
        float f_y = 0.0F;
        float f_z = 0.0F;

        for (int j = 0; j < count; ++j) {
            float mask = (float)(i != j);
            float p_dx = arrs.x[j] - t_x;
            float p_dy = arrs.y[j] - t_y;
            float p_dz = arrs.z[j] - t_z;

            float p_dist_sq =
                (p_dx * p_dx) + (p_dy * p_dy) + (p_dz * p_dz) + 0.1F;
            float inv_dist  = 1.0F / sqrtf(p_dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;

            float force = arrs.mass[j] * inv_dist3 * mask;

            f_x += p_dx * force;
            f_y += p_dy * force;
            f_z += p_dz * force;
        }

        arrs.fx[i] += t_mass * f_x;
        arrs.fy[i] += t_mass * f_y;
        arrs.fz[i] += t_mass * f_z;
    }
    double end = get_wall_time();

    ALIGNED_FREE(px);
    ALIGNED_FREE(py);
    ALIGNED_FREE(pz);
    ALIGNED_FREE(mass);
    ALIGNED_FREE(fx);
    ALIGNED_FREE(fy);
    ALIGNED_FREE(fz);
    ALIGNED_FREE(id);
    ALIGNED_FREE(morton_codes);

    return end - start;
}

const int test_sizes[] = {1000,  2500,  5000,  10000,  15000,  20000,
                          25000, 50000, 75000, 100000, 150000, 200000};

double get_test_time(int count, size_t thrd_cnt) {
    void*         buffer = malloc(ARENA_SIZE);
    ps_context_t* ctx    = NULL;
    ps_config_t   cfg    = {buffer, ARENA_SIZE, count, THETA, thrd_cnt};
    ps_init(&ctx, &cfg);

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

    // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
    srand(7331);
    for (int i = 0; i < count; i++) {
        // NOLINTNEXTLINE(cert-msc30-c,cert-msc50-cpp)
        px[i] = ((float)rand() / RAND_MAX) * 100.0F;
        // NOLINTNEXTLINE(cert-msc30-c,cert-msc50-cpp)
        py[i] = ((float)rand() / RAND_MAX) * 100.0F;
        // NOLINTNEXTLINE(cert-msc30-c,cert-msc50-cpp)
        pz[i]   = ((float)rand() / RAND_MAX) * 100.0F;
        mass[i] = 1.0F;
        id[i]   = i;
    }

    ps_particle_arrs_t arrs = {px, py, pz, mass, fx, fy, fz, id, count};

    for (int i = 0; i < count; ++i) {
        fx[i] = 0;
        fy[i] = 0;
        fz[i] = 0;
    }

    double start = get_wall_time();

    float min_b, max_b, range;
    ps_prepare_particles(&arrs, morton_codes, &min_b, &max_b, &range);

    float root_c  = min_b + (range / 2.0F);
    float root_hw = range / 2.0F;
    ps_calc_forces(ctx, &arrs, morton_codes, root_c, root_c, root_c, root_hw);

    double end = get_wall_time();

    ps_destroy(ctx);

    ALIGNED_FREE(px);
    ALIGNED_FREE(py);
    ALIGNED_FREE(pz);
    ALIGNED_FREE(mass);
    ALIGNED_FREE(fx);
    ALIGNED_FREE(fy);
    ALIGNED_FREE(fz);
    ALIGNED_FREE(id);
    ALIGNED_FREE(morton_codes);

    return end - start;
}

int main(void) {
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("N,Direct_Time,FMM_Time_ST,FMM_Time_MT\n");
    for (int i = 0; i < num_tests; ++i) {
        int count = test_sizes[i];
        printf("%d,%.6f,%.6f,%.6f\n", count, get_baseline_time(count),
               get_test_time(count, 1), get_test_time(count, 8));
    }

    return 0;
}
