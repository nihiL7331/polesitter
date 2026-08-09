#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PS_MULTITHREADING
#define PS_ORDER 1
#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#define THETA 3.4641F // sqrt of 12

#if defined(_MSC_VER)

#include <malloc.h>
#define ALIGNED_MALLOC(size, align) _aligned_malloc((size), (align))
#define ALIGNED_FREE(ptr)           _aligned_free((ptr))

#else

#include <mm_malloc.h>
#define ALIGNED_MALLOC(size, align) _mm_malloc((size), (align))
#define ALIGNED_FREE(ptr)           _mm_free((ptr))

#endif

static void ps_assert(int cond, const char* expr_str, const char* msg,
                      const char* file, int line) {
    if (!cond) {
        (void)fprintf(stderr, "\n[FAIL] %s\n\t Assertion: %s (%s:%d)\n", msg,
                      expr_str, file, line);
        exit(EXIT_FAILURE);
    }
}

#define TEST_ASSERT(expr, msg)                                                 \
    ps_assert(!!(expr), #expr, msg, __FILE__, __LINE__)

#define TEST_ASSERT_FLOAT_EQ(expected, actual, eps)                            \
    TEST_ASSERT(fabsf((expected) - (actual)) < (eps), "Float mismatch")

#define RUN_TEST(test_func)                                                    \
    do {                                                                       \
        printf("[ ] %s...\n", #test_func);                                     \
        (void)fflush(stdout);                                                  \
        test_func();                                                           \
        printf("\r[X] %s - PASSED\n", #test_func);                             \
    } while (0)

void test_morton_encoding(void) {
    // interleaves the bits of x y and z to produce a single integer

    // x = 1 (001)
    // should land at bit 0 -> 0001
    TEST_ASSERT(ps_impl_morton_encode(1, 0, 0) == 1 /* 0b0001 */,
                "Morton code for x=1 failed");

    // y = 1 (001)
    // should land at bit 1 -> 0010
    TEST_ASSERT(ps_impl_morton_encode(0, 1, 0) == 2 /* 0b0010 */,
                "Morton code for y=1 failed");

    // z = 1 (001)
    // should land at bit 2 -> 0100
    TEST_ASSERT(ps_impl_morton_encode(0, 0, 1) == 4 /* 0b0100 */,
                "Morton code for z=1 failed");

    // x = 3 (011)
    // should land at bits 0 and 3 -> 1001
    TEST_ASSERT(ps_impl_morton_encode(3, 0, 0) == 9 /* 0b1001 */,
                "Morton code for x=3 failed");
}

void test_arena_allocator(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + 512;

    void* buffer = ALIGNED_MALLOC(total_mem, 64);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 4, THETA, 4};

    ps_result_t res = ps_init(&ctx, &cfg);
    TEST_ASSERT(res == PS_OK, "Failed to initialize context");
    TEST_ASSERT(ctx != NULL, "Context pointer is null");

    TEST_ASSERT(ctx->arena.cap == total_mem - ctx_size - 256,
                "Arena capacity mismatch");
    TEST_ASSERT(ctx->arena.off == 0, "Arena offset should start at 0");

    // basic alloc with SIMD align
    void* ptr1 = ps_impl_arena_alloc(ctx, 10);
    TEST_ASSERT(ptr1 != NULL, "1st allocation failed");
    TEST_ASSERT((uintptr_t)ptr1 % 64 == 0, "1st allocation not aligned to 64B");

    // padding test
    void* ptr2 = ps_impl_arena_alloc(ctx, 10);
    TEST_ASSERT(ptr2 != NULL, "2nd allocation failed");
    TEST_ASSERT((uintptr_t)ptr2 % 64 == 0, "2nd allocation not aligned to 64B");

    // ptr2 should be 64B after ptr1
    TEST_ASSERT((uintptr_t)ptr2 == (uintptr_t)ptr1 + 64,
                "2nd allocation not 16B after 1st allocation");

    // oom test
    void* ptr3 = ps_impl_arena_alloc(ctx, 256);
    TEST_ASSERT(ptr3 == NULL, "3rd allocation should have failed due to OOM");

    // clear test
    ps_impl_arena_clear(&ctx->arena);
    TEST_ASSERT(ctx->arena.off == 0, "Arena offset should be reset to 0");

    // reusability test
    void* ptr4 = ps_impl_arena_alloc(ctx, 32);
    TEST_ASSERT(ptr4 != NULL, "4th allocation failed after clear");
    TEST_ASSERT(
        ptr4 == ptr1,
        "4th allocation should reuse the same memory as 1st allocation");

    ps_destroy(ctx);
    ALIGNED_FREE(buffer);
}

void test_octree_insertion(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 16);

    void* buffer = malloc(total_mem);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 2, THETA, 4};

    ps_result_t res = ps_init(&ctx, &cfg);
    TEST_ASSERT(res == PS_OK, "Failed to initialize context");
    TEST_ASSERT(ctx != NULL, "Context pointer is null");

    // alloc the root node directly from arena
    ctx->root = ps_impl_alloc_node(ctx);
    TEST_ASSERT(ctx->root != NULL, "Failed to allocate root node");

    // seed geometry for hw
    ctx->root->x          = 0.0F;
    ctx->root->y          = 0.0F;
    ctx->root->z          = 0.0F;
    ctx->root->half_width = 10.0F;

    // create a sorted array containing both particles
    uint32_t morton_zero     = ps_impl_morton_encode(0, 0, 0);
    uint32_t morton_max      = ps_impl_morton_encode(1023, 1023, 1023);
    uint32_t morton_codes[2] = {morton_zero, morton_max};

    // bind to radix_state so worker threads can read it
    ctx->radix_state.m_src = morton_codes;

    ps_impl_build_tree(ctx, 0, ctx->root, 0, 0, 2, morton_codes);
    ps_impl_pool_wait(ctx);

    ps_node_t* curr = ctx->root;
    while (!(curr->data.leaf.is_leaf & 1)) {
        ps_node_t* curr_child =
            ps_impl_get_node(ctx, curr->data.children_offs[0]);
        TEST_ASSERT(curr_child != NULL, "Missing child in origin path");
        curr = curr_child;
    }

    TEST_ASSERT(curr->data.leaf.is_leaf, "Bottom node is not marked as leaf");
    TEST_ASSERT(curr->data.leaf.particle_cnt == 1,
                "Leaf particle count is wrong");
    TEST_ASSERT(curr->data.leaf.first_particle_idx == 0,
                "Leaf stored wrong particle idx");

    curr = ctx->root;
    while (!(curr->data.leaf.is_leaf & 1)) {
        ps_node_t* curr_child =
            ps_impl_get_node(ctx, curr->data.children_offs[7]);
        TEST_ASSERT(curr_child != NULL, "Missing child in max bounds path");
        curr = curr_child;
    }

    TEST_ASSERT(curr->data.leaf.is_leaf, "Bottom node is not marked as leaf");
    TEST_ASSERT(curr->data.leaf.particle_cnt == 1,
                "Leaf particle count is wrong");
    TEST_ASSERT(curr->data.leaf.first_particle_idx == 1,
                "Leaf stored wrong particle idx");

    ps_destroy(ctx);
    free(buffer);
}

void test_radix_sort(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 1024);

    void* buffer = ALIGNED_MALLOC(total_mem, 64);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 4, THETA, 4};

    ps_result_t res = ps_init(&ctx, &cfg);
    TEST_ASSERT(res == PS_OK, "Failed to initialize context");
    TEST_ASSERT(ctx != NULL, "Context pointer is null");

    uint32_t morton_codes[4] = {999, 10, 500, 42};
    uint32_t ids[4]          = {0, 1, 2, 3};
    float    x[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    y[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    z[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    mass[4]         = {9.0F, 1.0F, 5.0F, 4.0F};
    float    fx[4] = {0.0F}, fy[4] = {0.0F}, fz[4] = {0.0F};

    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, ids, 4};

    res = ps_impl_sort_particles(ctx, morton_codes, &arrs);
    TEST_ASSERT(res == PS_OK, "Sort function aborted early");

    // verify morton codes are strictly ascending
    TEST_ASSERT(morton_codes[0] == 10, "Sort failed at idx 0");
    TEST_ASSERT(morton_codes[1] == 42, "Sort failed at idx 1");
    TEST_ASSERT(morton_codes[2] == 500, "Sort failed at idx 2");
    TEST_ASSERT(morton_codes[3] == 999, "Sort failed at idx 3");

    // mass should match the morton code
    TEST_ASSERT_FLOAT_EQ(1.0F, arrs.mass[0], 1e-6F);
    TEST_ASSERT_FLOAT_EQ(4.0F, arrs.mass[1], 1e-6F);
    TEST_ASSERT_FLOAT_EQ(5.0F, arrs.mass[2], 1e-6F);
    TEST_ASSERT_FLOAT_EQ(9.0F, arrs.mass[3], 1e-6F);

    ps_destroy(ctx);
    ALIGNED_FREE(buffer);
}

void test_fmm_upward_pass(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 32);

    void* buffer = malloc(total_mem);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 2, THETA, 4};
    ps_init(&ctx, &cfg);

    float    x[2]    = {2.0F, -1.0F};
    float    y[2]    = {3.0F, -2.0F};
    float    z[2]    = {4.0F, -3.0F};
    float    mass[2] = {2.0F, 3.0F};
    float    fx[2] = {0.0F, 0.0F}, fy[2] = {0.0F, 0.0F}, fz[2] = {0.0F, 0.0F};
    uint32_t id[2] = {0, 1};

    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, id, 2};

    ctx->root                               = ps_impl_alloc_node(ctx);
    ctx->root->data.leaf.is_leaf            = 1;
    ctx->root->data.leaf.particle_cnt       = 2;
    ctx->root->data.leaf.first_particle_idx = 0;
    ctx->root->x                            = 0.0F;
    ctx->root->y                            = 0.0F;
    ctx->root->z                            = 0.0F;

    ps_impl_fmm_upward_pass(ctx, ctx->root, &arrs);

    // M0 = sum(mass) = 2.0 + 3.0 = 5.0
    // MX = sum(mass * (x - root_x)) = 2.0*2.0 + 3.0*-1.0 = 1.0
    // MY = sum(mass * (y - root_y)) = 2.0*3.0 + 3.0*-2.0 = 0.0
    // MZ = sum(mass * (z - root_z)) = 2.0*4.0 + 3.0*-3.0 = -1.0

    ps_node_t* root = ctx->root;
    TEST_ASSERT(root != NULL, "Root is null");

    float* r_multi = ps_impl_get_multipole(ctx, root);
    TEST_ASSERT_FLOAT_EQ(5.0F, r_multi[0], 1e-4F);  // M0
    TEST_ASSERT_FLOAT_EQ(1.0F, r_multi[1], 1e-4F);  // MX
    TEST_ASSERT_FLOAT_EQ(0.0F, r_multi[2], 1e-4F);  // MY
    TEST_ASSERT_FLOAT_EQ(-1.0F, r_multi[3], 1e-4F); // MZ

    ps_destroy(ctx);
    free(buffer);
}

void test_fmm_interaction_pass(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 32);

    void* buffer = malloc(total_mem);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 2, THETA, 4};
    ps_init(&ctx, &cfg);

    ps_node_t* node_a = ps_impl_alloc_node(ctx);
    ps_node_t* node_b = ps_impl_alloc_node(ctx);

    node_a->x          = -5.0F;
    node_a->y          = -5.0F;
    node_a->z          = -5.0F;
    node_a->half_width = 1.0F;

    node_b->x          = 5.0F;
    node_b->y          = 5.0F;
    node_b->z          = 5.0F;
    node_b->half_width = 1.0F;

    float* b_multi = ps_impl_get_multipole(ctx, node_b);
    b_multi[0]     = 1.0F;

    ps_impl_fmm_interaction_pass(ctx, node_a, node_b, THETA);

    // vector from A to B: dx=10, dy=10, dz=10
    // softened dist_sq = 300.1, dist ~= 17.3234
    // F_field = m * dx / (dist_sq^1.5) ~= 0.00192354
    float expected_field = 0.00192354F;

    // verify node a local expansion
    float* a_local = ps_impl_get_local(ctx, node_a);
    TEST_ASSERT_FLOAT_EQ(expected_field, a_local[1], 1e-6F); // F_x
    TEST_ASSERT_FLOAT_EQ(expected_field, a_local[2], 1e-6F); // F_y
    TEST_ASSERT_FLOAT_EQ(expected_field, a_local[3], 1e-6F); // F_z

    ps_destroy(ctx);
    free(buffer);
}

void test_fmm_downward_pass(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 16);

    void* buffer = malloc(total_mem);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 1, THETA, 4};
    ps_result_t   res = ps_init(&ctx, &cfg);
    TEST_ASSERT(res == PS_OK, "Init failed");

    uint32_t morton_max      = ps_impl_morton_encode(1023, 1023, 1023);
    uint32_t morton_codes[1] = {morton_max};

    float    x[1] = {0.0F}, y[1] = {0.0F}, z[1] = {0.0F}, mass[1] = {2.5F};
    float    fx[1] = {0.0F}, fy[1] = {0.0F}, fz[1] = {0.0F};
    uint32_t ids[1]         = {7331};
    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, ids, 1};

    // manually allocate and send the root
    ctx->root             = ps_impl_alloc_node(ctx);
    ctx->root->x          = 0.0F;
    ctx->root->y          = 0.0F;
    ctx->root->z          = 0.0F;
    ctx->root->half_width = 10.0F;

    ps_impl_build_tree(ctx, 0, ctx->root, 0, 0, 1, morton_codes);
    ps_impl_pool_wait(ctx);

    // inject a bg field at the root node
    float* root_local = ps_impl_get_local(ctx, ctx->root);
    root_local[1]     = 5.0F;  // F_x
    root_local[2]     = -3.5F; // F_y
    root_local[3]     = 42.0F; // F_z

    ps_impl_fmm_downward_pass(ctx, ctx->root, &arrs);

    // traverse down until we hit the adaptive leaf node
    ps_node_t* curr = ctx->root;
    while (!curr->data.leaf.is_leaf) {
        ps_node_t* curr_child =
            ps_impl_get_node(ctx, curr->data.children_offs[7]);
        TEST_ASSERT(curr_child != NULL, "Missing child in path");
        curr = curr_child;
    }

    TEST_ASSERT(curr->data.leaf.is_leaf, "Bottom node is not marked as leaf");

    // verify the field cascaded down properly (L2L)
    float* curr_local = ps_impl_get_local(ctx, curr);
    TEST_ASSERT_FLOAT_EQ(5.0F, curr_local[1], 1e-6F);  // F_x
    TEST_ASSERT_FLOAT_EQ(-3.5F, curr_local[2], 1e-6F); // F_y
    TEST_ASSERT_FLOAT_EQ(42.0F, curr_local[3], 1e-6F); // F_z

    // verify the L2P pass
    TEST_ASSERT_FLOAT_EQ(12.5F, arrs.fx[0], 1e-6F);  // F_x
    TEST_ASSERT_FLOAT_EQ(-8.75F, arrs.fy[0], 1e-6F); // F_y
    TEST_ASSERT_FLOAT_EQ(105.0F, arrs.fz[0], 1e-6F); // F_z

    ps_destroy(ctx);
    free(buffer);
}

void test_fmm_l2p_pass(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 8);

    void* buffer = malloc(total_mem);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 1, THETA, 4};
    ps_init(&ctx, &cfg);

    float x[1] = {0.0F}, y[1] = {0.0F}, z[1] = {0.0F}, mass[1] = {2.5F};
    float fx[1] = {0.0F}, fy[1] = {0.0F}, fz[1] = {0.0F};
    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, 0, 1};

    // manually create a leaf node
    ctx->root                               = ps_impl_alloc_node(ctx);
    ctx->root->data.leaf.is_leaf            = 1;
    ctx->root->data.leaf.particle_cnt       = 1;
    ctx->root->data.leaf.first_particle_idx = 0;

    // bg accel field
    float* root_local = ps_impl_get_local(ctx, ctx->root);
    root_local[1]     = 2.0F;
    root_local[2]     = -1.0F;
    root_local[3]     = 4.0F;

    ps_impl_fmm_downward_pass(ctx, ctx->root, &arrs);

    // check F = m * a
    TEST_ASSERT_FLOAT_EQ(5.0F, arrs.fx[0], 1e-6F);  // 2.5 * 2.0
    TEST_ASSERT_FLOAT_EQ(-2.5F, arrs.fy[0], 1e-6F); // 2.5 * -1.0
    TEST_ASSERT_FLOAT_EQ(10.0F, arrs.fz[0], 1e-6F); // 2.5 * 4.0

    ps_destroy(ctx);
    free(buffer);
}

void test_fmm_p2p_pass(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 8);

    void* buffer = malloc(total_mem);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 2, THETA, 4};
    ps_init(&ctx, &cfg);

    // 2 particles offset by 1.0 unit on the x axis
    float    x[2]    = {0.0F, 1.0F};
    float    y[2]    = {0.0F, 0.0F};
    float    z[2]    = {0.0F, 0.0F};
    float    mass[2] = {2.0F, 3.0F};
    float    fx[2]   = {0.0F, 0.0F};
    float    fy[2]   = {0.0F, 0.0F};
    float    fz[2]   = {0.0F, 0.0F};
    uint32_t id[2]   = {0, 1};

    const ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, id, 2};

    // manually create a leaf node containing both
    ctx->root                               = ps_impl_alloc_node(ctx);
    ctx->root->data.leaf.is_leaf            = 1;
    ctx->root->data.leaf.particle_cnt       = 2;
    ctx->root->data.leaf.first_particle_idx = 0;
    ctx->root->half_width                   = 10.0F; // not well-separated

    ps_impl_fmm_p2p_pass(ctx, ctx->root, ctx->root, &arrs, THETA);

    // dist = 1.0
    // dist_sq = 1.0^2 + 0.1 = 1.1
    // F_mag = (mass1 * mass2) / (dist_sq^1.5)
    float expected_f = (2.0F * 3.0F) * powf(1.1F, -1.5F);

    // 0 should be pulled towards 1 (+X dir)
    TEST_ASSERT_FLOAT_EQ(expected_f, arrs.fx[0], 1e-4F);
    TEST_ASSERT_FLOAT_EQ(0.0F, arrs.fy[0], 1e-6F);
    TEST_ASSERT_FLOAT_EQ(0.0F, arrs.fz[0], 1e-6F);

    // 1 should be pulled towards 0 (-X dir)
    TEST_ASSERT_FLOAT_EQ(-expected_f, arrs.fx[1], 1e-4F);
    TEST_ASSERT_FLOAT_EQ(0.0F, arrs.fy[1], 1e-6F);
    TEST_ASSERT_FLOAT_EQ(0.0F, arrs.fz[1], 1e-6F);

    ps_destroy(ctx);
    free(buffer);
}

int main(void) {
    RUN_TEST(test_morton_encoding);
    RUN_TEST(test_arena_allocator);
    RUN_TEST(test_octree_insertion);
    RUN_TEST(test_radix_sort);
    RUN_TEST(test_fmm_upward_pass);
    RUN_TEST(test_fmm_interaction_pass);
    RUN_TEST(test_fmm_downward_pass);
    RUN_TEST(test_fmm_l2p_pass);
    RUN_TEST(test_fmm_p2p_pass);

    return 0;
}
