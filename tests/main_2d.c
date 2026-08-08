#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PS_2D
#define PS_MULTITHREADING
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

void test_struct_sizes(void) {
    // quadtree node must be exactly 32 bytes to pack 2 per cache line
    TEST_ASSERT(sizeof(ps_node_t) == 32, "Quadtree node is not 32 bytes");
}

void test_morton_encoding_2d(void) {
    // 2D mode interleaves 16 bits per axis.

    // x = 1 (0x1) -> lands at bit 0 -> 0001
    TEST_ASSERT(ps_impl_morton_encode(1, 0) == 1, "Morton code for x=1 failed");

    // y = 1 (0x1) -> lands at bit 1 -> 0010
    TEST_ASSERT(ps_impl_morton_encode(0, 1) == 2, "Morton code for y=1 failed");

    // x = 3 (0b11) -> lands at bit 0 and 2 -> 0101 (5)
    TEST_ASSERT(ps_impl_morton_encode(3, 0) == 5, "Morton code for x=3 failed");

    // y = 3 (0b11) -> lands at bit 1 and 3 -> 1010 (10)
    TEST_ASSERT(ps_impl_morton_encode(0, 3) == 10,
                "Morton code for y=3 failed");
}

void test_arena_allocator_2d(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + 256;

    void* buffer = ALIGNED_MALLOC(total_mem, 64);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 4, THETA, 4};

    ps_result_t res = ps_init(&ctx, &cfg);
    TEST_ASSERT(res == PS_OK, "Failed to initialize context");
    TEST_ASSERT(ctx != NULL, "Context pointer is null");

    ps_destroy(ctx);
    ALIGNED_FREE(buffer);
}

void test_quadtree_insertion(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 16);

    void* buffer = malloc(total_mem);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 2, THETA, 4};
    ps_init(&ctx, &cfg);

    ctx->root             = ps_impl_alloc_node(ctx);
    ctx->root->x          = 0.0F;
    ctx->root->y          = 0.0F;
    ctx->root->half_width = 10.0F;

    uint32_t morton_zero     = ps_impl_morton_encode(0, 0);
    uint32_t morton_max      = ps_impl_morton_encode(65535, 65535);
    uint32_t morton_codes[2] = {morton_zero, morton_max};

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

    TEST_ASSERT(curr->data.leaf.first_particle_idx == 0,
                "Leaf stored wrong idx");

    curr = ctx->root;
    while (!(curr->data.leaf.is_leaf & 1)) {
        ps_node_t* curr_child =
            ps_impl_get_node(ctx, curr->data.children_offs[3]);
        TEST_ASSERT(curr_child != NULL, "Missing child in max bounds path");
        curr = curr_child;
    }

    TEST_ASSERT(curr->data.leaf.first_particle_idx == 1,
                "Leaf stored wrong idx");

    ps_destroy(ctx);
    free(buffer);
}

void test_radix_sort_2d(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 1024);

    void* buffer = ALIGNED_MALLOC(total_mem, 64);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, total_mem, 4, THETA, 4};
    ps_init(&ctx, &cfg);

    uint32_t morton_codes[4] = {999, 10, 500, 42};
    uint32_t ids[4]          = {0, 1, 2, 3};
    float    x[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    y[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    mass[4]         = {9.0F, 1.0F, 5.0F, 4.0F};
    float    fx[4] = {0.0F}, fy[4] = {0.0F};

    ps_particle_arrs_t arrs = {x, y, mass, fx, fy, ids, 4};

    ps_result_t res = ps_impl_sort_particles(ctx, morton_codes, &arrs);
    TEST_ASSERT(res == PS_OK, "Sort function aborted early");

    TEST_ASSERT(morton_codes[0] == 10, "Sort failed at idx 0");
    TEST_ASSERT(morton_codes[3] == 999, "Sort failed at idx 3");

    ps_destroy(ctx);
    ALIGNED_FREE(buffer);
}

void test_fmm_upward_pass_2d(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 32);

    void*         buffer = malloc(total_mem);
    ps_context_t* ctx    = NULL;
    ps_config_t   cfg    = {buffer, total_mem, 2, THETA, 4};
    ps_init(&ctx, &cfg);

    float    x[2]    = {2.0F, -1.0F};
    float    y[2]    = {3.0F, -2.0F};
    float    mass[2] = {2.0F, 3.0F};
    float    fx[2] = {0.0F}, fy[2] = {0.0F};
    uint32_t id[2] = {0, 1};

    ps_particle_arrs_t arrs = {x, y, mass, fx, fy, id, 2};

    ctx->root                               = ps_impl_alloc_node(ctx);
    ctx->root->data.leaf.is_leaf            = 1;
    ctx->root->data.leaf.particle_cnt       = 2;
    ctx->root->data.leaf.first_particle_idx = 0;
    ctx->root->x                            = 0.0F;
    ctx->root->y                            = 0.0F;

    ps_impl_fmm_upward_pass(ctx, ctx->root, &arrs);

    // M0 = sum(mass) = 5.0
    // MX = sum(mass * (x - root_x)) = 1.0
    // MY = sum(mass * (y - root_y)) = 0.0

    float* r_multi = ps_impl_get_multipole(ctx, ctx->root);
    TEST_ASSERT_FLOAT_EQ(5.0F, r_multi[0], 1e-4F); // M0
    TEST_ASSERT_FLOAT_EQ(1.0F, r_multi[1], 1e-4F); // MX
    TEST_ASSERT_FLOAT_EQ(0.0F, r_multi[2], 1e-4F); // MY

    ps_destroy(ctx);
    free(buffer);
}

void test_fmm_interaction_pass_2d(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 32);

    void*         buffer = malloc(total_mem);
    ps_context_t* ctx    = NULL;
    ps_config_t   cfg    = {buffer, total_mem, 2, THETA, 4};
    ps_init(&ctx, &cfg);

    ps_node_t* node_a = ps_impl_alloc_node(ctx);
    ps_node_t* node_b = ps_impl_alloc_node(ctx);

    node_a->x          = -5.0F;
    node_a->y          = -5.0F;
    node_a->half_width = 1.0F;

    node_b->x          = 5.0F;
    node_b->y          = 5.0F;
    node_b->half_width = 1.0F;

    float* b_multi = ps_impl_get_multipole(ctx, node_b);
    b_multi[0]     = 1.0F;

    ps_impl_fmm_interaction_pass(ctx, node_a, node_b, THETA);

    // 2D vector A to B: dx=10, dy=10
    // softened dist_sq = 200.1, dist ~= 14.1456
    // F_field = m * dx / (dist_sq^1.5) ~= 0.0035329
    float expected_field = 0.0035329F;

    float* a_local = ps_impl_get_local(ctx, node_a);
    TEST_ASSERT_FLOAT_EQ(expected_field, a_local[1], 1e-5F); // F_x
    TEST_ASSERT_FLOAT_EQ(expected_field, a_local[2], 1e-5F); // F_y

    ps_destroy(ctx);
    free(buffer);
}

void test_fmm_downward_pass_2d(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 16);

    void*         buffer = malloc(total_mem);
    ps_context_t* ctx    = NULL;
    ps_config_t   cfg    = {buffer, total_mem, 1, THETA, 4};
    ps_init(&ctx, &cfg);

    uint32_t morton_max      = ps_impl_morton_encode(65535, 65535);
    uint32_t morton_codes[1] = {morton_max};

    float              x[1] = {0.0F}, y[1] = {0.0F}, mass[1] = {2.5F};
    float              fx[1] = {0.0F}, fy[1] = {0.0F};
    uint32_t           ids[1] = {7331};
    ps_particle_arrs_t arrs   = {x, y, mass, fx, fy, ids, 1};

    ctx->root             = ps_impl_alloc_node(ctx);
    ctx->root->x          = 0.0F;
    ctx->root->y          = 0.0F;
    ctx->root->half_width = 10.0F;

    ps_impl_build_tree(ctx, 0, ctx->root, 0, 0, 1, morton_codes);
    ps_impl_pool_wait(ctx);

    float* root_local = ps_impl_get_local(ctx, ctx->root);
    root_local[1]     = 5.0F;  // F_x
    root_local[2]     = -3.5F; // F_y

    ps_impl_fmm_downward_pass(ctx, ctx->root, &arrs);

    ps_node_t* curr = ctx->root;
    while (!curr->data.leaf.is_leaf) {
        ps_node_t* curr_child =
            ps_impl_get_node(ctx, curr->data.children_offs[3]);
        curr = curr_child;
    }

    float* curr_local = ps_impl_get_local(ctx, curr);
    TEST_ASSERT_FLOAT_EQ(5.0F, curr_local[1], 1e-6F);  // F_x
    TEST_ASSERT_FLOAT_EQ(-3.5F, curr_local[2], 1e-6F); // F_y

    TEST_ASSERT_FLOAT_EQ(12.5F, arrs.fx[0], 1e-6F);  // F_x
    TEST_ASSERT_FLOAT_EQ(-8.75F, arrs.fy[0], 1e-6F); // F_y

    ps_destroy(ctx);
    free(buffer);
}

void test_fmm_p2p_pass_2d(void) {
    size_t ctx_size  = ps_impl_align_forward(sizeof(ps_context_t), 64);
    size_t total_mem = ctx_size + (1024ULL * 8);

    void*         buffer = malloc(total_mem);
    ps_context_t* ctx    = NULL;
    ps_config_t   cfg    = {buffer, total_mem, 2, THETA, 4};
    ps_init(&ctx, &cfg);

    float    x[2]    = {0.0F, 1.0F};
    float    y[2]    = {0.0F, 0.0F};
    float    mass[2] = {2.0F, 3.0F};
    float    fx[2]   = {0.0F, 0.0F};
    float    fy[2]   = {0.0F, 0.0F};
    uint32_t id[2]   = {0, 1};

    const ps_particle_arrs_t arrs = {x, y, mass, fx, fy, id, 2};

    ctx->root                               = ps_impl_alloc_node(ctx);
    ctx->root->data.leaf.is_leaf            = 1;
    ctx->root->data.leaf.particle_cnt       = 2;
    ctx->root->data.leaf.first_particle_idx = 0;
    ctx->root->half_width                   = 10.0F;

    ps_impl_fmm_p2p_pass(ctx, ctx->root, ctx->root, &arrs, THETA);

    // dist_sq = 1.0^2 + 0.1 = 1.1
    float expected_f = (2.0F * 3.0F) * powf(1.1F, -1.5F);

    TEST_ASSERT_FLOAT_EQ(expected_f, arrs.fx[0], 1e-4F);
    TEST_ASSERT_FLOAT_EQ(0.0F, arrs.fy[0], 1e-6F);

    TEST_ASSERT_FLOAT_EQ(-expected_f, arrs.fx[1], 1e-4F);
    TEST_ASSERT_FLOAT_EQ(0.0F, arrs.fy[1], 1e-6F);

    ps_destroy(ctx);
    free(buffer);
}

int main(void) {
    RUN_TEST(test_struct_sizes);
    RUN_TEST(test_morton_encoding_2d);
    RUN_TEST(test_arena_allocator_2d);
    RUN_TEST(test_quadtree_insertion);
    RUN_TEST(test_radix_sort_2d);
    RUN_TEST(test_fmm_upward_pass_2d);
    RUN_TEST(test_fmm_interaction_pass_2d);
    RUN_TEST(test_fmm_downward_pass_2d);
    RUN_TEST(test_fmm_p2p_pass_2d);

    return 0;
}
