#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#define THETA 3.4641F // sqrt of 12

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
    void* buffer = malloc(64);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_arena_t arena;
    ps_arena_init(&arena, buffer, 64);

    TEST_ASSERT(arena.cap == 64, "Arena capacity mismatch");
    TEST_ASSERT(arena.off == 0, "Arena offset should start at 0");

    // basic alloc with SIMD align
    void* ptr1 = ps_arena_alloc(&arena, 10, 16);
    TEST_ASSERT(ptr1 != NULL, "1st allocation failed");
    TEST_ASSERT((uintptr_t)ptr1 % 16 == 0, "1st allocation not aligned to 16B");

    // padding test
    void* ptr2 = ps_arena_alloc(&arena, 10, 16);
    TEST_ASSERT(ptr2 != NULL, "2nd allocation failed");
    TEST_ASSERT((uintptr_t)ptr2 % 16 == 0, "2nd allocation not aligned to 16B");

    // ptr2 should be 16B after ptr1
    TEST_ASSERT((uintptr_t)ptr2 == (uintptr_t)ptr1 + 16,
                "2nd allocation not 16B after 1st allocation");

    // oom test
    void* ptr3 = ps_arena_alloc(&arena, 64, 16);
    TEST_ASSERT(ptr3 == NULL, "3rd allocation should have failed due to OOM");

    // clear test
    ps_arena_clear(&arena);
    TEST_ASSERT(arena.off == 0, "Arena offset should be reset to 0");

    // reusability test
    void* ptr4 = ps_arena_alloc(&arena, 32, 16);
    TEST_ASSERT(ptr4 != NULL, "4th allocation failed after clear");
    TEST_ASSERT(
        ptr4 == ptr1,
        "4th allocation should reuse the same memory as 1st allocation");

    free(buffer);
}

void test_octree_insertion(void) {
    void* buffer = malloc(1024ULL * 4);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, 1024ULL * 4, THETA};

    ps_result_t res = ps_init(&ctx, &cfg);
    TEST_ASSERT(res == PS_OK, "Failed to initialize context");
    TEST_ASSERT(ctx != NULL, "Context pointer is null");

    // alloc the root node directly from arena
    ctx->root = (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    TEST_ASSERT(ctx->root != NULL, "Failed to allocate root node");
    ps_impl_node_init(ctx->root);

    // morton code is 0, tree should traverse down children[0] at every level
    uint32_t morton_zero = ps_impl_morton_encode(0, 0, 0);
    res = ps_impl_tree_insert(&ctx->arena, ctx->root, morton_zero, 42);
    TEST_ASSERT(res == PS_OK, "Failed to insert origin particle");

    ps_node_t* curr = ctx->root;
    for (int i = 0; i < 10 /* PS_MAX_DEPTH */; ++i) {
        TEST_ASSERT(curr->children[0] != NULL, "Missing child in origin path");
        curr = curr->children[0];
    }
    TEST_ASSERT(curr->is_leaf, "Bottom node is not marked as leaf");
    TEST_ASSERT(curr->particle_cnt == 1, "Leaf particle count is wrong");
    TEST_ASSERT(curr->first_particle_idx == 42,
                "Leaf stored wrong particle idx");

    // maximum bounds test, tree should traverse down children[7] at every level
    uint32_t morton_max = ps_impl_morton_encode(1023, 1023, 1023);
    res = ps_impl_tree_insert(&ctx->arena, ctx->root, morton_max, 99);
    TEST_ASSERT(res == PS_OK, "Failed to insert max bounds particle");

    curr = ctx->root;
    for (int i = 0; i < 10 /* PS_MAX_DEPTH */; ++i) {
        TEST_ASSERT(curr->children[7] != NULL,
                    "Missing child in max bounds path");
        curr = curr->children[7];
    }

    TEST_ASSERT(curr->is_leaf, "Bottom node is not marked as leaf");
    TEST_ASSERT(curr->particle_cnt == 1, "Leaf particle count is wrong");
    TEST_ASSERT(curr->first_particle_idx == 99,
                "Leaf stored wrong particle idx");

    free(buffer);
}

void test_radix_sort(void) {
    void* buffer = malloc(1024ULL * 4);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, 1024ULL * 4, THETA};
    ps_init(&ctx, &cfg);

    uint32_t morton_codes[4] = {999, 10, 500, 42};
    uint32_t ids[4]          = {0, 1, 2, 3};
    float    x[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    y[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    z[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    mass[4]         = {9.0F, 1.0F, 5.0F, 4.0F};
    float    fx[4] = {0.0F}, fy[4] = {0.0F}, fz[4] = {0.0F};

    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, ids, 4};

    ps_impl_sort_particles(&ctx->arena, morton_codes, &arrs);

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

    free(buffer);
}

void test_fmm_upward_pass(void) {
    void* buffer = malloc(1024ULL * 64);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, 1024ULL * 64, THETA};
    ps_init(&ctx, &cfg);

    float    x[2]    = {2.0F, -1.0F};
    float    y[2]    = {3.0F, -2.0F};
    float    z[2]    = {4.0F, -3.0F};
    float    mass[2] = {2.0F, 3.0F};
    float    fx[2] = {0.0F, 0.0F}, fy[2] = {0.0F, 0.0F}, fz[2] = {0.0F, 0.0F};
    uint32_t id[2] = {0, 1};

    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, id, 2};

    ctx->root = (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    ps_impl_node_init(ctx->root);
    ctx->root->is_leaf            = 1;
    ctx->root->particle_cnt       = 2;
    ctx->root->first_particle_idx = 0;
    ctx->root->x                  = 0.0F;
    ctx->root->y                  = 0.0F;
    ctx->root->z                  = 0.0F;

    ps_impl_fmm_upward_pass(ctx->root, &arrs);

    // M0 = sum(mass) = 2.0 + 3.0 = 5.0
    // MX = sum(mass * (x - root_x)) = 2.0*2.0 + 3.0*-1.0 = 1.0
    // MY = sum(mass * (y - root_y)) = 2.0*3.0 + 3.0*-2.0 = 0.0
    // MZ = sum(mass * (z - root_z)) = 2.0*4.0 + 3.0*-3.0 = -1.0

    ps_node_t* root = ctx->root;
    TEST_ASSERT(root != NULL, "Root is null");

    TEST_ASSERT_FLOAT_EQ(5.0F, root->multipole[0], 1e-4F);  // M0
    TEST_ASSERT_FLOAT_EQ(1.0F, root->multipole[1], 1e-4F);  // MX
    TEST_ASSERT_FLOAT_EQ(0.0F, root->multipole[2], 1e-4F);  // MY
    TEST_ASSERT_FLOAT_EQ(-1.0F, root->multipole[3], 1e-4F); // MZ

    free(buffer);
}

void test_fmm_interaction_pass(void) {
    void* buffer = malloc(1024ULL * 64);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, 1024ULL * 64, THETA};
    ps_init(&ctx, &cfg);

    ps_node_t* node_a =
        (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    ps_node_t* node_b =
        (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    ps_impl_node_init(node_a);
    ps_impl_node_init(node_b);

    node_a->x          = -5.0F;
    node_a->y          = -5.0F;
    node_a->z          = -5.0F;
    node_a->half_width = 1.0F;

    node_b->x          = 5.0F;
    node_b->y          = 5.0F;
    node_b->z          = 5.0F;
    node_b->half_width = 1.0F;

    node_b->multipole[0] = 1.0F;

    ps_impl_fmm_interaction_pass(node_a, node_b, THETA);

    // vector from A to B: dx=10, dy=10, dz=10
    // softened dist_sq = 300.1, dist ~= 17.3234
    // F_field = m * dx / (dist_sq^1.5) ~= 0.00192354
    float expected_field = 0.00192354F;

    // verify node a local expansion
    TEST_ASSERT_FLOAT_EQ(expected_field, node_a->local[1], 1e-6F); // F_x
    TEST_ASSERT_FLOAT_EQ(expected_field, node_a->local[2], 1e-6F); // F_y
    TEST_ASSERT_FLOAT_EQ(expected_field, node_a->local[3], 1e-6F); // F_z

    free(buffer);
}

void test_fmm_downward_pass(void) {
    void* buffer = malloc(1024ULL * 64);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, 1024ULL * 64, THETA};
    ps_init(&ctx, &cfg);

    // manually allocate and send the root
    ctx->root = (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    ps_impl_node_init(ctx->root);
    ctx->root->half_width = 10.0F;

    // insert a single particle to create a deep branch
    uint32_t morton_max = ps_impl_morton_encode(1023, 1023, 1023);
    ps_impl_tree_insert(&ctx->arena, ctx->root, morton_max, 42);

    // inject a bg field at the root node
    ctx->root->local[1] = 5.0F;  // F_x
    ctx->root->local[2] = -3.5F; // F_y
    ctx->root->local[3] = 42.0F; // F_z

    ps_impl_fmm_downward_pass(ctx->root);

    // traverse to the bottom leaf node
    ps_node_t* curr = ctx->root;
    for (int i = 0; i < 10 /* PS_MAX_DEPTH */; ++i) {
        TEST_ASSERT(curr->children[7] != NULL, "Missing child in path");
        curr = curr->children[7];
    }

    TEST_ASSERT(curr->is_leaf, "Bottom nod eis not marked as leaf");

    // verify the field cascaded down properly
    TEST_ASSERT_FLOAT_EQ(5.0F, curr->local[1], 1e-6F);  // F_x
    TEST_ASSERT_FLOAT_EQ(-3.5F, curr->local[2], 1e-6F); // F_y
    TEST_ASSERT_FLOAT_EQ(42.0F, curr->local[3], 1e-6F); // F_z

    free(buffer);
}

void test_fmm_l2p_pass(void) {
    void* buffer = malloc(1024ULL * 4);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, 1024ULL * 4, THETA};
    ps_init(&ctx, &cfg);

    float x[1] = {0.0F}, y[1] = {0.0F}, z[1] = {0.0F}, mass[1] = {2.5F};
    float fx[1] = {0.0F}, fy[1] = {0.0F}, fz[1] = {0.0F};
    ps_particle_arrs_t arrs = {x, y, z, mass, fx, fy, fz, 0, 1};

    // manually create a leaf node
    ctx->root = (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    ps_impl_node_init(ctx->root);
    ctx->root->is_leaf            = 1;
    ctx->root->particle_cnt       = 1;
    ctx->root->first_particle_idx = 0;

    // bg accel field
    ctx->root->local[1] = 2.0F;
    ctx->root->local[2] = -1.0F;
    ctx->root->local[3] = 4.0F;

    ps_impl_fmm_l2p_pass(ctx->root, &arrs);

    // check F = m * a
    TEST_ASSERT_FLOAT_EQ(5.0F, arrs.fx[0], 1e-6F);  // 2.5 * 2.0
    TEST_ASSERT_FLOAT_EQ(-2.5F, arrs.fy[0], 1e-6F); // 2.5 * -1.0
    TEST_ASSERT_FLOAT_EQ(10.0F, arrs.fz[0], 1e-6F); // 2.5 * 4.0

    free(buffer);
}

void test_fmm_p2p_pass(void) {
    void* buffer = malloc(1024ULL * 4);
    TEST_ASSERT(buffer != NULL, "Test buffer allocation failed");

    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, 1024ULL * 4, THETA};
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
    ctx->root = (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    ps_impl_node_init(ctx->root);
    ctx->root->is_leaf            = 1;
    ctx->root->particle_cnt       = 2;
    ctx->root->first_particle_idx = 0;
    ctx->root->half_width         = 10.0F; // not well-separated

    ps_impl_fmm_p2p_pass(ctx->root, ctx->root, &arrs);

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
