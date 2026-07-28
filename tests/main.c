#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

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
    TEST_ASSERT(ps__morton_encode(1, 0, 0) == 1 /* 0b0001 */,
                "Morton code for x=1 failed");

    // y = 1 (001)
    // should land at bit 1 -> 0010
    TEST_ASSERT(ps__morton_encode(0, 1, 0) == 2 /* 0b0010 */,
                "Morton code for y=1 failed");

    // z = 1 (001)
    // should land at bit 2 -> 0100
    TEST_ASSERT(ps__morton_encode(0, 0, 1) == 4 /* 0b0100 */,
                "Morton code for z=1 failed");

    // x = 3 (011)
    // should land at bits 0 and 3 -> 1001
    TEST_ASSERT(ps__morton_encode(3, 0, 0) == 9 /* 0b1001 */,
                "Morton code for x=3 failed");
}

void test_arena_allocator(void) {
    uint8_t    buffer[64];
    ps_arena_t arena;
    ps_arena_init(&arena, buffer, sizeof(buffer));

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
}

void test_octree_insertion(void) {
    uint8_t       buffer[4096];
    ps_context_t* ctx    = NULL;
    ps_config_t   config = {buffer, sizeof(buffer)};

    ps_result_t res = ps_init(&ctx, &config);
    TEST_ASSERT(res == PS_OK, "Failed to initialize context");
    TEST_ASSERT(ctx != NULL, "Context pointer is null");

    // alloc the root node directly from arena
    ctx->root = (ps_node_t*)ps_arena_alloc(&ctx->arena, sizeof(ps_node_t), 16);
    TEST_ASSERT(ctx->root != NULL, "Failed to allocate root node");
    ps__node_init(ctx->root);

    // morton code is 0, tree should traverse down children[0] at every level
    uint32_t morton_zero = ps__morton_encode(0, 0, 0);
    res = ps__tree_insert(&ctx->arena, ctx->root, morton_zero, 42);
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
    uint32_t morton_max = ps__morton_encode(1023, 1023, 1023);
    res = ps__tree_insert(&ctx->arena, ctx->root, morton_max, 99);
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
}

void test_radix_sort(void) {
    uint8_t       buffer[4096];
    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, sizeof(buffer)};
    ps_init(&ctx, &cfg);

    uint32_t morton_codes[4] = {999, 10, 500, 42};
    float    x[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    y[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    z[4]            = {9.0F, 1.0F, 5.0F, 4.0F};
    float    mass[4]         = {9.0F, 1.0F, 5.0F, 4.0F};

    ps_particle_arrs_t arrs = {x, y, z, mass, NULL, NULL, NULL, 4};

    ps__sort_particles(&ctx->arena, morton_codes, &arrs);

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
}

void test_fmm_upward_pass(void) {
    uint8_t       buffer[1024 * 64];
    ps_context_t* ctx = NULL;
    ps_config_t   cfg = {buffer, sizeof(buffer)};
    ps_init(&ctx, &cfg);

    float              x[2]    = {2.0F, -1.0F};
    float              y[2]    = {3.0F, -2.0F};
    float              z[2]    = {4.0F, -3.0F};
    float              mass[2] = {2.0F, 3.0F};
    ps_particle_arrs_t arrs    = {x, y, z, mass, NULL, NULL, NULL, 2};

    ps_result_t res = ps_calc_forces(ctx, &arrs, 0.0F, 0.0F, 0.0F, 10.0F);
    TEST_ASSERT(res == PS_OK, "Failed to calculate forces");

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
}

int main(void) {
    RUN_TEST(test_morton_encoding);
    RUN_TEST(test_arena_allocator);
    RUN_TEST(test_octree_insertion);
    RUN_TEST(test_fmm_upward_pass);

    return 0;
}
