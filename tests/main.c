#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#define TEST_ASSERT(expr)                                                      \
    do {                                                                       \
        if (!(expr)) {                                                         \
            (void)fprintf(stderr, "\n[FAIL] Assertion failed: %s (%s:%d)\n",   \
                          #expr, __FILE__, __LINE__);                          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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
    TEST_ASSERT(ps__morton_encode(1, 0, 0) == 1 /* 0b0001 */ &&
                "Morton code for x=1 failed");

    // y = 1 (001)
    // should land at bit 1 -> 0010
    TEST_ASSERT(ps__morton_encode(0, 1, 0) == 2 /* 0b0010 */ &&
                "Morton code for y=1 failed");

    // z = 1 (001)
    // should land at bit 2 -> 0100
    TEST_ASSERT(ps__morton_encode(0, 0, 1) == 4 /* 0b0100 */ &&
                "Morton code for z=1 failed");

    // x = 3 (011)
    // should land at bits 0 and 3 -> 1001
    TEST_ASSERT(ps__morton_encode(3, 0, 0) == 9 /* 0b1001 */ &&
                "Morton code for x=3 failed");
}

void test_arena_allocator(void) {
    uint8_t    buffer[64];
    ps_arena_t arena;
    ps_arena_init(&arena, buffer, sizeof(buffer));

    TEST_ASSERT(arena.cap == 64 && "Arena capacity mismatch");
    TEST_ASSERT(arena.off == 0 && "Arena offset should start at 0");

    // basic alloc with SIMD align
    void* ptr1 = ps_arena_alloc(&arena, 10, 16);
    TEST_ASSERT(ptr1 != NULL && "1st allocation failed");
    TEST_ASSERT((uintptr_t)ptr1 % 16 == 0 &&
                "1st allocation not aligned to 16B");

    // padding test
    void* ptr2 = ps_arena_alloc(&arena, 10, 16);
    TEST_ASSERT(ptr2 != NULL && "2nd allocation failed");
    TEST_ASSERT((uintptr_t)ptr2 % 16 == 0 &&
                "2nd allocation not aligned to 16B");

    // ptr2 should be 16B after ptr1
    TEST_ASSERT((uintptr_t)ptr2 == (uintptr_t)ptr1 + 16 &&
                "2nd allocation not 16B after 1st allocation");

    // oom test
    void* ptr3 = ps_arena_alloc(&arena, 64, 16);
    TEST_ASSERT(ptr3 == NULL && "3rd allocation should have failed due to OOM");

    // clear test
    ps_arena_clear(&arena);
    TEST_ASSERT(arena.off == 0 && "Arena offset should be reset to 0");

    // reusability test
    void* ptr4 = ps_arena_alloc(&arena, 32, 16);
    TEST_ASSERT(ptr4 != NULL && "4th allocation failed after clear");
    TEST_ASSERT(
        ptr4 == ptr1 &&
        "4th allocation should reuse the same memory as 1st allocation");
}

int main(void) {
    RUN_TEST(test_morton_encoding);
    RUN_TEST(test_arena_allocator);

    return 0;
}
