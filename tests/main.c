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

int main(void) {
    RUN_TEST(test_morton_encoding);

    return 0;
}
