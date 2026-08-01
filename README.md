# polesitter.h

> A zero-dependency, stb-like O(N) FMM N-body physics solver written in C99.

<p align="center">
      <img src="docs/output.webp" alt="Polesitter demo" width="600" />
</p>

## Quickstart

Include the header in one C file with `POLESITTER_IMPLEMENTATION` defined.

```c
#define POLESITTER_IMPLEMENTATION
#include "polesitter.h"
```

### Basic setup

```c
#include <stdint.h>
#include <stdlib.h>

#define MEMORY_SIZE 1024ULL * 1024 * 16 // 16 MB
#define PARTICLE_CNT 1024

int main(void) {
    // provide a raw memory block
    void* memory_block = malloc(MEMORY_SIZE);

    // initialize the context
    ps_config_t cfg = { memory_block, MEMORY_SIZE };
    ps_context_t* ctx = NULL;
    ps_init(&ctx, &cfg);

    // SoA particle arrays
    float x[PARTICLE_CNT], y[PARTICLE_CNT], z[PARTICLE_CNT];
    float mass[PARTICLE_CNT];
    float fx[PARTICLE_CNT] = {0}, fy[PARTICLE_CNT] = {0}, fz[PARTICLE_CNT] = {0};
    uint32_t morton_codes[PARTICLE_CNT];
    uint32_t ids[PARTICLE_CNT];

    // initialize positions and masses 
    // ...

    ps_particle_arrs_t arrs = {
        x, y, z, mass,
        fx, fy, fz,
        ids, PARTICLE_CNT
    };

    // calculate the global bounding box, generate morton codes
    float min_b, max_b, range;
    ps_prepare_particles(&arrs, morton_codes, &min_b, &max_b, &range);

    // run the physics tick
    // it builds the octree, computes multipoles and evaluates forces
    float root_c  = min_b + (range / 2.0F);
    float root_hw = range / 2.0F;
    ps_calc_forces(ctx, &arrs, morton_codes, root_c, root_c, root_c, root_hw);

    // integrate velocities/positions externally,
    // polesitter doesn't handle that
    // ...

    free(memory_block);
    return 0;
}

```

## Benchmarks

![Linear Naive/FMM comparison graph.](/docs/performance_graph.png)
![Logarithmic Naive/FMM comparison graph.](/docs/performance_graph_log.png)

## License

MIT License.

## Sources

- [The Fastest Gravity Algorithm You've Never Heard Of by Keyframe Codes](https://youtu.be/FhMftauQZqU?si=E3nmNp6FuSqhn2OD)
