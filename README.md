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

## Pipeline

The most common N-body tree codes rely on the Barnes-Hut algorithm, ending up with O(N log N) time complexity. 
It computes interactions between individual particles and distant cell multipoles (a Particle-to-Multipole approach).

`polesitter` implements the Greengard-Rokhlin Fast Multipole Method, dropping the time complexity to O(N).
It calculates interactions between cells and treats distant forces as a local background field.

<p align="center">
    <img src="docs/fmm_passes.svg" alt="FMM passes diagram" width="600" />
</p>

The solver executes physics ticks in four distinct passes over the Z-ordered octree:

1. **Upward pass (P2M & M2M)**
    - **Particle-to-Multipole:** Leaf nodes calculate their initial multipole expansion (total mass and cneter of mass) from their particles.
    - **Multipole-to-Multipole:** These expansions are aggregated up the tree from the leaves to the root. Every node now represents the center of mass of all its children.

2. **Interaction pass (M2L)**
    - **Multipole-to-Local:** For every node, the solver finds well-separated neighbors (cells far enough away to be approximated). It takes their multipole expansions and translated them into a local expansion (a Taylor series approximation of the gravitional field entering the target cell).

3. **Downward pass (L2L)**
    - **Local-to-Local:** Starting from the root, local background fields are translated and pushed down to the children. By the time it reaches the leaf nodes, every leaf has a single local expansion representing the gravitational pull of the entire universe.

4. **Evaluation pass (P2P & L2P)**
    - **Local-to-Particle:** The accumulated local expansion at the leaf is evaluated and applied to the particle, adding the force of all distant bodies.
    - **Particle-to-Particle:** Particles in adjacent, touching leaf nodes cannot be approximated, so they are evaluated using direct O(N^2) gravity.

## Benchmarks

![Linear Naive/FMM comparison graph.](/docs/performance_graph.png)
![Logarithmic Naive/FMM comparison graph.](/docs/performance_graph_log.png)

## License

MIT License.

## Sources

- [The Fastest Gravity Algorithm You've Never Heard Of by Keyframe Codes](https://youtu.be/FhMftauQZqU?si=E3nmNp6FuSqhn2OD)
