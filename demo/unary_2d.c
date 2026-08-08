#include <math.h>
#include <raylib.h>
#include <raymath.h>
#include <stdint.h>
#include <stdlib.h>

#define PS_2D
#define PS_MULTITHREADING
#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#define PARTICLE_COUNT     30000
#define BLACKHOLE_MASS     7000.0F
#define ARENA_SIZE         (1024ULL * 1024ULL * 512ULL)
#define MAX_EXPECTED_SPEED 750
#define THETA              3.4641F // sqrt of 12

float    vx[PARTICLE_COUNT]           = {0};
float    vy[PARTICLE_COUNT]           = {0};
uint32_t morton_codes[PARTICLE_COUNT] = {0};
uint32_t id[PARTICLE_COUNT]           = {0};

uint32_t frame_counter = 0;

Camera2D camera = {0};

float              px[PARTICLE_COUNT], py[PARTICLE_COUNT];
float              mass[PARTICLE_COUNT];
float              fx[PARTICLE_COUNT], fy[PARTICLE_COUNT];
ps_context_t*      ctx  = NULL;
ps_particle_arrs_t arrs = {0};

void update(void);
void draw(void);

void init_raylib(void) {
    const int screenWidth  = 1080;
    const int screenHeight = 1080;
    InitWindow(screenWidth, screenHeight, "polesitter 2D demo");
    SetTargetFPS(60);

    camera.offset =
        (Vector2){(float)screenWidth / 2.0F, (float)screenHeight / 2.0F};
    camera.target   = (Vector2){0.0F, 0.0F};
    camera.rotation = 0.0F;
    camera.zoom     = 5.0F;
}

void* init_polesitter(void) {
    void* buffer = malloc(ARENA_SIZE);
    if (!buffer) {
        return NULL;
    }

    ps_config_t cfg = {buffer, ARENA_SIZE, PARTICLE_COUNT, THETA, 8};
    ps_init(&ctx, &cfg);

    arrs.x    = px;
    arrs.y    = py;
    arrs.mass = mass;
    arrs.fx   = fx;
    arrs.fy   = fy;
    arrs.id   = id;
    arrs.cnt  = PARTICLE_COUNT;

    return buffer;
}

void init_particles(void) {
    for (int i = 0; i < PARTICLE_COUNT; i++) {
        float angle  = (float)GetRandomValue(0, 360) * DEG2RAD;
        float radius = (float)GetRandomValue(1, 1000) / 1000.0F;
        radius       = radius * radius * 30.0F;

        px[i] = cosf(angle) * radius;
        py[i] = sinf(angle) * radius;

        mass[i] = 0.1F;

        float dist_sq = (radius * radius) + 0.1F;
        float v_mag =
            sqrtf((BLACKHOLE_MASS * radius * radius) / powf(dist_sq, 1.5F));

        vx[i] = -sinf(angle) * v_mag;
        vy[i] = cosf(angle) * v_mag;
    }

    px[0]   = 0.0F;
    py[0]   = 0.0F;
    vx[0]   = 0.0F;
    vy[0]   = 0.0F;
    mass[0] = BLACKHOLE_MASS;
}

int main(void) {
    init_raylib();
    void* buffer = init_polesitter();
    init_particles();

    while (!WindowShouldClose()) {
        update();

        BeginDrawing();
        draw();
        EndDrawing();
    }

    free(buffer);
    CloseWindow();
    return 0;
}

void update(void) {
    camera.zoom += GetMouseWheelMove() * 0.5F;
    if (camera.zoom < 0.1F) {
        camera.zoom = 0.1F;
    }

    ps_impl_arena_clear(&ctx->arena);

    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        id[i] = i;
        fx[i] = 0.0F;
        fy[i] = 0.0F;
    }

    float min_b, max_b, range;
    ps_prepare_particles(&arrs, morton_codes, &min_b, &max_b, &range);

    float root_c  = min_b + (range / 2.0F);
    float root_hw = range / 2.0F;

    ps_calc_forces(ctx, &arrs, morton_codes, root_c, root_c, root_hw);

    float tmp_vx[PARTICLE_COUNT];
    float tmp_vy[PARTICLE_COUNT];

    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        uint32_t old_idx = id[i];
        tmp_vx[i]        = vx[old_idx];
        tmp_vy[i]        = vy[old_idx];
    }

    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        vx[i] = tmp_vx[i];
        vy[i] = tmp_vy[i];
    }

    float dt = GetFrameTime();

    for (int i = 0; i < PARTICLE_COUNT; i++) {
        if (mass[i] == BLACKHOLE_MASS) {
            vx[i] = 0.0F;
            vy[i] = 0.0F;
            px[i] = 0.0F;
            py[i] = 0.0F;
            continue;
        }

        if (mass[i] <= 0.0F) {
            continue;
        }

        // a = F / m
        float ax = fx[i] / mass[i];
        float ay = fy[i] / mass[i];

        // v += a * dt
        vx[i] += ax * dt;
        vy[i] += ay * dt;

        // p += v * dt
        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
    }

    frame_counter++;
}

void draw(void) {
    ClearBackground((Color){8, 10, 22, 255});
    BeginMode2D(camera);

    DrawCircle(0, 0, 0.7F, BLACK);

    BeginBlendMode(BLEND_ADDITIVE);

    for (int i = 0; i < PARTICLE_COUNT; i++) {
        float speed = (vx[i] * vx[i]) + (vy[i] * vy[i]);

        float t = speed / MAX_EXPECTED_SPEED;
        if (t > 1.0F) {
            t = 1.0F;
        }

        Color p_color;
        if (t < 0.5F) {
            float f = t * 2.0F;
            p_color = (Color){(unsigned char)(0), (unsigned char)(f * 180),
                              (unsigned char)(200 + (f * 55)), 200};
        } else {
            float f = (t - 0.5F) * 2.0F;
            p_color = (Color){(unsigned char)(200 - (f * 55)),
                              (unsigned char)(180 - (f * 100)),
                              (unsigned char)(255 - (f * 255)), 255};
        }

        DrawRectangleV((Vector2){px[i], py[i]}, (Vector2){0.2F, 0.2F}, p_color);
    }

    EndBlendMode();
    EndMode2D();

    DrawFPS(10, 10);
}
