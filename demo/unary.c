#include <math.h>
#include <raylib.h>
#include <raymath.h>
#include <stdint.h>
#include <stdlib.h>

#define POLESITTER_IMPLEMENTATION
#include "../src/polesitter.h"

#define PARTICLE_COUNT     5000
#define BLACKHOLE_MASS     1000.0F
#define ARENA_SIZE         (1024ULL * 1024ULL * 256ULL)
#define MAX_EXPECTED_SPEED 20

float    vx[PARTICLE_COUNT]           = {0};
float    vy[PARTICLE_COUNT]           = {0};
float    vz[PARTICLE_COUNT]           = {0};
uint32_t morton_codes[PARTICLE_COUNT] = {0};
uint32_t id[PARTICLE_COUNT]           = {0};

uint32_t frame_counter = 0;

Camera3D camera = {0};

float              px[PARTICLE_COUNT], py[PARTICLE_COUNT], pz[PARTICLE_COUNT];
float              mass[PARTICLE_COUNT];
float              fx[PARTICLE_COUNT], fy[PARTICLE_COUNT], fz[PARTICLE_COUNT];
ps_context_t*      ctx  = NULL;
ps_particle_arrs_t arrs = {0};

void update(void);
void draw(void);

void init_raylib(void) {
    const int screenWidth  = 1080;
    const int screenHeight = 1080;
    InitWindow(screenWidth, screenHeight, "polesitter unary demo");
    SetTargetFPS(60);

    camera.position   = (Vector3){0.0F, 60.0F, 60.0F};
    camera.target     = (Vector3){0.0F, 0.0F, 0.0F};
    camera.up         = (Vector3){0.0F, 1.0F, 0.0F};
    camera.fovy       = 45.0F;
    camera.projection = CAMERA_PERSPECTIVE;
}

void* init_polesitter(void) {
    void* buffer = malloc(ARENA_SIZE);
    if (!buffer) {
        return NULL;
    }

    ps_config_t cfg = {buffer, ARENA_SIZE};
    ps_init(&ctx, &cfg);

    arrs.x    = px;
    arrs.y    = py;
    arrs.z    = pz;
    arrs.mass = mass;
    arrs.fx   = fx;
    arrs.fy   = fy;
    arrs.fz   = fz;
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
        py[i] = ((float)GetRandomValue(-100, 100) / 100.0F) *
                (2.0F / (radius + 1.0F));
        pz[i] = sinf(angle) * radius;

        mass[i] = 0.1F;

        float v_mag = sqrtf(BLACKHOLE_MASS / (radius + 1.0F));
        vx[i]       = -sinf(angle) * v_mag;
        vy[i]       = 0.0F;
        vz[i]       = cosf(angle) * v_mag;
    }

    px[0]   = 0.0F;
    py[0]   = 0.0F;
    pz[0]   = 0.0F;
    vx[0]   = 0.0F;
    vy[0]   = 0.0F;
    vz[0]   = 0.0F;
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
    UpdateCamera(&camera, CAMERA_ORBITAL);

    ps_arena_clear(&ctx->arena);

    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        id[i] = i;
        fx[i] = 0.0F;
        fy[i] = 0.0F;
        fz[i] = 0.0F;
    }

    float min_b, max_b, range;
    ps_prepare_particles(&arrs, morton_codes, &min_b, &max_b, &range);

    float root_c  = min_b + (range / 2.0F);
    float root_hw = range / 2.0F;
    ps_calc_forces(ctx, &arrs, morton_codes, root_c, root_c, root_c, root_hw);

    float tmp_vx[PARTICLE_COUNT];
    float tmp_vy[PARTICLE_COUNT];
    float tmp_vz[PARTICLE_COUNT];

    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        uint32_t old_idx = id[i];
        tmp_vx[i]        = vx[old_idx];
        tmp_vy[i]        = vy[old_idx];
        tmp_vz[i]        = vz[old_idx];
    }

    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        vx[i] = tmp_vx[i];
        vy[i] = tmp_vy[i];
        vz[i] = tmp_vz[i];
    }

    float dt = GetFrameTime();

    for (int i = 0; i < PARTICLE_COUNT; i++) {
        if (mass[i] == BLACKHOLE_MASS) {
            vx[i] = 0.0F;
            vy[i] = 0.0F;
            vz[i] = 0.0F;
            px[i] = 0.0F;
            py[i] = 0.0F;
            pz[i] = 0.0F;
            continue;
        }

        if (mass[i] <= 0.0F) {
            continue;
        }

        // a = F / m
        float ax = fx[i] / mass[i];
        float ay = fy[i] / mass[i];
        float az = fz[i] / mass[i];

        // v += a * dt
        vx[i] += ax * dt;
        vy[i] += ay * dt;
        vz[i] += az * dt;

        // p += v * dt
        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
        pz[i] += vz[i] * dt;
    }

    if (frame_counter++ == 30 * 40) {
        // CloseWindow();
    }
}

void draw(void) {
    ClearBackground((Color){8, 10, 22, 255});
    BeginMode3D(camera);

    DrawSphere((Vector3){0, 0, 0}, 0.7F, BLACK);

    BeginBlendMode(BLEND_ADDITIVE);

    for (int i = 0; i < PARTICLE_COUNT; i++) {
        float speed =
            sqrtf((vx[i] * vx[i]) + (vy[i] * vy[i]) + (vz[i] * vz[i]));

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
        DrawPoint3D((Vector3){px[i], py[i], pz[i]}, p_color);
    }

    EndBlendMode();
    EndMode3D();

    // TakeScreenshot(TextFormat("frames/frame_%04d.png", frame_counter));
}
