#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_X 32

static constexpr float SFM_MASS = 80;
static constexpr float SFM_TAU = 0.5;
static constexpr float SFM_A = 2.0e+3;
static constexpr float SFM_B = 0.08;
static constexpr float SFM_K = 1.2e+5;
static constexpr float SFM_KAPPA = 2.5e+5;

struct
{
    float *x;
    float *y;
    float *vx;
    float *vy;
} typedef pedestrians_t;

// struct __align__(8)
// {
//     float x;
//     float y;
//     float vx;
//     float vy;
// }
// typedef pedestrians_t;

__global__ void calc_accel(pedestrians_t pedestrians, float *axs, float *ays, size_t n)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    float ax, ay;
    float x = pedestrians.x[idx];
    float y = pedestrians.y[idx];
    float vx = pedestrians.vx[idx];
    float vy = pedestrians.vy[idx];

    for (size_t i = 0; i < n; i++)
    {
        float ox = pedestrians.x[i];
        float oy = pedestrians.y[i];
        float ovx = pedestrians.vx[i];
        float ovy = pedestrians.vy[i];

        float dirx = x - ox;
        float diry = y - oy;
        float d = hypotf(dirx, diry);
        if (d > 2.0f || d < 1e-9f)
        {
            continue;
        }

        float r = 2f * AGENT_SIZE;
        float diff = r - d;
        float nx = dirx / d;
        float ny = diry / d;
        float sfx = 0f;
        float sfy = 0f;
        if (diff >= 0.0)
        {
            sfx += nx * SFM_K * diff;
            sfy += ny * SFM_K * diff;
            float tx = -ny;
            float ty = nx;
            float tvd = (ovx - vx) * tx + (ovy - vy) * ty;
            sfx += tx * (SFM_KAPPA * diff * tvd);
            sfy += ty * (SFM_KAPPA * diff * tvd);
        }

        ax += sfx / SFM_MASS;
        ay += sfy / SFM_MASS;
    }

    axs[idx] = ax;
    ays[idx] = ay;
}

extern void tick_pedestrians(pedestrians_t pedestrians, size_t n)
{
    size_t bytes = sizeof(float) * n;

    float *axs, *ays;
    cudaMalloc(&axs, bytes);
    cudaMemset(axs, 0, bytes);
    cudaMalloc(&ays, bytes);
    cudaMemset(ays, 0, bytes);

    pedestrians_t d_pedestrians;
    cudaMalloc(&d_pedestrians.x, bytes);
    cudaMemcpy(d_pedestrians.x, pedestrians.x, bytes,
               cudaMemcpyHostToDevice);
    cudaMalloc(&d_pedestrians.y, bytes);
    cudaMemcpy(d_pedestrians.y, pedestrians.y, bytes,
               cudaMemcpyHostToDevice);
    cudaMalloc(&d_pedestrians.vx, bytes);
    cudaMemcpy(d_pedestrians.vx, pedestrians.vx, bytes,
               cudaMemcpyHostToDevice);
    cudaMalloc(&d_pedestrians.vy, bytes);
    cudaMemcpy(d_pedestrians.vy, pedestrians.vy, bytes,
               cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X);
    dim3 grid(n / BLOCK_X + 1);
    calc_accel<<<grid, block>>>(d_pedestrians, axs, ays, n);

    cudaFree(accels);
    cudaFree(d_pedestrians);
}

extern void hello()
{
    printf("hello from gpu");
}
