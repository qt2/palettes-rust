#include <cuda_runtime.h>

#define BLOCK_X 32

struct {
    float *x;
    float *y;
    float *vx;
    float *vy;
    size_t n;
} typedef pedestrians_t;

struct __align__(8) {
    float x;
    float y;
    float vx;
    float vy;
}
typedef pedestrian_t;

__global__ void calc_accel() {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
}

extern void tick_pedestrians(pedestrian_t *pedestrians, size_t n) {
    float *accels;
    cudaMalloc(&accels, sizeof(float) * n);
    cudaMemset(accels, 0, sizeof(float) * n);

    pedestrian_t *d_pedestrians;
    cudaMalloc(&d_pedestrians, sizeof(pedestrian_t) * n);
    cudaMemcpy(d_pedestrians, pedestrians, sizeof(pedestrian_t) * n,
               cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X);
    dim3 grid(n / BLOCK_X + 1);
    calc_accel<<<grid, block>>>();

    cudaFree(accels);
    cudaFree(d_pedestrians);
}
