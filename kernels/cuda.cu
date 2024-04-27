#include "types.h"
#include <cuda_runtime.h>

typedef struct {
    float position[2];
    float velocity[2];
} pedestrian_t;

__global__ void calc_accel(pedestrian_t *pedestrians, float *accels, usize n) {
    u64 idx = blockDim.x * blockIdx.x + threadIdx.x;
    pedestrian_t *self = &pedestrians[idx];
    float *self_accel = accels + idx;

    for (u64 i = 0; i < n; i++) {
        pedestrian_t *other = &pedestrians[i];
    }

    //
}

extern "C" void determine_accel(pedestrian_t *pedestrians, float *accels,
                                usize n) {
    //
}
