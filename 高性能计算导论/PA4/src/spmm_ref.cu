#include "spmm_ref.h"

__global__ void spmm_kernel_ref(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < INFEATURE; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            result += vin[idx[i] * INFEATURE + j] * val[i];
        }
        vout[tid * INFEATURE + j] = result;
    }
}


void SpMMRef::preprocess(float *vin, float *vout)
{
    int BLOCK_SIZE = 128;
    grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block.x = BLOCK_SIZE;
}

void SpMMRef::run(float *vin, float *vout)
{
    spmm_kernel_ref<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
