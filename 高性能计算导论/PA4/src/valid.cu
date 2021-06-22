#include "valid.h"

__global__ void validate_float(float *ref, float *ans, int num, int *diffnum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num)
    {
        if (abs((ref[tid] - ans[tid]) / ref[tid]) > 1e-2)
        {
            atomicAdd(diffnum, 1);
        }
    }
}

__global__ void validate_int(int *ref, int *ans, int num, int *diffnum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num)
    {
        if (ref[tid] != ans[tid])
        {
            atomicAdd(diffnum, 1);
        }
    }
}

int valid(int *y, int *y2, int num)
{
    assert(num > 0);
    int* diffnum;
    checkCudaErrors(cudaMalloc2((void **)&diffnum, 1 * sizeof(int)));
    checkCudaErrors(cudaMemset(diffnum, 0, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());
    validate_int<<<(num + 127) / 128, 128>>>(y, y2, num, diffnum);
    checkCudaErrors(cudaDeviceSynchronize());
    int ans = -1;
    checkCudaErrors(cudaMemcpy(&ans, diffnum, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    return ans;
}

int valid(float *y, float *y2, int num)
{
    int *diffnum;
    int *d_zeronum;
    assert(num > 0);
    checkCudaErrors(cudaMalloc2((void **)&diffnum, 1 * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_zeronum, 1 * sizeof(int)));
    checkCudaErrors(cudaMemset(diffnum, 0, sizeof(int)));
    checkCudaErrors(cudaMemset(d_zeronum, 0, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());
    validate_float<<<(num + 127) / 128, 128>>>(y, y2, num, diffnum);
    checkCudaErrors(cudaDeviceSynchronize());
    int ans = -1;
    checkCudaErrors(cudaMemcpy(&ans, diffnum, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    return ans;
}