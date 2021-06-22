#include "spmm_opt.h"

static constexpr int WARP_SIZE = 32;

// Sparse-Dense Matrix-Matrix Multiplication (SpMM)
// Compute C = A * B, where
// * A is an NxN sparse matrix in CSR format (N = a_size).
// * B and C are NxK dense matrices (K = b_cols).
__global__ void spmm_kernel_opt(int *a_ptr, int *a_idx, float *a_val,
    float *b_val, float *c_val, int a_size, int b_cols) {

    __shared__ int sm_idxs[WARP_SIZE];
    __shared__ float sm_vals[WARP_SIZE];

    int row_id = blockIdx.x;
    int col_id = blockIdx.y * WARP_SIZE + threadIdx.y;

    if (row_id >= a_size) { return; }

    int begin_ptr = a_ptr[row_id];
    int end_ptr = a_ptr[row_id + 1];

    float sum = 0;

    for (int base_ptr = begin_ptr; base_ptr < end_ptr; base_ptr += WARP_SIZE) {
        int thr_ptr = base_ptr + threadIdx.y;
        if (thr_ptr < end_ptr) {
            // pre-compute row start
            sm_idxs[threadIdx.y] = a_idx[thr_ptr] * b_cols;
            sm_vals[threadIdx.y] = a_val[thr_ptr];
        }
        __syncwarp();
        int sm_end = min(WARP_SIZE, end_ptr - base_ptr);
        for (int i = 0; i < sm_end; i++) {
            int b_idx = sm_idxs[i] + col_id;
            sum += sm_vals[i] * b_val[b_idx];
        }
        __syncwarp();
    }
    c_val[row_id * b_cols + col_id] = sum;
}

__global__ void spmm_kernel_opt2(int *a_ptr, int *a_idx, float *a_val,
    float *b_val, float *c_val, int a_size, int b_cols) {

    __shared__ int sm_idxs[WARP_SIZE];
    __shared__ float sm_vals[WARP_SIZE];

    int row_id = blockIdx.x;
    int col_id = blockIdx.y * WARP_SIZE * 2 + threadIdx.y;

    if (row_id >= a_size) { return; }

    int begin_ptr = a_ptr[row_id];
    int end_ptr = a_ptr[row_id + 1];

    float sum0 = 0;
    float sum1 = 0;

    for (int base_ptr = begin_ptr; base_ptr < end_ptr; base_ptr += WARP_SIZE) {
        int thr_ptr = base_ptr + threadIdx.y;
        if (thr_ptr < end_ptr) {
            // pre-compute row start
            sm_idxs[threadIdx.y] = a_idx[thr_ptr] * b_cols;
            sm_vals[threadIdx.y] = a_val[thr_ptr];
        }
        __syncwarp();
        int sm_end = min(WARP_SIZE, end_ptr - base_ptr);
        for (int i = 0; i < sm_end; i++) {
            int b_idx = sm_idxs[i] + col_id;
            float val = sm_vals[i];
            sum0 += val * b_val[b_idx];
            sum1 += val * b_val[b_idx + WARP_SIZE];
        }
        __syncwarp();
    }
    int c_idx = row_id * b_cols + col_id;
    c_val[c_idx] = sum0;
    c_val[c_idx + WARP_SIZE] = sum1;
}

static inline int ceiling(int a, int b) {
    return (a + b - 1) / b;
}

void SpMMOpt::preprocess(float *vin, float *vout) {
    if (feat_in % WARP_SIZE != 0) {
        fprintf(stderr, "Error: K must be a multiple of %d, got %d\n",
            WARP_SIZE, feat_in);
    }

    block.x = 1;
    block.y = WARP_SIZE;
    grid.x = num_v;

    if (feat_in <= WARP_SIZE) {
        grid.y = ceiling(feat_in, WARP_SIZE);
    } else {
        grid.y = ceiling(feat_in, WARP_SIZE * 2);
    }
}

void SpMMOpt::run(float *vin, float *vout) {
    if (feat_in <= WARP_SIZE) {
        spmm_kernel_opt<<<grid, block>>>(
            d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    } else {
        spmm_kernel_opt2<<<grid, block>>>(
            d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    }
}