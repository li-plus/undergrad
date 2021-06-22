// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include "cuda_utils.h"
#include <cstdio>
#include <algorithm>

static constexpr int BLOCK_SIZE = 32;

namespace {

__global__ void diag_block_kernel(/* device */ int *graph, int n, int diag_start, int diag_size) {
    __shared__ int diag_block[BLOCK_SIZE][BLOCK_SIZE];

    int ti = threadIdx.y;
    int tj = threadIdx.x;

    int ci = diag_start + ti;
    int cj = diag_start + tj;
    int idx = ci * n + cj;

    if (ti < diag_size && tj < diag_size) {
        diag_block[ti][tj] = graph[idx];
    }
    __syncthreads();

    for (int k = 0; k < diag_size; k++) {
        diag_block[ti][tj] = min(diag_block[ti][tj], diag_block[ti][k] + diag_block[k][tj]);
        __syncthreads();
    }

    if (ti < diag_size && tj < diag_size) {
        graph[idx] = diag_block[ti][tj];
    }
}

__global__ void cross_block_kernel(/* device */ int *graph, int n, int diag_idx, int diag_start, int diag_size) {
    int curr_start = (blockIdx.x < diag_idx) ? blockIdx.x * BLOCK_SIZE : (blockIdx.x + 1) * BLOCK_SIZE;

    __shared__ int diag_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int curr_block[BLOCK_SIZE][BLOCK_SIZE];

    int ti = threadIdx.y;
    int tj = threadIdx.x;

    if (ti < diag_size && tj < diag_size) {
        int di = diag_start + ti;
        int dj = diag_start + tj;
        diag_block[ti][tj] = graph[di * n + dj];
    }

    int ci, cj;
    if (blockIdx.y == 0) {
        // horizontal line of blocks
        ci = diag_start + ti;
        cj = curr_start + tj;
    } else {
        // vertical line of blocks
        ci = curr_start + ti;
        cj = diag_start + tj;
    }

    int cid = ci * n + cj;
    if (ci < n && cj < n) {
        curr_block[ti][tj] = graph[cid];
    }

    __syncthreads();

    if (blockIdx.y == 0) {
        // horizontal
        for (int k = 0; k < diag_size; k++) {
            curr_block[ti][tj] = min(curr_block[ti][tj],
                diag_block[ti][k] + curr_block[k][tj]);
            __syncthreads();
        }
    } else {
        // vertical
        for (int k = 0; k < diag_size; k++) {
            curr_block[ti][tj] = min(curr_block[ti][tj],
                curr_block[ti][k] + diag_block[k][tj]);
            __syncthreads();
        }
    }

    if (ci < n && cj < n) {
        graph[cid] = curr_block[ti][tj];
    }
}

__global__ void remain_block_kernel(/* device */ int *graph, int n, int diag_idx, int diag_start, int diag_size) {
    int ti = threadIdx.y;
    int tj = threadIdx.x;

    int row_start = (blockIdx.y < diag_idx) ? blockIdx.y * BLOCK_SIZE : (blockIdx.y + 1) * BLOCK_SIZE;
    int col_start = (blockIdx.x < diag_idx) ? blockIdx.x * BLOCK_SIZE : (blockIdx.x + 1) * BLOCK_SIZE;

    __shared__ int row_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int col_block[BLOCK_SIZE][BLOCK_SIZE];

    // current block index
    int ci = row_start + ti;
    int cj = col_start + tj;

    // cross block index
    int xi = diag_start + ti;
    int xj = diag_start + tj;
    // row block
    if (xi < n && cj < n) {
        row_block[ti][tj] = graph[xi * n + cj];
    }
    // col block
    if (ci < n && xj < n) {
        col_block[ti][tj] = graph[ci * n + xj];
    }

    __syncthreads();

    if (ci < n && cj < n) {
        int cid = ci * n + cj;
        int dist = graph[cid];

        for (int k = 0; k < diag_size; k++) {
            // no need to sync because row & col blocks are read-only
            dist = min(dist, col_block[ti][k] + row_block[k][tj]);
        }

        graph[cid] = dist;
    }
}

}

void apsp(int n, /* device */ int *graph) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    constexpr int blk1 = 1;
    dim3 blk2(num_blocks - 1, 2);
    dim3 blk3(num_blocks - 1, num_blocks - 1);
    constexpr dim3 thr(BLOCK_SIZE, BLOCK_SIZE);

    for (int diag_idx = 0; diag_idx < num_blocks; diag_idx++) {
        int diag_start = diag_idx * BLOCK_SIZE;
        int diag_size = std::min(n - diag_start, BLOCK_SIZE);
        diag_block_kernel<<<blk1, thr>>>(graph, n, diag_start, diag_size);
        cross_block_kernel<<<blk2, thr>>>(graph, n, diag_idx, diag_start, diag_size);
        remain_block_kernel<<<blk3, thr>>>(graph, n, diag_idx, diag_start, diag_size);
    }
}

