#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

static void merge_lo(float *x, int x_len, float *y, int y_len, float *z, int z_len) {
  int i = 0, j = 0, k = 0;
  while (k < z_len) {
    z[k++] = (j >= y_len || (i < x_len && x[i] < y[j])) ? x[i++] : y[j++];
  }
}

static void merge_hi(float *x, int x_len, float *y, int y_len, float *z, int z_len) {
  int i = x_len - 1, j = y_len - 1, k = z_len - 1;
  while (k >= 0) {
    z[k--] = (j < 0 || (i >= 0 && x[i] > y[j])) ? x[i--] : y[j--];
  }
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  if (out_of_range) {
    return;
  }

  int mean_len = ceiling(n, nprocs);
  int max_rank = ceiling(n, mean_len) - 1;
  int last_len = n - max_rank * mean_len;

  std::sort(data, data + block_len);

  auto buf = new float[mean_len * 2];
  auto nbr_data = buf + block_len;

  for (int i = 0; i < nprocs; i++) {
    int nbr_rank;
    if ((i % 2 == 0) ^ (rank % 2 == 0)) {
      if (rank == 0) { continue; }
      nbr_rank = rank - 1;
    } else {
      if (last_rank) { continue; }
      nbr_rank = rank + 1;
    }

    int nbr_len = (nbr_rank == max_rank) ? last_len : mean_len;

    MPI_Sendrecv(data, block_len, MPI_FLOAT, nbr_rank, 0,
      nbr_data, nbr_len, MPI_FLOAT, nbr_rank, 0,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (nbr_rank == rank - 1) {
      if (data[0] < nbr_data[nbr_len - 1]) {
        memcpy(buf, data, block_len * sizeof(float));
        merge_hi(buf, block_len, nbr_data, nbr_len, data, block_len);
      }
    } else {
      if (data[block_len - 1] > nbr_data[0]) {
        memcpy(buf, data, block_len * sizeof(float));
        merge_lo(buf, block_len, nbr_data, nbr_len, data, block_len);
      }
    }
  }
  delete[] buf;
}
