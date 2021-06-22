# 高性能计算导论 PA1

> 2017011620  计73  李家昊

`odd_even_sort.cpp` 中 `sort` 函数的源代码如下，实现思路见注释。

```cpp
// Given 2 sorted arrays x & y, copy the smallest z_len numbers into array z.
static void merge_lo(float *x, int x_len, float *y, int y_len, float *z, int z_len) {
  int i = 0, j = 0, k = 0;
  while (k < z_len) {
    z[k++] = (j >= y_len || (i < x_len && x[i] < y[j])) ? x[i++] : y[j++];
  }
}

// Given 2 sorted arrays x & y, copy the largest z_len numbers into array z.
static void merge_hi(float *x, int x_len, float *y, int y_len, float *z, int z_len) {
  int i = x_len - 1, j = y_len - 1, k = z_len - 1;
  while (k >= 0) {
    z[k--] = (j < 0 || (i >= 0 && x[i] > y[j])) ? x[i--] : y[j--];
  }
}

void Worker::sort() {
  if (out_of_range) {
    return;  // Skip out of range process
  }

  // Compute the rank & block_len of the last process
  int mean_len = ceiling(n, nprocs);
  int max_rank = ceiling(n, mean_len) - 1;
  int last_len = n - max_rank * mean_len;

  // Intra-process sort
  std::sort(data, data + block_len);

  // Alloc merge buffer
  auto buf = new float[mean_len * 2];
  auto nbr_data = buf + block_len;

  // Run for nprocs round, and finally the array must be sorted
  for (int i = 0; i < nprocs; i++) {
    // Figure out which neighbor to communicate
    int nbr_rank;
    if ((i % 2 == 0) ^ (rank % 2 == 0)) {
      // When (even round and odd rank) or (odd round and even rank),
      // pick the left neighbor
      if (rank == 0) { continue; }
      nbr_rank = rank - 1;
    } else {
      // Otherwise, pick the right neighbor
      if (last_rank) { continue; }
      nbr_rank = rank + 1;
    }

    // Get the neighbor's block_len
    int nbr_len = (nbr_rank == max_rank) ? last_len : mean_len;

    // Send my data and recv neighbor's data into buf[block_len:block_len+nbr_len]
    MPI_Sendrecv(data, block_len, MPI_FLOAT, nbr_rank, 0,
      nbr_data, nbr_len, MPI_FLOAT, nbr_rank, 0,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (nbr_rank == rank - 1) {
      if (data[0] < nbr_data[nbr_len - 1]) {
        // Keep the largest block_len numbers
        memcpy(buf, data, block_len * sizeof(float));
        merge_hi(buf, block_len, nbr_data, nbr_len, data, block_len);
      }
    } else {
      if (data[block_len - 1] > nbr_data[0]) {
        // Keep the smallest block_len numbers
        memcpy(buf, data, block_len * sizeof(float));
        merge_lo(buf, block_len, nbr_data, nbr_len, data, block_len);
      }
    }
  }
  delete[] buf;
}
```

在 $1\times 1$, $1 \times 2$, $1\times 4$, $1\times 8$, $1\times 16$, $2\times 16$ 进程（$N\times P$ 表示 $N$ 台机器，每台机器 $P$ 个进程）及元素数量 $n=100000000$ 的情况下，`sort` 函数的运行时间及相对单进程的加速比，如下表。使用的测试命令为：

```sh
srun -N <N> -n <NxP> --cpu-bind sockets ./odd_even_sort 100000000 data/100000000.dat
```

| 进程数       | 运行时间（ms） | 加速比  |
| ------------ | -------------- | ------- |
| $1\times 1$  | 9487.541       | 1.0000  |
| $1\times 2$  | 5272.529       | 1.7994  |
| $1\times 4$  | 2973.292       | 3.1909  |
| $1\times 8$  | 1857.345       | 5.1081  |
| $1\times 16$ | 1256.132       | 7.5530  |
| $2\times 16$ | 921.675        | 10.2938 |

