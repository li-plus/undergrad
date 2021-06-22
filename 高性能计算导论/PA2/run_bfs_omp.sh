#!/bin/bash

# run with 28 threads with binding

export OMP_NUM_THREADS=28
export OMP_PROC_BIND=true
export OMP_PLACES=cores

srun -n 1 ./bfs_omp $*

