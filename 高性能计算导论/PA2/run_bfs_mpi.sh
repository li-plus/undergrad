#!/bin/bash

# run on 4 machines * 1 process * 28 threads with process binding

export OMP_NUM_THREADS=28
export OMP_PROC_BIND=true
export OMP_PLACES=cores

srun -N 4 -n 4 ./bfs_omp_mpi $*

