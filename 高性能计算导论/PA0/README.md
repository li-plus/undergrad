# OpenMP & MPI

Build project

```sh
make
```

Run OpenMP version

```sh
export OMP_PLACES=threads
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=8
./openmp_pow 112000 100000 0
```

Run MPI version

```sh
mpirun -n 8 --bind-to hwthread ./mpi_pow 112000 100000 0
```

