# Parallel BFS

Build the project

```sh
make
```

Run OpenMP solution

```sh
OMP_PLACES=threads OMP_PROC_BIND=true OMP_NUM_THREADS=8 ./bfs_omp graph/500k.graph
```

Run OpenMP + MPI solution

```sh
OMP_PLACES=threads OMP_PROC_BIND=true OMP_NUM_THREADS=2 mpirun --bind-to core -n 4 ./bfs_omp_mpi graph/500k.graph
```

References:

+ Beamer, S., Asanovic, K., & Patterson, D. (2012, November). Direction-optimizing breadth-first search. In SC'12: Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (pp. 1-10). IEEE. [[paper]](https://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf) [[slides]](https://people.csail.mit.edu/jshun/6886-s18/lectures/lecture4-1.pdf)
