# Floyd-Warshall on CUDA

Build this project. You may need to add your device in `Makefile`.

```sh
make
```

Run on the largest graph with 10000 nodes.

```sh
./benchmark 10000
```

References:

+ Katz, G. J., & Kider, J. T. (2008). All-pairs shortest-paths for large graphs on the GPU.
+ Lund, B., & Smith, J. W. (2010). A multi-stage cuda kernel for floyd-warshall. arXiv preprint arXiv:1001.4108.
