# Sparse Matrix-Matrix multiplication (SpMM)

Build

```sh
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

Run unit test & performance test

```sh
./test/unit_tests --dataset ddi --datadir ../data/ --len 32
```
