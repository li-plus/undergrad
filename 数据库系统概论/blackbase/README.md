# BlackBase

> 本来想用纯C写的，写完PF, RM, IX, parser之后，赶DDL被迫上C++，写的很乱，请勿参考。
> 请参考 redbase 实现：https://github.com/li-plus/redbase-cpp

Build

```sh
mkdir -p build && cd build
cmake ..
make -j4
make test
```

Run

```
./src/main
```
