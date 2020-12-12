# Inter-Process Communication

## Pipe

Usage:

```sh
cd pipe
make
./main
```

## Message Queue

Usage:

```sh
cd mq
make
# terminal A
./main -r
# terminal B
./main -w
```

## Shared Memory

Usage:

```sh
cd shm
make
# terminal A
./main -r
# terminal B
./main -w
```
