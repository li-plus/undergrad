# Odd Even Sort

Compile

```sh
make
```

Run

```sh
./generate 100000000 100000000.dat
mpirun -n 8 --bind-to hwthread ./odd_even_sort 100000000 100000000.dat
```
