# Multi-core Scheduler

Type `python o1_scheduler.py -h` for available options.

Run by default

```bash
python o1_scheduler.py --num-cpus 2 --jobs-priority 1 2 3 --jobs-runtime 35 25 15 --time-slice 10
```

Output

```
-----------------------------------------
   0:  CPU 0: 0  [ 34]  CPU 1: 1  [ 24]
   1:  CPU 0: 0  [ 33]  CPU 1: 1  [ 23]
   2:  CPU 0: 0  [ 32]  CPU 1: 1  [ 22]
   3:  CPU 0: 0  [ 31]  CPU 1: 1  [ 21]
   4:  CPU 0: 0  [ 30]  CPU 1: 1  [ 20]
   5:  CPU 0: 0  [ 29]  CPU 1: 1  [ 19]
   6:  CPU 0: 0  [ 28]  CPU 1: 1  [ 18]
   7:  CPU 0: 0  [ 27]  CPU 1: 1  [ 17]
   8:  CPU 0: 0  [ 26]  CPU 1: 1  [ 16]
   9:  CPU 0: 0  [ 25]  CPU 1: 1  [ 15]
-----------------------------------------
  10:  CPU 0: 2  [ 14]  CPU 1: 0  [ 24]
  11:  CPU 0: 2  [ 13]  CPU 1: 0  [ 23]
  12:  CPU 0: 2  [ 12]  CPU 1: 0  [ 22]
  13:  CPU 0: 2  [ 11]  CPU 1: 0  [ 21]
  14:  CPU 0: 2  [ 10]  CPU 1: 0  [ 20]
  15:  CPU 0: 2  [  9]  CPU 1: 0  [ 19]
  16:  CPU 0: 2  [  8]  CPU 1: 0  [ 18]
  17:  CPU 0: 2  [  7]  CPU 1: 0  [ 17]
  18:  CPU 0: 2  [  6]  CPU 1: 0  [ 16]
  19:  CPU 0: 2  [  5]  CPU 1: 0  [ 15]
-----------------------------------------
  20:  CPU 0: 1  [ 14]  CPU 1: 0  [ 14]
  21:  CPU 0: 1  [ 13]  CPU 1: 0  [ 13]
  22:  CPU 0: 1  [ 12]  CPU 1: 0  [ 12]
  23:  CPU 0: 1  [ 11]  CPU 1: 0  [ 11]
  24:  CPU 0: 1  [ 10]  CPU 1: 0  [ 10]
  25:  CPU 0: 1  [  9]  CPU 1: 0  [  9]
  26:  CPU 0: 1  [  8]  CPU 1: 0  [  8]
  27:  CPU 0: 1  [  7]  CPU 1: 0  [  7]
  28:  CPU 0: 1  [  6]  CPU 1: 0  [  6]
  29:  CPU 0: 1  [  5]  CPU 1: 0  [  5]
-----------------------------------------
  30:  CPU 0: 2  [  4]  CPU 1: 0  [  4]
  31:  CPU 0: 2  [  3]  CPU 1: 0  [  3]
  32:  CPU 0: 2  [  2]  CPU 1: 0  [  2]
  33:  CPU 0: 2  [  1]  CPU 1: 0  [  1]
  34:  CPU 0: -  [   ]  CPU 1: -  [   ]
  35:  CPU 0: 1  [  4]  CPU 1: -  [   ]
  36:  CPU 0: 1  [  3]  CPU 1: -  [   ]
  37:  CPU 0: 1  [  2]  CPU 1: -  [   ]
  38:  CPU 0: 1  [  1]  CPU 1: -  [   ]
  39:  CPU 0: -  [   ]  CPU 1: -  [   ]
```
