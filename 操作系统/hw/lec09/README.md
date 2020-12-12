# Three-State Process Management Model

Usage

```bash
python process-run.py -l 2:0,4:50 -p -c
```

Output

```
Time     PID: 0     PID: 1        CPU        IOs
  1      RUN:io      READY          1
  2     WAITING    RUN:cpu          1          1
  3     WAITING    RUN:cpu          1          1
  4     WAITING     RUN:io          1          1
  5     WAITING    WAITING                     2
  6*     RUN:io    WAITING          1          1
  7     WAITING    WAITING                     2
  8     WAITING    WAITING                     2
  9*    WAITING    RUN:cpu          1          1
 10     WAITING       DONE                     1
 11*       DONE       DONE

Stats: Total Time 11
Stats: CPU Busy 6 (54.55%)
Stats: IO Busy  9 (81.82%)
```
