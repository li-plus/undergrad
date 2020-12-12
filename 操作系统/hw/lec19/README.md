# Disk Scheduling

Using FCFS (First Come First Serve)

```sh
python disksim.py -a 1,4,8,13,10,18,30,24,20 -p FIFO -c -G
```

Using SSTF (Shortest Seek Time First)

```sh
python disksim.py -a 1,4,8,13,10,18,30,24,20 -p SSTF -c -G
```

Using C-SCAN (Circular SCAN)

```sh
python disksim.py -a 1,4,8,13,10,18,30,24,20 -p CSCAN -c -G
```
