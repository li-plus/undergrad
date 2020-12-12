# Solving Reader-writer Problem with Semaphore or Monitor

## First reader-writer problem (reader-first)

Run

```bash
python main.py --mode sem --type reader-first
```

Output

```
R0      R1      R2      W0      W1      W2
aCount
rCount
rd 0
        aCount
        rCount
        rd 0
aCount
rCount
        aCount
        rWrite
        rCount
                                        aWrite
                aCount
                                        wr 1
                                        rWrite
                        aWrite
                        wr 2
                        rWrite
                                aWrite
                                wr 3
                                rWrite
                rCount
                rd 3
                aCount
                rWrite
                rCount
```

## Second reader-writer problem (writer-first)

Run

```bash
python main.py --mode cond --type writer-first
```

Output

```
R0      R1      R2      W0      W1      W2
                        aLock
                        rLock
                        wr 1
                                aLock
                                wait
                aLock
                wait
                        aLock
                        rLock
        aLock
        wait
                                        aLock
                                        rLock
                                        wr 2
                                wait
aLock
wait
                                        aLock
                                        rLock
                                rLock
                                wr 3
                                aLock
                                rLock
                rLock
                rd 3
rLock
rd 3
        rLock
        rd 3
                aLock
                rLock
aLock
rLock
        aLock
        rLock
```

## Third reader-writer problem (first come first serve, FCFS)

Using semaphore

```bash
python main.py --mode sem --type fcfs
```

Output

```
R0      R1      R2      W0      W1      W2
aOrder
aCount
aAccess
rCount
rOrder
rd 0
aCount
rAccess
                aOrder
rCount
                aCount
                aAccess
                rCount
                rOrder
                rd 0
                aCount
                rAccess
                rCount
                                        aOrder
                                        aAccess
                                        rOrder
                                        wr 1
                                        rAccess
        aOrder
        aCount
        aAccess
        rCount
        rOrder
        rd 1
                                aOrder
        aCount
        rAccess
        rCount
                                aAccess
                                rOrder
                                wr 2
                                rAccess
                        aOrder
                        aAccess
                        rOrder
                        wr 3
                        rAccess
```

Using condition var

```bash
python main.py --mode cond --type fcfs
```

Output

```
R0      R1      R2      W0      W1      W2
                aLock
                rLock
                rd 0
                                        aLock
                                        wait
                        aLock
                        wait
                aLock
                rLock
        aLock
        rLock
        rd 0
aLock
wait
                                aLock
                                wait
                                        wait
        aLock
        rLock
                        rLock
                        wr 1
                        aLock
                        rLock
rLock
rd 1
aLock
rLock
                                rLock
                                wr 2
                                aLock
                                rLock
                                        rLock
                                        wr 3
                                        aLock
                                        rLock
```

Note: condition var does not guarantee strict FCFS behavior due to the Hansen monitor.
