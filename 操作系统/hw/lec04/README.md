# Memory Allocation

https://github.com/chyyuu/os_tutorial_lab/blob/master/ostep/ostep3-malloc.md

## Analysis

Execute the following script.

```bash
python ./ostep3-malloc.py -S 100 -b 1000 -H 4 -a 4 -l ADDRSORT -p FIRST -n 5 -c
```

And you will get the output below.

```
ptr[0] = Alloc(3)  returned 1004 (searched 1 elements)
Free List [ Size 1 ]:  [ addr:1008 sz:92 ] 

Free(ptr[0]) returned 0
Free List [ Size 2 ]:  [ addr:1000 sz:8 ] [ addr:1008 sz:92 ] 

ptr[1] = Alloc(5)  returned 1012 (searched 2 elements)
Free List [ Size 2 ]:  [ addr:1000 sz:8 ] [ addr:1020 sz:80 ] 

Free(ptr[1]) returned 0
Free List [ Size 3 ]:  [ addr:1000 sz:8 ] [ addr:1008 sz:12 ] [ addr:1020 sz:80 ] 

ptr[2] = Alloc(8)  returned 1012 (searched 2 elements)
Free List [ Size 2 ]:  [ addr:1000 sz:8 ] [ addr:1020 sz:80 ] 
```

Some explanation are as follows:

1. Firstly, the Alloc(3) request 3 units from the memory. With a 4-unit header, and an alignment of 4 units, the total request size is 8. In the beginning, the free list is ((addr:1000, sz:100)) and is sufficient for allocation. It returns after searching 1 element and breaks the free list into (addr:1008, sz:92).
2. The previously allocated space is free and goes back to the free list, so the free list becomes ((addr:1000, sz:8), (addr:1008, sz:92)). Without coalescence, the adjacent free blocks will not merge.
3. The Alloc(5) calls for 5 units, but it results in a request of 12 units due to the header size and the alignment. Note that the FIRST policy searches for free space in the list from the beginning to the end. The first block (addr:1000, sz:8) is too small, so the algorithm uses the following (addr:1008, sz:92) as the solution. It breaks the free list into ((addr:1000, sz:8), (addr:1020, sz:80)).
4. The space is free again and returns to the free list.
5. The Alloc(8) requests 12 units in total. The first block in the list is too small, but the second element in the free list is as large as 12 units, ready for allocation. The FIRST policy will not search further since the second element is large enough, so it searches 2 elements totally.

## Buddy System

Buddy system is implemented in this assignment. Please specify `-B` for buddy system.

```bash
python ./ostep3-malloc.py -S 128 -b 1000 -H 4 -a 4 -n 16 -B -c
```

And you should get the following output.

```bash
ptr[0] = Alloc(3)  returned 1004 (searched 1 elements)
Free List [ Size 4 ]:  [ addr:1008 sz:8 ] [ addr:1016 sz:16 ] [ addr:1032 sz:32 ] [ addr:1064 sz:64 ] 

Free(ptr[0]) returned 0
Free List [ Size 1 ]:  [ addr:1000 sz:128 ] 

ptr[1] = Alloc(5)  returned 1004 (searched 1 elements)
Free List [ Size 3 ]:  [ addr:1016 sz:16 ] [ addr:1032 sz:32 ] [ addr:1064 sz:64 ] 

Free(ptr[1]) returned 0
Free List [ Size 1 ]:  [ addr:1000 sz:128 ] 

ptr[2] = Alloc(8)  returned 1004 (searched 1 elements)
Free List [ Size 3 ]:  [ addr:1016 sz:16 ] [ addr:1032 sz:32 ] [ addr:1064 sz:64 ] 

Free(ptr[2]) returned 0
Free List [ Size 1 ]:  [ addr:1000 sz:128 ] 

ptr[3] = Alloc(8)  returned 1004 (searched 1 elements)
Free List [ Size 3 ]:  [ addr:1016 sz:16 ] [ addr:1032 sz:32 ] [ addr:1064 sz:64 ] 

Free(ptr[3]) returned 0
Free List [ Size 1 ]:  [ addr:1000 sz:128 ] 

ptr[4] = Alloc(2)  returned 1004 (searched 1 elements)
Free List [ Size 4 ]:  [ addr:1008 sz:8 ] [ addr:1016 sz:16 ] [ addr:1032 sz:32 ] [ addr:1064 sz:64 ] 

ptr[5] = Alloc(7)  returned 1020 (searched 2 elements)
Free List [ Size 3 ]:  [ addr:1008 sz:8 ] [ addr:1032 sz:32 ] [ addr:1064 sz:64 ] 

Free(ptr[5]) returned 0
Free List [ Size 4 ]:  [ addr:1008 sz:8 ] [ addr:1016 sz:16 ] [ addr:1032 sz:32 ] [ addr:1064 sz:64 ] 

ptr[6] = Alloc(9)  returned 1020 (searched 2 elements)
Free List [ Size 3 ]:  [ addr:1008 sz:8 ] [ addr:1032 sz:32 ] [ addr:1064 sz:64 ] 

ptr[7] = Alloc(9)  returned 1036 (searched 2 elements)
Free List [ Size 3 ]:  [ addr:1008 sz:8 ] [ addr:1048 sz:16 ] [ addr:1064 sz:64 ] 

Free(ptr[4]) returned 0
Free List [ Size 3 ]:  [ addr:1000 sz:16 ] [ addr:1048 sz:16 ] [ addr:1064 sz:64 ] 

Free(ptr[6]) returned 0
Free List [ Size 3 ]:  [ addr:1000 sz:32 ] [ addr:1048 sz:16 ] [ addr:1064 sz:64 ] 

Free(ptr[7]) returned 0
Free List [ Size 1 ]:  [ addr:1000 sz:128 ] 
```
