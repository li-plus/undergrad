# Page Replacement Policy

The working-set replacement policy is implemented here.

The example below from slide 9-7 page 10 is used to validate the correctness. Run

```bash
python working-set-replacement.py --window-size 4 --addrs 5 4 1 3 3 4 2 3 5 3 5 1 4
```

Output

```
Access: 5  MISS  Working Set: [5]           Pages: {5}
Access: 4  MISS  Working Set: [5, 4]        Pages: {4, 5}
Access: 1  MISS  Working Set: [5, 4, 1]     Pages: {1, 4, 5}
Access: 3  MISS  Working Set: [5, 4, 1, 3]  Pages: {1, 3, 4, 5}
Access: 3   HIT  Working Set: [4, 1, 3, 3]  Pages: {1, 3, 4}
Access: 4   HIT  Working Set: [1, 3, 3, 4]  Pages: {1, 3, 4}
Access: 2  MISS  Working Set: [3, 3, 4, 2]  Pages: {2, 3, 4}
Access: 3   HIT  Working Set: [3, 4, 2, 3]  Pages: {2, 3, 4}
Access: 5  MISS  Working Set: [4, 2, 3, 5]  Pages: {2, 3, 4, 5}
Access: 3   HIT  Working Set: [2, 3, 5, 3]  Pages: {2, 3, 5}
Access: 5   HIT  Working Set: [3, 5, 3, 5]  Pages: {3, 5}
Access: 1  MISS  Working Set: [5, 3, 5, 1]  Pages: {1, 3, 5}
Access: 4  MISS  Working Set: [3, 5, 1, 4]  Pages: {1, 3, 4, 5}

FINALSTATS hits 5  misses 8  hitrate 0.38
```
