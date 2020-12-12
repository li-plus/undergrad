# Address Translation

https://github.com/LearningOS/os_course_exercises/blob/2020spring/all/04-1-spoc-discussion.md#虚拟页式存储的地址转换

## Calculate by Hand

My computation results are perfectly consistent with the program outputs. See below.

## Programming

Run:

```bash
python translate.py --virtual-addr 0x6653 0x1c13 0x6890 0x0af6 0x1e6f
```

Output:

```
Virtual Address 0x6653:
  --> Invalid virtual address

Virtual Address 0x1c13:
  --> pde index:0x7  pde contents:(valid 1, pfn 0x3d)
    --> pte index:0x0  pte contents:(valid 1, pfn 0x76)
      --> Translates to Physical Address 0xed3 --> Value: 0x12

Virtual Address 0x6890:
  --> Invalid virtual address

Virtual Address 0xaf6:
  --> pde index:0x2  pde contents:(valid 1, pfn 0x21)
    --> pte index:0x17  pte contents:(valid 0, pfn 0x7f)
      --> Fault (page table entry not valid)

Virtual Address 0x1e6f:
  --> pde index:0x7  pde contents:(valid 1, pfn 0x3d)
    --> pte index:0x13  pte contents:(valid 0, pfn 0x16)
      --> Translates to Physical Address 0x2cf --> Value: 0x1c
```
