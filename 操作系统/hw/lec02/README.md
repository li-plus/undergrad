# Back Trace

Usage:

```bash
gcc print_stack.c -rdynamic -o print_stack
./print_stack
```

Output:

```
./print_stack(print_stack+0x2e) [0x400974]
./print_stack(bar+0xe) [0x400a12]
./print_stack(foo+0xe) [0x400a23]
./print_stack(main+0xe) [0x400a34]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf0) [0x7f2698c50830]
./print_stack(_start+0x29) [0x400879]
```
