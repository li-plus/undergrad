# Divide by Zero

## Analysis

Say there is a divide-by-zero error in an application, then

+ The zero divisor is detected by CPU while performing `idiv`.
+ CPU raises Divide Error (#0), saves exception-related registers, jumps to the interrupt vector table, and executes the codes of corresponding handler defined by OS.
+ OS handler saves the scene and sends SIGFPE to the application, typically resulting in a termination.
+ OS handler restores the scene and returns.

## Run

Compile

```bash
gcc divide.c -o divide
```

Run

```bash
$ ./divide 128 2
Result: 64
$ ./divide 128 0
Error: Floating Point Exception: 128 / 0
```
