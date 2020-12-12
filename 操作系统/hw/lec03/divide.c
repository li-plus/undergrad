
#include <stdio.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>

void fpe_handler(int signo)
{
    uint64_t dividend;
    asm volatile("movq %%rax, %0;"
                 : "=r"(dividend)
                 :
                 :);

    printf("Error: Floating Point Exception: %lu / 0\n", dividend);
    exit(1);
    // signal(SIGFPE, SIG_DFL);
}

int main(int argc, char *argv[])
{
    signal(SIGFPE, fpe_handler);

    if (argc != 3)
    {
        printf("Usage: %s dividend divisor\n", argv[0]);
        printf("Example:\n$ %s 128 2\nResult: 64\n", argv[0]);
        exit(1);
    }

    int dividend = atoi(argv[1]);
    int divisor = atoi(argv[2]);

    printf("Result: %d\n", dividend / divisor);
    return 0;
}