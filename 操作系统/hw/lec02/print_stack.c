#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#define BUF_SIZE 64

void print_stack()
{
    void *buffer[BUF_SIZE];
    int nptrs = backtrace(buffer, BUF_SIZE);
    char **symbols = backtrace_symbols(buffer, nptrs);

    for (int i = 0; i < nptrs; i++)
    {
        printf("%s\n", symbols[i]);
    }

    free(symbols);
}

void bar()
{
    print_stack();
}

void foo()
{
    bar();
}

int main()
{
    foo();
    return 0;
}