.text

strlen:     # long strlen(char * str);
    pushq   %rbp

    movq    $0, %rbp        # len
    cmpq    $NULL, %rdi     # if str == nullptr
    je      strlen_ret      # return 0

    strlen_loop:
    movb    (%rdi), %al     # copy byte
    cmpb    $0, %al         # if byte == '\0'
    je      strlen_ret      # return len
    incq    %rdi            # len++
    incq    %rbp            # str++
    jmp     strlen_loop

    strlen_ret:
    movq    %rbp, %rax
    popq    %rbp
    ret

print_string:    # void print_string(char* str, long descriptor);
    pushq   %rbp
    pushq   %rbx

    movq    %rsi, %rbx          # descriptor
    movq    %rdi, %rbp          # str
    call    strlen
    cmpq    $0, %rax
    je      print_string_ret
    movq    %rax, %rdx          # strlen
    movq    %rbp, %rsi          # str
    movq    $SYS_write, %rax
    movq    %rbx, %rdi
    syscall

    print_string_ret:
    popq    %rbx
    popq    %rbp
    ret

read_file:      # long read_file(long descriptor, char * buffer, long count)
    movq    $SYS_read, %rax
    syscall
    ret

open_file:      # long open_file(char * filename, long mode)
    movq    $SYS_open, %rax
    syscall
    ret

close_file:     # long close_file(long descriptor)
    movq    $SYS_close, %rax
    syscall
    ret

write_file:     # long write_file(long descriptor, char * src, long count);
    movq    $SYS_write, %rax
    syscall
    ret

create_file:    # long create_file(char * filename)
    movq    $SYS_creat, %rax
    movq    $S_IRUSR|S_IWUSR, %rsi
    syscall
    ret

getchar:        # long getchar(long descriptor); return -1 if EOF. return char otherwise.
    subq    $8, %rsp
    movq    %rsp, %rsi
    movq    $1, %rdx
    call    read_file
    cmpq    $0, %rax
    je      getchar_eof
    movq    $0, %rax
    movb    (%rsp), %al

    getchar_exit:
    addq    $8, %rsp
    ret

    getchar_eof:
    movq    $-1, %rax
    jmp     getchar_exit

itoa:           # void itoa(long n, char * dst);
    pushq   %rbp
    movq    $10, %rbp   # base
    movq    %rdi, %rax  # dividend

    pushq   $0          # '\0' is the end of string

    itoa_loop:
    movq    $0, %rdx    # clear high 64 bits
    idivq   %rbp        # rax / 10
    addq    $0x30, %rdx # convert remainder to char
    pushq   %rdx        # push into stack
    cmpq    $0, %rax    # if quotient != 0,
    jne     itoa_loop   # loop

    popout_loop:
    popq    %rdx
    movb    %dl, (%rsi)         # if byte == '\0', return
    incq    %rsi
    cmpb    $0x00, -1(%rsi)
    jne     popout_loop

    itoa_ret:
    popq    %rbp
    ret
