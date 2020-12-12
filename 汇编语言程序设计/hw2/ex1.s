.include "def.s"
.include "utils.s"

.data
    str_return: .string "\n"

.text
.globl _start
_start:
    movq    %rsp, %rbp  
    subq    $64, %rsp       # buffer for itoa

    cmpq    $3, (%rbp)      # if argc != 3, return.
    jne     main_ret

    movq    16(%rbp), %rdi  # argv[1]. input file name
    movq    $O_RDONLY, %rsi
    call    open_file
    movq    %rax, %r12      # input file descriptor

    movq    24(%rbp), %rdi  # argv[2]. output file name
    call    create_file
    movq    %rax, %r13      # output file descriptor

    movq    %r12, %rdi
    movq    $0, %r14        # '\n' counter
    main_read_loop:
    call    getchar
    cmpq    $EOF, %rax      # if EOF, break.
    je      main_read_loop_end
    movb    %al, %bl        # record last byte

    leaq    1(%r14), %rdx
    cmpq    $LF, %rax       # if byte == '\n'
    cmove   %rdx, %r14      # count++
    jmp     main_read_loop

    main_read_loop_end:
    leaq    1(%r14), %rdx
    cmpb    $LF, %bl        # if last byte != '\n'
    cmovne  %rdx, %r14      # count++

    call    close_file
    movq    %r14, %rdi
    movq    %rsp, %rsi
    call    itoa            # convert count to string

    movq    %rsp, %rdi
    movq    %r13, %rsi
    call    print_string    # write to output file
    movq    $str_return, %rdi
    movq    %r13, %rsi
    call    print_string    # write '\n'
    call    close_file

    main_ret:
    addq    $64, %rsp
    movq    $SYS_exit, %rax
    movq    $EXIT_SUCCESS, %rdi
    syscall
