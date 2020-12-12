.include "def.s"
.include "utils.s"

.section .data
    str_return: .string "\n"
    str_line: .string "getline: "
    input_filename: .string "input.txt"

.text
.globl _start
_start:
    movq    %rsp, %rbp  
    subq    $64, %rsp       # buffer

    movq    $input_filename, %rdi
    movq    $O_RDONLY, %rsi
    call    open_file
    movq    %rax, %r12      # input file descriptor

    main_read_loop:
    movq    %r12, %rdi
    movq    %rsp, %rsi
    movq    $64, %rdx
    call    getline
    cmpq    $0, %rax
    jl      main_read_loop_end
    movq    $str_line, %rdi
    movq    $STDOUT, %rsi
    call    print_string
    movq    %rsp, %rdi
    movq    $STDOUT, %rsi
    call    print_string
    movq    $str_return, %rdi   # print '\n'
    movq    $STDOUT, %rsi
    call    print_string
    jmp     main_read_loop

    main_read_loop_end:
    call    close_file

    addq    $64, %rsp
    movq    $SYS_exit, %rax
    movq    $EXIT_SUCCESS, %rdi
    syscall

getline:    # long getline(long descriptor, char * dst, long len)
    pushq   %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13

    movq    %rdx, %rbp      # max len
    movq    %rsi, %rbx      # dst
    movq    $0, %r12        # return value
    movq    $0, %r13        # tmp

    getline_loop:
    call    getchar
    cmpq    $LF, %rax           # if '\n'
    je      getline_loop_end    # return bytes count

    movq    $-1, %r13
    cmpq    $EOF, %rax          # if EOF
    cmove   %r13, %r12          # return -1
    je      getline_loop_end

    movb    %al, (%rbx)
    incq    %r12
    incq    %rbx

    movq    $-1, %r13
    cmpq    %rbp, %r12          # if greater than max len
    cmovge  %r13, %r12          # return -1
    jge     getline_loop_end

    jmp     getline_loop

    getline_loop_end:
    movq    $0, (%rbx)
    movq    %r12, %rax

    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret
