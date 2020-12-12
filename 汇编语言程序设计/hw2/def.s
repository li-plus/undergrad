.section .data
    .equ LF, 0x0A           # line feed
    .equ NULL, 0            # end of string
    .equ TRUE, 1
    .equ FALSE, 0
    .equ EXIT_SUCCESS, 0    # success code
    .equ EOF, -1            # end of file

    .equ STDIN, 0           # standard input
    .equ STDOUT, 1          # standard output
    .equ STDERR, 2          # standard error

    .equ SYS_read, 0        # read
    .equ SYS_write, 1       # write
    .equ SYS_open, 2        # file open
    .equ SYS_close, 3       # file close
    .equ SYS_fork, 57       # fork
    .equ SYS_exit, 60       # terminate
    .equ SYS_creat, 85      # file open/create
    .equ SYS_time, 201      # get 

    .equ O_CREAT, 0x40
    .equ O_TRUNC, 0x200
    .equ O_APPEND, 0x400
    .equ O_RDONLY, 000000   # read only
    .equ O_WRONLY, 000001   # write only
    .equ O_RDWR, 000002     # read and 

    .equ S_IRUSR, 0x100 
    .equ S_IWUSR, 0x80
    .equ S_IXUSR, 0x40
