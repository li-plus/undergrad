#include <stdio.h>
#include <unistd.h>

#define MAX_ROUND 5

#define PIPE_ARRAY_SIZE 8

struct pipe_msg {
    unsigned int array[8];
} message;

void print_array(unsigned int *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }

    printf("\n");
}

int main() {
    // +--------+ --- out pipe --> +--------+
    // | parent |                  | child  |
    // +--------+ <--  in pipe --- +--------+
    int out_pipe[2];
    int in_pipe[2];
    pipe(out_pipe);
    pipe(in_pipe);

    if (fork()) {
        // parent
        close(out_pipe[0]);
        close(in_pipe[1]);

        for (int r = 0; r < MAX_ROUND; r++) {
            for (int i = 0; i < PIPE_ARRAY_SIZE; i++) {
                message.array[i]++;
            }

            printf("%-25s", "Parent sent:");
            print_array(message.array, PIPE_ARRAY_SIZE);
            write(out_pipe[1], &message, sizeof(struct pipe_msg));
            read(in_pipe[0], &message, sizeof(struct pipe_msg));
            printf("%-25s", "Parent received:");
            print_array(message.array, PIPE_ARRAY_SIZE);
        }
    } else {
        // child
        close(out_pipe[1]);
        close(in_pipe[0]);

        for (int r = 0; r < MAX_ROUND; r++) {
            read(out_pipe[0], &message, sizeof(struct pipe_msg));
            printf("%-25s", "Child received:");
            print_array(message.array, PIPE_ARRAY_SIZE);

            for (int i = 0; i < PIPE_ARRAY_SIZE; i++) {
                message.array[i]++;
            }

            printf("%-25s", "Child sent:");
            print_array(message.array, PIPE_ARRAY_SIZE);
            write(in_pipe[1], &message, sizeof(struct pipe_msg));
        }
    }

    return 0;
}