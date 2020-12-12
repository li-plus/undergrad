#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/ipc.h>
#include <sys/shm.h>

#define MAX_ROUND 5
#define SHM_KEY 0x1234

#define SHM_TYPE_READER 1
#define SHM_TYPE_WRITER 2

#define SHM_ARRAY_SIZE 8

struct shmseg {
    int type;
    unsigned int array[SHM_ARRAY_SIZE];
};

void print_array(unsigned int *array, int size) {
    for (int i = 0; i < SHM_ARRAY_SIZE; i++) {
        printf("%d ", array[i]);
    }

    printf("\n");
}

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        printf("Usage:\n./main -r|-w\n");
        return 1;
    }

    int my_type = !strcmp(argv[1], "-r") ? SHM_TYPE_READER : SHM_TYPE_WRITER;
    int shmid = shmget(SHM_KEY, sizeof(struct shmseg), 0644 | IPC_CREAT);
    struct shmseg *shmp = shmat(shmid, NULL, 0);

    for (int r = 0; r < MAX_ROUND; r++) {
        while (shmp->type == my_type) {
            // block
        }

        printf("%-15s", "Data received:");
        print_array(shmp->array, SHM_ARRAY_SIZE);

        for (int i = 0; i < SHM_ARRAY_SIZE; i++) {
            shmp->array[i]++;
        }

        printf("%-15s", "Data sent:");
        print_array(shmp->array, SHM_ARRAY_SIZE);
        shmp->type = my_type;
    }

    shmdt(shmp);
    shmctl(shmid, IPC_RMID, 0);
    return 0;
}
