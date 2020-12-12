#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#define MAX_ROUND 5

#define MSG_TYPE_WRITER 1
#define MSG_TYPE_READER 2

#define MSG_ARRAY_SIZE 8

struct mq_msg {
    long type;
    unsigned int array[MSG_ARRAY_SIZE];
} message;

void print_message() {
    for (int i = 0; i < MSG_ARRAY_SIZE; i++) {
        printf("%u ", message.array[i]);
    }

    printf("\n");
}

void writer(int msgid) {
    for (int r = 0; r < MAX_ROUND; r++) {
        // increase the data
        for (int i = 0; i < MSG_ARRAY_SIZE; i++) {
            message.array[i]++;
        }

        // msgsnd to send message
        message.type = MSG_TYPE_WRITER;
        msgsnd(msgid, &message, sizeof(message), 0);
        // display the message
        printf("%-15s", "Data Sent:");
        print_message();

        // msgrcv to receive message
        msgrcv(msgid, &message, sizeof(message), MSG_TYPE_READER, 0);
        // display the message
        printf("%-15s", "Data Received:");
        print_message();
    }
}

void reader(int msgid) {
    for (int r = 0; r < MAX_ROUND; r++) {
        // msgrcv to receive message
        msgrcv(msgid, &message, sizeof(message), MSG_TYPE_WRITER, 0);
        // display the message
        printf("%-15s", "Data Received:");
        print_message();

        // increase the data
        for (int i = 0; i < MSG_ARRAY_SIZE; i++) {
            message.array[i]++;
        }

        // msgsnd to send message
        message.type = MSG_TYPE_READER;
        msgsnd(msgid, &message, sizeof(message), 0);
        // display the message
        printf("%-15s", "Data Sent:");
        print_message();
    }
}

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        printf("Usage:\n./main -r|-w\n");
        return 1;
    }

    int is_reader = !strcmp(argv[1], "-r");
    // ftok to generate unique key
    key_t key = ftok("progfile", 65);
    // msgget creates a message queue and returns identifier
    int msgid = msgget(key, 0666 | IPC_CREAT);

    if (is_reader) {
        reader(msgid);
    } else {
        writer(msgid);
    }

    // to destroy the message queue
    msgctl(msgid, IPC_RMID, NULL);
    return 0;
}