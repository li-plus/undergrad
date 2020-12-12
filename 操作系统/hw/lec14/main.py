import threading
import time
import random
import argparse

from sem_fcfs import SemFCFS
from sem_reader_first import SemReaderFirst
from cond_fcfs import CondFCFS
from cond_writer_first import CondWriterFirst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cond', choices=['sem', 'cond'],
                        help='Semaphore or condition mode?')
    parser.add_argument('--type', type=str, default='fcfs', choices=['fcfs', 'reader-first', 'writer-first'],
                        help='Types of the reader-writer problem')
    parser.add_argument('--num-readers', type=int, default=3)
    parser.add_argument('--num-writers', type=int, default=3)
    args = parser.parse_args()

    try:
        Worker = {
            'fcfs': {
                'sem': SemFCFS,
                'cond': CondFCFS
            },
            'reader-first': {
                'sem': SemReaderFirst
            },
            'writer-first': {
                'cond': CondWriterFirst
            }
        }[args.type][args.mode]
    except KeyError:
        print('Not implemented')
        exit(-1)

    readers = [threading.Thread(target=Worker.read, args=['\t' * i]) for i in range(args.num_readers)]
    writers = [threading.Thread(target=Worker.write, args=['\t' * (i + args.num_readers)]) for i in range(args.num_writers)]

    headers = []
    headers += ['R{}\t'.format(i) for i in range(args.num_readers)]
    headers += ['W{}\t'.format(i) for i in range(args.num_writers)]

    print(''.join(headers))

    threads = readers + writers
    random.shuffle(threads)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
