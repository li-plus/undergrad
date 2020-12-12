import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--addrs', nargs='+', type=int, 
                        default=[5, 4, 1, 3, 3, 4, 2, 3, 5, 3, 5, 1, 4])
    parser.add_argument('-w', '--window-size', type=int, default=4)
    args = parser.parse_args()

    # FIFO: first element is head and last element is tail.
    working_set = []
    pages = set()
    hits = 0
    for addr in args.addrs:
        print('Access:', addr, end='')
        if addr not in working_set:
            pages.add(addr)
            print('  MISS ', end='')
        else:
            hits += 1
            print('   HIT ', end='')

        working_set.append(addr)
        if len(working_set) > args.window_size:
            page = working_set.pop(0)
            if page in pages and page not in working_set:
                pages.remove(page)

        print((' Working Set: {:%d}' % (3 * args.window_size)).format(str(working_set)), ' Pages:', pages)

    print('\nFINALSTATS hits {}  misses {}  hitrate {:.2f}\n'.format(hits, len(args.addrs) - hits, hits / len(args.addrs)))


if __name__ == "__main__":
    main()
