#include "rm/bitmap.h"
#include <cassert>
#include <ctime>
#include <bitset>

#define MAX_N 4096

void check_equal(bitmap_t bm, std::bitset<MAX_N> &mock) {
    for (int i = 0; i < MAX_N; i++) {
        assert(bitmap_test(bm, i) == mock.test(i));
    }
}

int main() {
    srand((unsigned) time(nullptr));

    uint8_t bm[MAX_N / BITMAP_WIDTH];
    bitmap_init(bm, MAX_N / BITMAP_WIDTH);
    std::bitset<MAX_N> mock;

    for (int round = 0; round < 10000; round++) {
        int choice = rand() % 2;
        int pos = rand() % MAX_N;
        if (choice == 0) {
            bitmap_set(bm, pos);
            mock.set(pos);
        } else {
            bitmap_reset(bm, pos);
            mock.reset(pos);
        }
        check_equal(bm, mock);
    }
    return 0;
}