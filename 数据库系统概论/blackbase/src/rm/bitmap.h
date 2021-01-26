#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "defs.h"
#include <string.h>

typedef uint8_t *bitmap_t;
typedef const uint8_t *input_bitmap_t;

#define BITMAP_WIDTH 8
#define BITMAP_HIGHEST_BIT 0x80u

static inline int _bitmap_get_bucket(int pos) {
    return pos / BITMAP_WIDTH;
}

static inline uint8_t _bitmap_get_bit(int pos) {
    return BITMAP_HIGHEST_BIT >> (pos % BITMAP_WIDTH);
}

static inline void bitmap_init(bitmap_t bm, int size) {
    memset(bm, 0, size);
}

static inline void bitmap_set(bitmap_t bm, int pos) {
    bm[_bitmap_get_bucket(pos)] |= _bitmap_get_bit(pos);
}

static inline void bitmap_reset(bitmap_t bm, int pos) {
    bm[_bitmap_get_bucket(pos)] &= ~_bitmap_get_bit(pos);
}

static inline bool bitmap_test(input_bitmap_t bm, int pos) {
    return (bm[_bitmap_get_bucket(pos)] & _bitmap_get_bit(pos)) != 0;
}

static inline bool bitmap_next_zero(input_bitmap_t bm, int max_n, int curr) {
    for (int i = curr + 1; i < max_n; i++) {
        if (!bitmap_test(bm, i)) {
            return i;
        }
    }
    return max_n;
}

static inline int bitmap_first_zero(input_bitmap_t bm, int max_n) {
    return bitmap_next_zero(bm, max_n, -1);
}

#ifdef __cplusplus
}
#endif
