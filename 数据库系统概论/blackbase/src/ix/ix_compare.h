#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "ix_defs.h"
#include <string.h>
#include <assert.h>

static int ix_compare(const void *a, const void *b, attr_type_t type, int attr_len) {
    if (*(uint8_t *) a < *(uint8_t *) b) {
        return -1;
    } else if (*(uint8_t *) a > *(uint8_t *) b) {
        return 1;
    }
    a = (char *) a + 1;
    b = (char *) b + 1;
    switch (type) {
        case ATTR_INT: {
            int ia = *(int *) a;
            int ib = *(int *) b;
            return (ia < ib) ? -1 : ((ia > ib) ? 1 : 0);
        }
        case ATTR_FLOAT: {
            float fa = *(float *) a;
            float fb = *(float *) b;
            return (fa < fb) ? -1 : ((fa > fb) ? 1 : 0);
        }
        case ATTR_STRING:
            return memcmp(a, b, attr_len);
        default:
            assert(0);
    }
}

#ifdef __cplusplus
}
#endif
