#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "defs.h"
#include <sys/stat.h>

static inline bool pf_exists(const char *path) {
    static struct stat buffer;
    return stat(path, &buffer) == 0;
}

RC pf_create_file(const char *filename);

RC pf_destroy_file(const char *filename);

RC pf_open_file(const char *filename, int *fd);

RC pf_close_file(int fd);

#ifdef __cplusplus
}
#endif
