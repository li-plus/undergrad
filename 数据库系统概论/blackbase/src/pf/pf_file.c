#include "pf_file.h"
#include "pf_cache.h"
#include "pf_error.h"
#include "pf_defs.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

RC pf_create_file(const char *filename) {
    if (pf_exists(filename)) { return PF_FILE_EXISTS; }
    int fd = open(filename, O_CREAT, S_IRUSR | S_IWUSR);
    if (fd < 0) { return PF_UNIX; }
    if (close(fd) != 0) { return PF_UNIX; }
    return 0;
}

RC pf_destroy_file(const char *filename) {
    if (!pf_exists(filename)) { return PF_FILE_NOT_FOUND; }
    if (unlink(filename) != 0) { return PF_UNIX; }
    return 0;
}

RC pf_open_file(const char *filename, int *fd) {
    if (!pf_exists(filename)) { return PF_FILE_NOT_FOUND; }
    *fd = open(filename, O_RDWR);
    if (*fd < 0) { return PF_UNIX; }
    return 0;
}

RC pf_close_file(int fd) {
    RC rc;
    rc = pf_flush_file(fd);
    if (rc) { return rc; }
    if (close(fd) != 0) { return PF_UNIX; }
    return 0;
}
