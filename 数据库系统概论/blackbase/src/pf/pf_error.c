#include "pf_error.h"
#include <assert.h>
#include <errno.h>
#include <string.h>

static const char *error_msgs[] = {
        "Unix system error",            // PF_UNIX
        "File already exists",          // PF_FILE_EXISTS
        "File not found",               // PF_FILE_NOT_FOUND
        "Incomplete read from file",    // PF_INCOMPLETE_READ
        "Incomplete write to file",     // PF_INCOMPLETE_WRITE
};

const char *pf_str_error(RC rc) {
    assert(PF_ERROR_START <= rc && rc < PF_ERROR_END);
    if (rc == PF_UNIX) {
        return strerror(errno);
    } else {
        return error_msgs[rc - PF_ERROR_START];
    }
}
