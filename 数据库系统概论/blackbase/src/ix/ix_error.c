#include "ix_error.h"
#include <assert.h>

static const char *error_msgs[] = {
        "Bad filename",         // IX_BAD_FILENAME
        "Entry not found",      // IX_ENTRY_NOT_FOUND
};

const char *ix_str_error(RC rc) {
    assert(IX_ERROR_START <= rc && rc < IX_ERROR_END);
    return error_msgs[rc - IX_ERROR_START];
}
