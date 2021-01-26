#include "rm_error.h"
#include <assert.h>

static const char *error_msgs[] = {
        "Record not found",     // RM_RECORD_NOT_FOUND
};

const char *rm_str_error(RC rc) {
    assert(RM_ERROR_START <= rc && rc < RM_ERROR_END);
    return error_msgs[rc - RM_ERROR_START];
}
