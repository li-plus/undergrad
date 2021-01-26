#include "sm_error.h"
#include <assert.h>
#include <string.h>
#include <errno.h>

static const char *error_msgs[] = {
        "Unix system error",        // SM_UNIX
        "Database not found",       // SM_DATABASE_NOT_FOUND
        "Database already exists",  // SM_DATABASE_EXISTS
        "Table not found",          // SM_TABLE_NOT_FOUND
        "Table already exists",     // SM_TABLE_EXISTS
        "Field not found",          // SM_FIELD_NOT_FOUND
        "Database not open",        // SM_DATABASE_NOT_OPEN
        "Bad root directory",       // SM_BAD_ROOT_DIR
};

const char *sm_str_error(RC rc) {
    assert(SM_ERROR_START <= rc && rc < SM_ERROR_END);
    if (rc == SM_UNIX) {
        return strerror(errno);
    } else {
        return error_msgs[rc - SM_ERROR_START];
    }
}
