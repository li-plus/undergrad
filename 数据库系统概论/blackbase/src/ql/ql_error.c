#include "ql_error.h"
#include <assert.h>

static const char *error_msgs[] = {
        "Multiple primary key defined",     // QL_MULTIPLE_PRIMARY_KEY_DEFINED
        "Primary key not found",            // QL_PRIMARY_KEY_NOT_FOUND
        "Value string too long",            // QL_VALUE_STRING_TOO_LONG
        "Duplicate entry for primary key",  // QL_DUPLICATE_ENTRY
        "Invalid value count",              // QL_INVALID_VALUE_COUNT
        "Incompatible type",                // QL_INCOMPATIBLE_TYPE
        "Expected a non-null value",        // QL_EXPECTED_NOT_NULL
        "Multiple foreign key defined",     // QL_MULTIPLE_FOREIGN_KEY_DEFINED
        "Local field not found",            // QL_LOCAL_FIELD_NOT_FOUND
        "Foreign field not found",          // QL_FOREIGN_FIELD_NOT_FOUND
        "Invalid foreign key count",        // QL_INVALID_FOREIGN_KEY_COUNT
        "Foreign key is not primary key",   // QL_FOREIGN_KEY_NOT_PRIMARY
        "Reference key not found",          // QL_REF_KEY_NOT_FOUND
        "Reference key exists",             // QL_REF_KEY_EXISTS
        "No more record",                   // QL_NO_MORE_RECORD
        "Ambiguous field",                  // QL_AMBIGUOUS_FIELD
};

const char *ql_str_error(RC rc) {
    assert(QL_ERROR_START <= rc && rc < QL_ERROR_END);
    return error_msgs[rc - QL_ERROR_START];
}
