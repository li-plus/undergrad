#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "ql_defs.h"

#define QL_MULTIPLE_PRIMARY_KEY_DEFINED     (QL_ERROR_START + 0)
#define QL_PRIMARY_KEY_NOT_FOUND            (QL_ERROR_START + 1)
#define QL_VALUE_STRING_TOO_LONG            (QL_ERROR_START + 2)
#define QL_DUPLICATE_ENTRY                  (QL_ERROR_START + 3)
#define QL_INVALID_VALUE_COUNT              (QL_ERROR_START + 4)
#define QL_INCOMPATIBLE_TYPE                (QL_ERROR_START + 5)
#define QL_EXPECTED_NOT_NULL                (QL_ERROR_START + 6)
#define QL_MULTIPLE_FOREIGN_KEY_DEFINED     (QL_ERROR_START + 7)
#define QL_LOCAL_FIELD_NOT_FOUND            (QL_ERROR_START + 8)
#define QL_FOREIGN_FIELD_NOT_FOUND          (QL_ERROR_START + 9)
#define QL_INVALID_FOREIGN_KEY_COUNT        (QL_ERROR_START + 10)
#define QL_FOREIGN_KEY_NOT_PRIMARY          (QL_ERROR_START + 11)
#define QL_REF_KEY_NOT_FOUND                (QL_ERROR_START + 12)
#define QL_REF_KEY_EXISTS                   (QL_ERROR_START + 13)
#define QL_NO_MORE_RECORD                   (QL_ERROR_START + 14)
#define QL_AMBIGUOUS_FIELD                  (QL_ERROR_START + 15)
#define QL_ERROR_END                        (QL_ERROR_START + 16)

const char *ql_str_error(RC rc);

#ifdef __cplusplus
}
#endif
