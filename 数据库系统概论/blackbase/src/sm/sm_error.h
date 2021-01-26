#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "sm_defs.h"

#define SM_UNIX                 (SM_ERROR_START + 0)
#define SM_DATABASE_NOT_FOUND   (SM_ERROR_START + 1)
#define SM_DATABASE_EXISTS      (SM_ERROR_START + 2)
#define SM_TABLE_NOT_FOUND      (SM_ERROR_START + 3)
#define SM_TABLE_EXISTS         (SM_ERROR_START + 4)
#define SM_FIELD_NOT_FOUND      (SM_ERROR_START + 5)
#define SM_DATABASE_NOT_OPEN    (SM_ERROR_START + 6)
#define SM_BAD_ROOT_DIR         (SM_ERROR_START + 7)
#define SM_ERROR_END            (SM_ERROR_START + 8)

//#define SM_CANNOTCLOSE          (START_SM_WARN + 0)
//#define SM_BADRELNAME           (START_SM_WARN + 1)
//#define SM_BADREL               (START_SM_WARN + 2)
//#define SM_BADATTR              (START_SM_WARN + 3)
//#define SM_INVALIDATTR          (START_SM_WARN + 4)
//#define SM_INDEXEDALREADY       (START_SM_WARN + 5)
//#define SM_NOINDEX              (START_SM_WARN + 6)
//#define SM_BADLOADFILE          (START_SM_WARN + 7)
//#define SM_BADSET               (START_SM_WARN + 8)

//#define SM_LASTWARN             SM_BAD_ROOT_DIR
//
//
//#define SM_INVALIDDB            (START_SM_ERR - 0)
//#define SM_ERROR                (START_SM_ERR - 1) // error
//#define SM_LASTERROR            SM_ERROR

const char *sm_str_error(RC rc);

#ifdef __cplusplus
}
#endif
