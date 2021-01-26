#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef int RC;     // return code

#include <inttypes.h>

#ifndef __cplusplus
typedef enum {
    false = 0,
    true = 1
} bool;
#endif

typedef uint8_t *buffer_t;             // read-write buffer
typedef uint8_t *output_buffer_t;      // write-only buffer
typedef const uint8_t *input_buffer_t; // read-only buffer

#define to_struct(ptr, type, member) ((type*)((char *)(ptr) - offsetof(type, member)))

typedef struct {
    int page_no;
    int slot_no;
} rid_t;

typedef enum {
    ATTR_INT,
    ATTR_FLOAT,
    ATTR_STRING
} attr_type_t;

#define SM_MAX_FIELDS           32
#define SM_MAX_NAME_LEN         256
#define SM_MAX_FIELD_NAME_LEN   64      // Max length of field name
#define SM_MAX_TABLES           256     // Max number of tables in a database


// Error no
#define PF_ERROR_START  100
#define RM_ERROR_START  200
#define IX_ERROR_START  300
#define SM_ERROR_START  400
#define QL_ERROR_START  500

#ifdef __cplusplus
}
#endif
