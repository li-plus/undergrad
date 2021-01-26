#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "pf/pf.h"
#include "defs.h"

#define RM_NO_PAGE              (-1)
#define RM_FILE_HDR_PAGE        0
#define RM_FIRST_RECORD_PAGE    1

typedef struct {
    int record_size;
    int num_pages;
    int num_records_per_page;
    int first_free_page;
    int bitmap_size;
} rm_file_hdr_t;

typedef struct {
    int next_free;
    int num_records;
} rm_page_hdr_t;

typedef struct {
    rm_file_hdr_t hdr;
    bool hdr_dirty;
    int fd;
} rm_file_handle_t;

typedef struct {
    rm_page_hdr_t *hdr;
    buffer_t bitmap;
    buffer_t slots;
    pf_page_ptr_t page;
} rm_page_handle_t;

typedef struct {
    rid_t rid;
    buffer_t data;
    int size;
} rm_record_t;

#ifdef __cplusplus
}
#endif
