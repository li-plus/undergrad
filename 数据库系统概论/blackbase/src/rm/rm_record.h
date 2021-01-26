#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "rm_defs.h"

RC rm_get_record(const rm_file_handle_t *fh, const rid_t *rid, output_buffer_t buffer);

// Insert record stored in data, and point rid to the inserted position
RC rm_insert_record(rm_file_handle_t *fh, input_buffer_t data, rid_t *rid);

RC rm_delete_record(rm_file_handle_t *fh, const rid_t *rid);

RC rm_update_record(rm_file_handle_t *fh, const rid_t *rid, input_buffer_t data);

RC rm_scan_init(rm_file_handle_t *fh, rid_t *rid);

RC rm_scan_next(rm_file_handle_t *fh, rid_t *rid);

bool rm_scan_is_end(const rid_t *rec);

void rm_record_init(rm_record_t *rec, rm_file_handle_t *fh);

void rm_record_destroy(rm_record_t *rec);

#ifdef __cplusplus
}
#endif
