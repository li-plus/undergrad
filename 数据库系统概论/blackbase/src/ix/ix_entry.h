#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "ix_defs.h"
#include "defs.h"

RC ix_fetch_page(const ix_file_handle_t *fh, int page_no, ix_page_handle_t *ph);

RC ix_get_rid(const ix_file_handle_t *fh, const iid_t *iid, rid_t *out_rid);

RC ix_get_entry(const ix_file_handle_t *fh, const iid_t *iid, output_buffer_t out_key, rid_t *out_rid);

RC ix_lower_bound(const ix_file_handle_t *fh, input_buffer_t key, iid_t *iid);

RC ix_upper_bound(const ix_file_handle_t *fh, input_buffer_t key, iid_t *iid);

RC ix_insert_entry(ix_file_handle_t *fh, input_buffer_t key, const rid_t *rid);

RC ix_delete_entry(ix_file_handle_t *fh, input_buffer_t key, const rid_t *rid);

bool ix_scan_equal(const iid_t *x, const iid_t *y);

void ix_leaf_begin(const ix_file_handle_t *fh, iid_t *iid);

RC ix_leaf_end(const ix_file_handle_t *fh, iid_t *iid);

RC ix_scan_next(const ix_file_handle_t *fh, iid_t *iid);

#ifdef __cplusplus
}
#endif
