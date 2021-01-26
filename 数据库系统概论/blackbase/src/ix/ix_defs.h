#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "pf/pf.h"
#include "defs.h"

typedef struct {
    int first_free;
    int num_pages;              // # disk pages
    int root_page;              // root page no
    attr_type_t attr_type;
    int attr_len;
    int btree_order;     // # children per page
    int key_offset;             // offset of key array
    int rid_offset;             // offset of rid array (children array)
    int first_leaf;
    int last_leaf;
} ix_file_hdr_t;

typedef struct {
    int fd;
    ix_file_hdr_t hdr;
    bool hdr_dirty;
} ix_file_handle_t;

typedef struct {
    int next_free;
    int parent;
    int num_key;        // # current keys (always equals to #child - 1)
    int num_child;      // # current children
    bool is_leaf;
    int prev_leaf;      // previous leaf node, effective only when is_leaf is true
    int next_leaf;      // next leaf node, effective only when is_leaf is true
} ix_page_hdr_t;

typedef struct {
    ix_page_hdr_t *hdr;
    buffer_t p_key;
    rid_t *p_rid;
    pf_page_ptr_t page;
} ix_page_handle_t;

typedef struct {
    int page_no;
    int slot_no;
} iid_t;

#define IX_NO_PAGE          (-1)
#define IX_FILE_HDR_PAGE    0
#define IX_LEAF_HEADER_PAGE 1
#define IX_INIT_ROOT_PAGE   2
#define IX_INIT_NUM_PAGES   3

#ifdef __cplusplus
}
#endif
