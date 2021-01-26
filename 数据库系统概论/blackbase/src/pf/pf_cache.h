#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "pf_hash_table.h"
#include "pf_defs.h"

typedef struct {
    uint8_t page_cache[PF_MAX_CACHE_PAGES * PAGE_SIZE];
    pf_page_t pages[PF_MAX_CACHE_PAGES];    // page nodes
    list_entry_t busy_pages;
    list_entry_t free_pages;
    pf_hash_table_t hash_table;
} pf_cache_t;

extern pf_cache_t cache;

RC pf_init();

RC pf_clear();

// Allocate a new page on cache. If marked dirty, will be written to disk later.
// If page already exists on memory or disk, will cause error.
RC pf_create_page(int fd, int page_no, pf_page_ptr_t *out_page);

// Get the page from memory or disk.
// If page is on memory, point buffer to the page cache.
// If page is on disk, load it into memory and point buffer to the page cache.
// If page is on neither memory or disk, will cause error.
RC pf_fetch_page(int fd, int page_no, pf_page_ptr_t *out_page);

RC pf_access(pf_page_ptr_t page);

void pf_mark_dirty(pf_page_ptr_t page);

RC pf_force_page(pf_page_ptr_t page);

RC pf_flush_page(pf_page_ptr_t page);

RC pf_flush_file(int fd);

RC pf_read_page(int fd, int page_no, output_buffer_t buf, int num_bytes);

RC pf_write_page(int fd, int page_no, input_buffer_t buf, int num_bytes);

#ifdef __cplusplus
}
#endif
