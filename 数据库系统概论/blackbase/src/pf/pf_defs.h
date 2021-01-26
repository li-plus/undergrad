#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "list.h"

#define PAGE_SIZE 4096

typedef struct {
    int fd;
    int page_no;
} pf_page_id_t;

struct pf_hash_entry;

typedef struct {
    list_entry_t le;
    struct pf_hash_entry *hash;
    pf_page_id_t page_id;
    buffer_t buffer;
    bool is_dirty;
} pf_page_t;

typedef pf_page_t *pf_page_ptr_t;

#define le2page(ptr) to_struct((ptr), pf_page_t, le)

#define PF_MAX_CACHE_PAGES  65536   // Number of cache pages
#define PF_HASH_BUCKET_SIZE 131071  // Max size of hashmap: should be a prime

#ifdef __cplusplus
}
#endif
