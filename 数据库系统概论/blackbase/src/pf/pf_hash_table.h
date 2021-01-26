#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "pf_defs.h"
#include "list.h"

typedef struct pf_hash_entry {
    list_entry_t le_hash;
    pf_page_id_t page_id;       // key
    pf_page_ptr_t page;         // value
} pf_hash_entry_t;

#define le2hash(ptr) to_struct((ptr), pf_hash_entry_t, le_hash)

typedef struct {
    list_entry_t buckets[PF_HASH_BUCKET_SIZE];
    pf_hash_entry_t hashes[PF_MAX_CACHE_PAGES]; // pre-alloc hash entries
    list_entry_t free_hashes;   // free hash entries
} pf_hash_table_t;

static inline uint32_t pf_hash(const pf_page_id_t *key) {
    return (key->fd << 16) | key->page_no;
}

static inline bool pf_hash_key_equals(const pf_page_id_t *x, const pf_page_id_t *y) {
    return x->fd == y->fd && x->page_no == y->page_no;
}

void pf_hash_table_init(pf_hash_table_t *ht);

void pf_hash_table_destroy(pf_hash_table_t *ht);

pf_page_ptr_t pf_hash_table_get(pf_hash_table_t *ht, const pf_page_id_t *key);

pf_hash_entry_t *pf_hash_table_insert(pf_hash_table_t *ht, const pf_page_id_t *key, pf_page_ptr_t value);

void pf_hash_table_erase_entry(pf_hash_table_t *ht, pf_hash_entry_t *entry);

void pf_hash_table_erase(pf_hash_table_t *ht, const pf_page_id_t *key);

#ifdef __cplusplus
}
#endif
