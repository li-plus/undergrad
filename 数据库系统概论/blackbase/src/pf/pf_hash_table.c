#include "pf_hash_table.h"
#include <assert.h>
#include <string.h>

void pf_hash_table_init(pf_hash_table_t *ht) {
    for (int i = 0; i < PF_HASH_BUCKET_SIZE; i++) {
        list_init(&ht->buckets[i]);
    }
    list_init(&ht->free_hashes);
    for (int i = 0; i < PF_MAX_CACHE_PAGES; i++) {
        list_push_back(&ht->free_hashes, &ht->hashes[i].le_hash);
    }
}

void pf_hash_table_destroy(pf_hash_table_t *ht) {
    pf_hash_table_init(ht);
}

static inline list_entry_t *pf_get_bucket(pf_hash_table_t *ht, const pf_page_id_t *key) {
    int bkt_id = pf_hash(key) % PF_HASH_BUCKET_SIZE;
    return &ht->buckets[bkt_id];
}

static list_entry_t *pf_hash_table_find_in_bkt(const pf_page_id_t *key, list_entry_t *bkt) {
    list_entry_t *le_hash = list_begin(bkt);
    while (le_hash != list_end(bkt)) {
        pf_hash_entry_t *entry = le2hash(le_hash);
        if (pf_hash_key_equals(&entry->page_id, key)) {
            break;
        }
        le_hash = list_next(le_hash);
    }
    return le_hash;
}

pf_page_ptr_t pf_hash_table_get(pf_hash_table_t *ht, const pf_page_id_t *key) {
    list_entry_t *bkt = pf_get_bucket(ht, key);
    list_entry_t *le_hash = pf_hash_table_find_in_bkt(key, bkt);
    if (le_hash == list_end(bkt)) {
        return NULL;
    }
    pf_hash_entry_t *entry = le2hash(le_hash);
    return entry->page;
}

pf_hash_entry_t *pf_hash_table_insert(pf_hash_table_t *ht, const pf_page_id_t *key, pf_page_ptr_t value) {
    // TODO: remove assert in production
    list_entry_t *bkt = pf_get_bucket(ht, key);
    assert(pf_hash_table_find_in_bkt(key, bkt) == list_end(bkt));
    assert(!list_empty(&ht->free_hashes));
    list_entry_t *le_hash = list_begin(&ht->free_hashes);
    list_erase(le_hash);
    pf_hash_entry_t *entry = le2hash(le_hash);
    memcpy(&entry->page_id, key, sizeof(pf_page_id_t));
    entry->page = value;
    list_push_front(bkt, &entry->le_hash);
    return entry;
}

void pf_hash_table_erase_entry(pf_hash_table_t *ht, pf_hash_entry_t *entry) {
    list_erase(&entry->le_hash);
    list_push_front(&ht->free_hashes, &entry->le_hash);
}

void pf_hash_table_erase(pf_hash_table_t *ht, const pf_page_id_t *key) {
    list_entry_t *bkt = pf_get_bucket(ht, key);
    list_entry_t *le_hash = pf_hash_table_find_in_bkt(key, bkt);
    assert(le_hash != list_end(bkt));   // assert exists
    pf_hash_table_erase_entry(ht, le2hash(le_hash));
}
