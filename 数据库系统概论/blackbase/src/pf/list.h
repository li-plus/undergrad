#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "defs.h"
#include <stddef.h>

typedef struct list_entry {
    struct list_entry *prev;
    struct list_entry *next;
} list_entry_t;

static inline void list_init(list_entry_t *header) {
    header->next = header->prev = header;
}

static inline bool list_empty(list_entry_t *header) {
    return (header->next == header->prev) && (header->next == header);
}

static inline list_entry_t *list_next(list_entry_t *pos) {
    return pos->next;
}

static inline list_entry_t *list_begin(list_entry_t *header) {
    return header->next;
}

static inline list_entry_t *list_end(list_entry_t *header) {
    return header;
}

static inline list_entry_t *list_rbegin(list_entry_t *header) {
    return header->prev;
}

static inline void list_insert(list_entry_t *pos, list_entry_t *node) {
    node->prev = pos->prev;
    node->next = pos;
    pos->prev = node;
    node->prev->next = node;
}

static inline void list_erase(list_entry_t *pos) {
    list_entry_t *prev = pos->prev;
    list_entry_t *next = pos->next;
    prev->next = next;
    next->prev = prev;
}

static inline void list_push_back(list_entry_t *header, list_entry_t *node) {
    list_insert(header, node);
}

static inline void list_push_front(list_entry_t *header, list_entry_t *node) {
    list_insert(header->next, node);
}

#ifdef __cplusplus
}
#endif
