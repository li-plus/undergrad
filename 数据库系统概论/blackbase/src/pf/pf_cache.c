#include "pf_cache.h"
#include "pf_error.h"
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

pf_cache_t cache;

static inline bool pf_page_in_cache(pf_page_id_t *page_id) {
    return pf_hash_table_get(&cache.hash_table, page_id) != NULL;
}

RC pf_read_page(int fd, int page_no, output_buffer_t buf, int num_bytes) {
    lseek(fd, page_no * PAGE_SIZE, SEEK_SET);
    ssize_t bytes_read = read(fd, buf, num_bytes);
    if (bytes_read != num_bytes) {
        return PF_INCOMPLETE_READ;
    }
    return 0;
}

RC pf_write_page(int fd, int page_no, input_buffer_t buf, int num_bytes) {
    lseek(fd, page_no * PAGE_SIZE, SEEK_SET);
    ssize_t bytes_write = write(fd, buf, num_bytes);
    if (bytes_write != num_bytes) {
        return PF_INCOMPLETE_WRITE;
    }
    return 0;
}

static inline pf_page_ptr_t pf_find_replace() {
    assert(!list_empty(&cache.busy_pages));
    list_entry_t *le_page = list_rbegin(&cache.busy_pages);
    return le2page(le_page);
}

RC pf_force_page(pf_page_ptr_t page) {
    RC rc;
    if (page->is_dirty) {
        rc = pf_write_page(page->page_id.fd, page->page_id.page_no, page->buffer, PAGE_SIZE);
        if (rc) { return rc; }
        page->is_dirty = false;
    }
    return 0;
}

RC pf_init() {
    pf_hash_table_init(&cache.hash_table);
    list_init(&cache.busy_pages);
    list_init(&cache.free_pages);
    for (int i = 0; i < PF_MAX_CACHE_PAGES; i++) {
        cache.pages[i].buffer = cache.page_cache + i * PAGE_SIZE;
        list_push_back(&cache.free_pages, &cache.pages[i].le);
    }
    return 0;
}

// flush all pages
RC pf_clear() {
    RC rc;
    list_entry_t *le_page = list_begin(&cache.busy_pages);
    while (le_page != list_end(&cache.busy_pages)) {
        pf_page_ptr_t page = le2page(le_page);
        rc = pf_force_page(page);
        if (rc) { return rc; }
        list_entry_t *le_bak = le_page;
        le_page = list_next(le_page);
        list_erase(le_bak);
        list_push_back(&cache.free_pages, le_bak);
    }
    pf_hash_table_destroy(&cache.hash_table);
    return 0;
}

static RC _pf_alloc_page(int fd, int page_no, pf_page_ptr_t *out_page) {
    RC rc;
    pf_page_ptr_t page;
    pf_page_id_t page_id = {.fd = fd, .page_no = page_no};
    if (list_empty(&cache.free_pages)) {
        // Cache is full. Need to flush a page to disk.
        page = pf_find_replace();
        assert(page != NULL);
        rc = pf_force_page(page);
        if (rc) { return rc; }
        pf_hash_table_erase_entry(&cache.hash_table, page->hash);
    } else {
        // Allocate from free pages.
        list_entry_t *le_page = list_begin(&cache.free_pages);
        page = le2page(le_page);
    }
    page->page_id.fd = fd;
    page->page_id.page_no = page_no;
    page->is_dirty = false;
    list_erase(&page->le);
    list_push_front(&cache.busy_pages, &page->le);
    page->hash = pf_hash_table_insert(&cache.hash_table, &page_id, page);
    *out_page = page;
    return 0;
}

static RC _pf_get_page(int fd, int page_no, bool exists, pf_page_ptr_t *out_page) {
    RC rc;
    pf_page_id_t page_id = {.fd = fd, .page_no = page_no};
    pf_page_ptr_t page = pf_hash_table_get(&cache.hash_table, &page_id);
    if (page == NULL) {
        // Page is not in memory (i.e. on disk). Allocate new cache page for it.
        _pf_alloc_page(fd, page_no, &page);
        // If page already exists on disk, read it into memory.
        if (exists) {
            rc = pf_read_page(fd, page_no, page->buffer, PAGE_SIZE);
            if (rc) { return rc; }
        }
    }
    pf_access(page);
    *out_page = page;
    return 0;
}

RC pf_create_page(int fd, int page_no, pf_page_ptr_t *out_page) {
    RC rc;
    pf_page_ptr_t page;
    rc = _pf_get_page(fd, page_no, false, &page);
    if (rc) { return rc; }
    pf_mark_dirty(page);
    *out_page = page;
    return 0;
}

// Get the page from memory corresponding to the disk page.
// If the page if not in memory, allocate a page and read the disk.
RC pf_fetch_page(int fd, int page_no, pf_page_ptr_t *out_page) {
    RC rc;
    pf_page_ptr_t page;
    rc = _pf_get_page(fd, page_no, true, &page);
    if (rc) { return rc; }
    pf_access(page);
    *out_page = page;
    return 0;
}

RC pf_access(pf_page_ptr_t page) {
    assert(pf_page_in_cache(&page->page_id));
    list_erase(&page->le);
    list_push_front(&cache.busy_pages, &page->le);
    return 0;
}

void pf_mark_dirty(pf_page_ptr_t page) {
    assert(pf_page_in_cache(&page->page_id));
    page->is_dirty = true;
}

RC pf_flush_page(pf_page_ptr_t page) {
    RC rc;
    assert(pf_page_in_cache(&page->page_id));
    rc = pf_force_page(page);
    if (rc) { return rc; }
    list_erase(&page->le);
    list_push_front(&cache.free_pages, &page->le);
    pf_hash_table_erase_entry(&cache.hash_table, page->hash);
    assert(!pf_page_in_cache(&page->page_id));
    return 0;
}

RC pf_flush_file(int fd) {
    RC rc;
    list_entry_t *le_page = list_begin(&cache.busy_pages);
    while (le_page != list_end(&cache.busy_pages)) {
        pf_page_ptr_t page = le2page(le_page);
        le_page = list_next(le_page);
        if (page->page_id.fd == fd) {
            rc = pf_flush_page(page);
            if (rc) { return rc; }
        }
    }
    return 0;
}
