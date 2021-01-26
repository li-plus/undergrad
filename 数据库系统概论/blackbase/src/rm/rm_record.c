#include "rm_record.h"
#include "rm_error.h"
#include "bitmap.h"
#include <assert.h>
#include <stdlib.h>

static inline void init_page_handle(rm_page_handle_t *ph, const rm_file_handle_t *fh, pf_page_ptr_t page) {
    ph->page = page;
    ph->hdr = (rm_page_hdr_t *) page->buffer;
    ph->bitmap = page->buffer + sizeof(rm_page_hdr_t);
    ph->slots = ph->bitmap + fh->hdr.bitmap_size;
}

static inline buffer_t rm_get_slot(const rm_file_handle_t *fh, const rm_page_handle_t *ph, int slot_no) {
    return ph->slots + slot_no * fh->hdr.record_size;
}

static inline RC rm_fetch_page(const rm_file_handle_t *fh, int page_no, rm_page_handle_t *ph) {
    assert(page_no < fh->hdr.num_pages);
    RC rc;
    pf_page_ptr_t page;
    rc = pf_fetch_page(fh->fd, page_no, &page);
    if (rc) { return rc; }
    init_page_handle(ph, fh, page);
    return 0;
}

static inline RC rm_get_free_page(rm_file_handle_t *fh, rm_page_handle_t *ph) {
    RC rc;
    if (fh->hdr.first_free_page == RM_NO_PAGE) {
        // No free pages. Need to allocate a new page.
        pf_page_ptr_t page;
        rc = pf_create_page(fh->fd, fh->hdr.num_pages, &page);
        if (rc) { return rc; }
        // Init page handle
        init_page_handle(ph, fh, page);
        ph->hdr->num_records = 0;
        ph->hdr->next_free = RM_NO_PAGE;
        bitmap_init(ph->bitmap, fh->hdr.bitmap_size);
        // Update file header
        fh->hdr.num_pages++;
        fh->hdr_dirty = true;
        fh->hdr.first_free_page = page->page_id.page_no;
    } else {
        // Fetch the first free page.
        rc = rm_fetch_page(fh, fh->hdr.first_free_page, ph);
        if (rc) { return rc; }
    }
    pf_mark_dirty(ph->page);
    return 0;
}

RC rm_get_record(const rm_file_handle_t *fh, const rid_t *rid, output_buffer_t buffer) {
    RC rc;
    rm_page_handle_t ph;
    rc = rm_fetch_page(fh, rid->page_no, &ph);
    if (rc) { return rc; }
    if (!bitmap_test(ph.bitmap, rid->slot_no)) {
        return RM_RECORD_NOT_FOUND;
    }
    buffer_t slot = rm_get_slot(fh, &ph, rid->slot_no);
    // should be malloced first
    memcpy(buffer, slot, fh->hdr.record_size);
    return 0;
}

// Insert record stored in data, and point rid to the inserted position
RC rm_insert_record(rm_file_handle_t *fh, input_buffer_t data, rid_t *rid) {
    RC rc;
    rm_page_handle_t ph;
    rc = rm_get_free_page(fh, &ph);
    if (rc) { return rc; }
    // get slot number
    int slot_no = bitmap_first_zero(ph.bitmap, fh->hdr.num_records_per_page);
    assert(slot_no < fh->hdr.num_records_per_page);
    // update bitmap
    bitmap_set(ph.bitmap, slot_no);
    // update page header
    ph.hdr->num_records++;
    if (ph.hdr->num_records == fh->hdr.num_records_per_page) {
        // page is full
        fh->hdr.first_free_page = ph.hdr->next_free;
        fh->hdr_dirty = true;
    }
    // copy record data into slot
    buffer_t slot = rm_get_slot(fh, &ph, slot_no);
    memcpy(slot, data, fh->hdr.record_size);
    rid->page_no = ph.page->page_id.page_no;
    rid->slot_no = slot_no;
    return 0;
}

RC rm_delete_record(rm_file_handle_t *fh, const rid_t *rid) {
    RC rc;
    rm_page_handle_t ph;
    rc = rm_fetch_page(fh, rid->page_no, &ph);
    if (rc) { return rc; }
    pf_mark_dirty(ph.page);
    if (!bitmap_test(ph.bitmap, rid->slot_no)) {
        return RM_RECORD_NOT_FOUND;
    }
    if (ph.hdr->num_records == fh->hdr.num_records_per_page) {
        // originally full, now available for new record
        ph.hdr->next_free = fh->hdr.first_free_page;
        fh->hdr.first_free_page = rid->page_no;
        fh->hdr_dirty = true;
    }
    bitmap_reset(ph.bitmap, rid->slot_no);
    ph.hdr->num_records--;
    return 0;
}

RC rm_update_record(rm_file_handle_t *fh, const rid_t *rid, input_buffer_t data) {
    RC rc;
    rm_page_handle_t ph;
    rc = rm_fetch_page(fh, rid->page_no, &ph);
    if (rc) { return rc; }
    pf_mark_dirty(ph.page);
    if (!bitmap_test(ph.bitmap, rid->slot_no)) {
        return RM_RECORD_NOT_FOUND;
    }
    buffer_t slot = rm_get_slot(fh, &ph, rid->slot_no);
    memcpy(slot, data, fh->hdr.record_size);
    return 0;
}

RC rm_scan_init(rm_file_handle_t *fh, rid_t *rid) {
    rid->page_no = RM_FIRST_RECORD_PAGE;
    rid->slot_no = -1;
    return rm_scan_next(fh, rid);
}

bool rm_scan_is_end(const rid_t *rid) {
    return rid->page_no == RM_NO_PAGE;
}

RC rm_scan_next(rm_file_handle_t *fh, rid_t *rid) {
    assert(!rm_scan_is_end(rid));
    RC rc;
    rm_page_handle_t ph;
    rid->slot_no++;
    while (rid->page_no < fh->hdr.num_pages) {
        rc = rm_fetch_page(fh, rid->page_no, &ph);
        if (rc) { return rc; }
        while (rid->slot_no < fh->hdr.num_records_per_page) {
            if (bitmap_test(ph.bitmap, rid->slot_no)) {
                // found next record
                return 0;
            }
            rid->slot_no++;
        }
        rid->slot_no = 0;
        rid->page_no++;
    }
    // next record not found
    rid->page_no = RM_NO_PAGE;
    return 0;
}

void rm_record_init(rm_record_t *rec, rm_file_handle_t *fh) {
    rec->size = fh->hdr.record_size;
    rec->data = (buffer_t) malloc(rec->size);
    rec->rid.page_no = RM_NO_PAGE;
    rec->rid.slot_no = -1;
}

void rm_record_destroy(rm_record_t *rec) {
    free(rec->data);
    rec->data = NULL;
}
