#include "ix_file.h"
#include "ix_error.h"
#include <string.h>
#include <assert.h>

char *ix_get_filename(const char *filename, int index_no) {
    static char ix_filename[512];
    sprintf(ix_filename, "%s.%d.idx", filename, index_no);
    return ix_filename;
}

bool ix_exists(const char *filename, int index_no) {
    char *ix_filename = ix_get_filename(filename, index_no);
    return pf_exists(ix_filename);
}

RC ix_create_index(const char *filename, int index_no, attr_type_t attr_type, int attr_len) {
    if (filename == NULL) {
        return IX_BAD_FILENAME;
    }
    assert(index_no >= 0);
    RC rc;
    char *ix_filename = ix_get_filename(filename, index_no);
    // Create index file
    rc = pf_create_file(ix_filename);
    if (rc) { return rc; }
    int fd;
    // Open index file
    rc = pf_open_file(ix_filename, &fd);
    if (rc) { return rc; }

    // Create file header and write to file
    // Theoretically we have: |page_hdr| + (|attr| + |rid|) * n <= PAGE_SIZE
    // but we reserve one slot for convenient inserting and deleting, i.e.
    // |page_hdr| + (|attr| + |rid|) * (n + 1) <= PAGE_SIZE
    int btree_order = (PAGE_SIZE - sizeof(ix_page_hdr_t)) / (attr_len + sizeof(rid_t)) - 1;
    assert(btree_order > 2);
    int key_offset = sizeof(ix_page_hdr_t);
    int rid_offset = key_offset + (btree_order + 1) * attr_len;

    ix_file_hdr_t fhdr = {
            .first_free = IX_NO_PAGE,
            .num_pages = IX_INIT_NUM_PAGES,
            .root_page = IX_INIT_ROOT_PAGE,
            .attr_type = attr_type,
            .attr_len = attr_len,
            .btree_order = btree_order,
            .key_offset = key_offset,
            .rid_offset = rid_offset,
            .first_leaf = IX_INIT_ROOT_PAGE,
            .last_leaf = IX_INIT_ROOT_PAGE,
    };
    rc = pf_write_page(fd, IX_FILE_HDR_PAGE, (buffer_t) &fhdr, sizeof(fhdr));
    if (rc) { return rc; }

    // Create leaf list header page and write to file
    {
        ix_page_hdr_t phdr = {
                .next_free = IX_NO_PAGE,
                .parent = IX_NO_PAGE,
                .num_key = 0,
                .num_child = 0,
                .is_leaf = true,
                .prev_leaf = IX_INIT_ROOT_PAGE,
                .next_leaf = IX_INIT_ROOT_PAGE,
        };
        rc = pf_write_page(fd, IX_LEAF_HEADER_PAGE, (input_buffer_t) &phdr, sizeof(phdr));
        if (rc) { return rc; }
    }

    // Create root node and write to file
    {
        static uint8_t page_buf[PAGE_SIZE];
        ix_page_hdr_t phdr = {
                .next_free = IX_NO_PAGE,
                .parent = IX_NO_PAGE,
                .num_key = 0,
                .num_child = 0,
                .is_leaf = true,
                .prev_leaf = IX_LEAF_HEADER_PAGE,
                .next_leaf = IX_LEAF_HEADER_PAGE,
        };
        memcpy(page_buf, &phdr, sizeof(phdr));
        // Must write PAGE_SIZE here in case of future fetch_page()
        rc = pf_write_page(fd, IX_INIT_ROOT_PAGE, page_buf, PAGE_SIZE);
        if (rc) { return rc; }
    }

    // Close index file
    rc = pf_close_file(fd);
    if (rc) { return rc; }
    return 0;
}

RC ix_destroy_index(const char *filename, int index_no) {
    char *ix_filename = ix_get_filename(filename, index_no);
    return pf_destroy_file(ix_filename);
}

RC ix_open_index(const char *filename, int index_no, ix_file_handle_t *fh) {
    char *ix_filename = ix_get_filename(filename, index_no);
    RC rc;
    rc = pf_open_file(ix_filename, &fh->fd);
    if (rc) { return rc; }
    rc = pf_read_page(fh->fd, IX_FILE_HDR_PAGE, (buffer_t) &fh->hdr, sizeof(fh->hdr));
    if (rc) { return rc; }
    return 0;
}

RC ix_close_index(ix_file_handle_t *ih) {
    RC rc;
    if (ih->hdr_dirty) {
        rc = pf_write_page(ih->fd, IX_FILE_HDR_PAGE, (buffer_t) &ih->hdr, sizeof(ih->hdr));
        if (rc) { return rc; }
    }
    rc = pf_close_file(ih->fd);
    if (rc) { return rc; }
    return 0;
}
