#include "rm_file.h"
#include "bitmap.h"

RC rm_create_file(const char *filename, int record_size) {
    RC rc;
    rc = pf_create_file(filename);
    if (rc) { return rc; }
    int fd;
    rc = pf_open_file(filename, &fd);
    if (rc) { return rc; }

    rm_file_hdr_t hdr;
    hdr.record_size = record_size;
    hdr.num_pages = 1;
    hdr.first_free_page = RM_NO_PAGE;
    // We have: sizeof(hdr) + (n + 7) / 8 + n * record_size <= PAGE_SIZE
    hdr.num_records_per_page = (BITMAP_WIDTH * (PAGE_SIZE - 1 - (int) sizeof(rm_file_hdr_t)) + 1) /
                               (1 + record_size * BITMAP_WIDTH);
    hdr.bitmap_size = (hdr.num_records_per_page + BITMAP_WIDTH - 1) / BITMAP_WIDTH;
    rc = pf_write_page(fd, RM_FILE_HDR_PAGE, (buffer_t) &hdr, sizeof(hdr));
    if (rc) { return rc; }
    rc = pf_close_file(fd);
    if (rc) { return rc; }
    return 0;
}

RC rm_destroy_file(const char *filename) {
    return pf_destroy_file(filename);
}

RC rm_open_file(const char *filename, rm_file_handle_t *fh) {
    RC rc;
    rc = pf_open_file(filename, &fh->fd);
    if (rc) { return rc; }
    rc = pf_read_page(fh->fd, RM_FILE_HDR_PAGE, (buffer_t) &fh->hdr, sizeof(fh->hdr));
    if (rc) { return rc; }
    fh->hdr_dirty = false;
    return rc;
}

RC rm_close_file(rm_file_handle_t *fh) {
    RC rc;
    if (fh->hdr_dirty) {
        rc = pf_write_page(fh->fd, RM_FILE_HDR_PAGE, (buffer_t) &fh->hdr, sizeof(fh->hdr));
        if (rc) { return rc; }
    }
    return pf_close_file(fh->fd);
}
