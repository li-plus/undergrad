#include "ix_entry.h"
#include "ix_error.h"
#include "ix_compare.h"
#include <string.h>

static bool binary_search = false;

static inline void ix_page_handle_init(ix_page_handle_t *ph, const ix_file_handle_t *fh, pf_page_ptr_t page) {
    ph->hdr = (ix_page_hdr_t *) page->buffer;
    ph->p_key = page->buffer + fh->hdr.key_offset;
    ph->p_rid = (rid_t *) (page->buffer + fh->hdr.rid_offset);
    ph->page = page;
}

static RC ix_get_free_page(ix_file_handle_t *fh, ix_page_handle_t *ph) {
    RC rc;
    pf_page_ptr_t page;
    if (fh->hdr.first_free == IX_NO_PAGE) {
        rc = pf_create_page(fh->fd, fh->hdr.num_pages, &page);
        if (rc) { return rc; }
        ix_page_handle_init(ph, fh, page);
        fh->hdr.num_pages++;
    } else {
        rc = pf_fetch_page(fh->fd, fh->hdr.first_free, &page);
        if (rc) { return rc; }
        ix_page_handle_init(ph, fh, page);
        fh->hdr.first_free = ph->hdr->next_free;
    }
    pf_mark_dirty(page);
    fh->hdr_dirty = true;
    return 0;
}

RC ix_fetch_page(const ix_file_handle_t *fh, int page_no, ix_page_handle_t *ph) {
    RC rc;
    assert(page_no < fh->hdr.num_pages);
    pf_page_ptr_t page;
    rc = pf_fetch_page(fh->fd, page_no, &page);
    if (rc) { return rc; }
    ix_page_handle_init(ph, fh, page);
    return 0;
}

static inline buffer_t ix_get_key(const ix_file_handle_t *fh, const ix_page_handle_t *ph, int key_idx) {
    return ph->p_key + key_idx * fh->hdr.attr_len;
}

RC ix_get_rid(const ix_file_handle_t *fh, const iid_t *iid, rid_t *out_rid) {
    RC rc;
    ix_page_handle_t ph;
    rc = ix_fetch_page(fh, iid->page_no, &ph);
    if (rc) { return rc; }
    if (iid->slot_no >= ph.hdr->num_key) { return IX_ENTRY_NOT_FOUND; }
    memcpy(out_rid, &ph.p_rid[iid->slot_no], sizeof(rid_t));
    return 0;
}

RC ix_get_entry(const ix_file_handle_t *fh, const iid_t *iid, output_buffer_t out_key, rid_t *out_rid) {
    RC rc;
    ix_page_handle_t ph;
    rc = ix_fetch_page(fh, iid->page_no, &ph);
    if (rc) { return rc; }
    if (iid->slot_no >= ph.hdr->num_key) { return IX_ENTRY_NOT_FOUND; }
    memcpy(out_key, ix_get_key(fh, &ph, iid->slot_no), fh->hdr.attr_len);
    memcpy(out_rid, &ph.p_rid[iid->slot_no], sizeof(rid_t));
    return 0;
}

static int ix_page_lower_bound(const ix_file_handle_t *fh, const ix_page_handle_t *ph, input_buffer_t key) {
    if (binary_search) {
        int lo = 0, hi = ph->hdr->num_key;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            buffer_t key_addr = ix_get_key(fh, ph, mid);
            if (ix_compare(key, key_addr, fh->hdr.attr_type, fh->hdr.attr_len) <= 0) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    } else {
        int key_idx = 0;
        while (key_idx < ph->hdr->num_key) {
            buffer_t key_addr = ix_get_key(fh, ph, key_idx);
            if (ix_compare(key, key_addr, fh->hdr.attr_type, fh->hdr.attr_len) <= 0) {
                break;
            }
            key_idx++;
        }
        return key_idx;
    }
}

RC ix_lower_bound(const ix_file_handle_t *fh, input_buffer_t key, iid_t *iid) {
    RC rc;
    ix_page_handle_t ph;

    rc = ix_fetch_page(fh, fh->hdr.root_page, &ph);
    if (rc) { return rc; }
    // Travel through inner nodes
    while (!ph.hdr->is_leaf) {
        int key_idx = ix_page_lower_bound(fh, &ph, key);
        if (key_idx >= ph.hdr->num_key) { return ix_leaf_end(fh, iid); }
        rid_t *child = &ph.p_rid[key_idx];
        rc = ix_fetch_page(fh, child->page_no, &ph);
        if (rc) { return rc; }
    }
    // Now we come to a leaf node, we do a sequential search
    int key_idx = ix_page_lower_bound(fh, &ph, key);
    iid->page_no = ph.page->page_id.page_no;
    iid->slot_no = key_idx;
    return 0;
}

static int ix_page_upper_bound(const ix_file_handle_t *fh, const ix_page_handle_t *ph, input_buffer_t key) {
    if (binary_search) {
        int lo = 0, hi = ph->hdr->num_key;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            buffer_t key_addr = ix_get_key(fh, ph, mid);
            if (ix_compare(key, key_addr, fh->hdr.attr_type, fh->hdr.attr_len) < 0) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    } else {
        int key_idx = 0;
        while (key_idx < ph->hdr->num_key) {
            buffer_t key_addr = ix_get_key(fh, ph, key_idx);
            if (ix_compare(key, key_addr, fh->hdr.attr_type, fh->hdr.attr_len) < 0) {
                break;
            }
            key_idx++;
        }
        return key_idx;
    }
}

RC ix_upper_bound(const ix_file_handle_t *fh, input_buffer_t key, iid_t *iid) {
    RC rc;
    ix_page_handle_t ph;

    rc = ix_fetch_page(fh, fh->hdr.root_page, &ph);
    if (rc) { return rc; }
    // Travel through inner nodes
    while (!ph.hdr->is_leaf) {
        int key_idx = ix_page_upper_bound(fh, &ph, key);
        if (key_idx >= ph.hdr->num_key) { return ix_leaf_end(fh, iid); }
        rid_t *child = &ph.p_rid[key_idx];
        rc = ix_fetch_page(fh, child->page_no, &ph);
        if (rc) { return rc; }
    }
    // Now we come to a leaf node, we do a sequential search
    int key_idx = ix_page_upper_bound(fh, &ph, key);
    iid->page_no = ph.page->page_id.page_no;
    iid->slot_no = key_idx;
    return 0;
}

static inline void ix_page_insert_keys(ix_file_handle_t *fh, ix_page_handle_t *ph, int pos, input_buffer_t key, int n) {
    buffer_t key_slot = ix_get_key(fh, ph, pos);
    memmove(key_slot + n * fh->hdr.attr_len, key_slot, (ph->hdr->num_key - pos) * fh->hdr.attr_len);
    memcpy(key_slot, key, n * fh->hdr.attr_len);
    ph->hdr->num_key += n;
}

static inline void ix_page_insert_key(ix_file_handle_t *fh, ix_page_handle_t *ph, int pos, input_buffer_t key) {
    ix_page_insert_keys(fh, ph, pos, key, 1);
}

static inline void ix_page_erase_key(ix_file_handle_t *fh, ix_page_handle_t *ph, int pos) {
    buffer_t key = ix_get_key(fh, ph, pos);
    memmove(key, key + fh->hdr.attr_len, (ph->hdr->num_key - pos - 1) * fh->hdr.attr_len);
    ph->hdr->num_key--;
}

static inline void ix_page_insert_rids(ix_page_handle_t *ph, int pos, const rid_t *rid, int n) {
    rid_t *rid_slot = &ph->p_rid[pos];
    memmove(rid_slot + n, rid_slot, (ph->hdr->num_child - pos) * sizeof(rid_t));
    memcpy(rid_slot, rid, n * sizeof(rid_t));
    ph->hdr->num_child += n;
}

static inline void ix_page_insert_rid(ix_page_handle_t *ph, int pos, const rid_t *rid) {
    ix_page_insert_rids(ph, pos, rid, 1);
}

static inline void ix_page_erase_rid(ix_page_handle_t *ph, int pos) {
    rid_t *rid = &ph->p_rid[pos];
    memmove(rid, rid + 1, (ph->hdr->num_child - pos - 1) * sizeof(rid_t));
    ph->hdr->num_child--;
}

static inline int ix_get_child_idx(const ix_page_handle_t *parent, const ix_page_handle_t *child) {
    int child_idx = 0;
    while (parent->p_rid[child_idx].page_no != child->page->page_id.page_no) { child_idx++; }
    assert(child_idx < parent->hdr->num_child);
    return child_idx;
}

static RC maintain_parent(ix_file_handle_t *fh, const ix_page_handle_t *leaf) {
    RC rc;
//    assert(leaf->hdr->is_leaf);
    ix_page_handle_t child, parent;
    memcpy(&child, leaf, sizeof(ix_page_handle_t));
    while (child.hdr->parent != IX_NO_PAGE) {
        // Load its parent
        rc = ix_fetch_page(fh, child.hdr->parent, &parent);
        if (rc) { return rc; }
        int child_idx = ix_get_child_idx(&parent, &child);
        buffer_t parent_key = ix_get_key(fh, &parent, child_idx);
        buffer_t child_max_key = ix_get_key(fh, &child, child.hdr->num_key - 1);
        if (memcmp(parent_key, child_max_key, fh->hdr.attr_len) == 0) {
            break;
        }
        pf_mark_dirty(parent.page);
        memcpy(parent_key, child_max_key, fh->hdr.attr_len);
        memcpy(&child, &parent, sizeof(ix_page_handle_t));
    }
    return 0;
}

RC ix_insert_entry(ix_file_handle_t *fh, input_buffer_t key, const rid_t *rid) {
    RC rc;
    iid_t iid;
    rc = ix_upper_bound(fh, key, &iid);
    if (rc) { return rc; }
    ix_page_handle_t ph;
    rc = ix_fetch_page(fh, iid.page_no, &ph);
    if (rc) { return rc; }
    pf_mark_dirty(ph.page);
    // We need to insert at iid.slot_no
    ix_page_insert_key(fh, &ph, iid.slot_no, key);
    ix_page_insert_rid(&ph, iid.slot_no, rid);
    // Maintain parents' max key
    if (iid.page_no == fh->hdr.last_leaf && iid.slot_no == ph.hdr->num_key - 1) {
        // Max key updated
        maintain_parent(fh, &ph);
    }
    // Solve overflow
    while (ph.hdr->num_child > fh->hdr.btree_order) {
        // If leaf node is overflowed, we need to split it
        if (ph.hdr->parent == IX_NO_PAGE) {
            // If current page is root node, allocate new root
            ix_page_handle_t root_ph;
            rc = ix_get_free_page(fh, &root_ph);
            if (rc) { return rc; }
            root_ph.hdr->next_free = IX_NO_PAGE;
            root_ph.hdr->parent = IX_NO_PAGE;
            root_ph.hdr->is_leaf = false;
            root_ph.hdr->num_key = 0;
            root_ph.hdr->num_child = 0;
            // Insert current node's key
            ix_page_insert_key(fh, &root_ph, 0, ix_get_key(fh, &ph, ph.hdr->num_key - 1));
            rid_t curr_rid = {.page_no = ph.page->page_id.page_no, .slot_no = 0};
            ix_page_insert_rid(&root_ph, 0, &curr_rid);
            // update current node's parent
            ph.hdr->parent = root_ph.page->page_id.page_no;
            // update global root page
            fh->hdr.root_page = root_ph.page->page_id.page_no;
        }
        // Split at middle position
        int split_idx = ph.hdr->num_child / 2;
        // Allocate brother node
        ix_page_handle_t bro_ph;
        rc = ix_get_free_page(fh, &bro_ph);
        if (rc) { return rc; }
        pf_mark_dirty(bro_ph.page);
        bro_ph.hdr->next_free = IX_NO_PAGE;
        bro_ph.hdr->num_key = 0;
        bro_ph.hdr->num_child = 0;
        bro_ph.hdr->is_leaf = ph.hdr->is_leaf;  // Brother node is leaf only if current node is leaf.
        bro_ph.hdr->parent = ph.hdr->parent;    // They have the same parent
        if (bro_ph.hdr->is_leaf) {
            // maintain brother node's leaf pointer
            bro_ph.hdr->next_leaf = ph.hdr->next_leaf;
            bro_ph.hdr->prev_leaf = ph.page->page_id.page_no;
            // Let original next node's prev = brother node
            ix_page_handle_t next_ph;
            rc = ix_fetch_page(fh, ph.hdr->next_leaf, &next_ph);
            if (rc) { return rc; }
            pf_mark_dirty(next_ph.page);
            next_ph.hdr->prev_leaf = bro_ph.page->page_id.page_no;
            // curr's next = brother node
            ph.hdr->next_leaf = bro_ph.page->page_id.page_no;
        }
        // Keys in [0, split_idx) stay in current node, [split_idx, curr_keys) go to brother node
        ix_page_insert_keys(fh, &bro_ph, 0, ix_get_key(fh, &ph, split_idx), ph.hdr->num_key - split_idx);
        ix_page_insert_rids(&bro_ph, 0, ph.p_rid + split_idx, ph.hdr->num_key - split_idx);
        assert(ph.hdr->num_child == ph.hdr->num_key);
        ph.hdr->num_key = split_idx;
        ph.hdr->num_child = split_idx;
        // Update children's parent
        if (!ph.hdr->is_leaf) {
            for (int child_idx = 0; child_idx < bro_ph.hdr->num_child; child_idx++) {
                ix_page_handle_t child_ph;
                rid_t *child_rid = &bro_ph.p_rid[child_idx];
                rc = ix_fetch_page(fh, child_rid->page_no, &child_ph);
                if (rc) { return rc; }
                pf_mark_dirty(child_ph.page);
                child_ph.hdr->parent = bro_ph.page->page_id.page_no;
            }
        }
        // Copy the last key up to its parent
        buffer_t popup_key = ix_get_key(fh, &ph, split_idx - 1);
        // Load parent node
        ix_page_handle_t parent_ph;
        rc = ix_fetch_page(fh, ph.hdr->parent, &parent_ph);
        if (rc) { return rc; }
        pf_mark_dirty(parent_ph.page);
        // Find the rank of current node in its parent
#if 0
        // If you use lower bound, you may use page_lower_bound to find the child_idx
        int child_idx = _ix_page_lower_bound(fh, &parent_ph, popup_key);
        assert(parent_ph.p_rid[child_idx].page_no == ph.page->page_id.page_no);
#else
        // If upper bound, do a sequential search to find the rank of the node
        int child_idx = ix_get_child_idx(&parent_ph, &ph);
#endif
        // Insert popup key into parent
        ix_page_insert_key(fh, &parent_ph, child_idx, popup_key);
        rid_t bro_page = {.page_no = bro_ph.page->page_id.page_no, .slot_no = 0};
        ix_page_insert_rid(&parent_ph, child_idx + 1, &bro_page);
        // Update global last_leaf if needed
        if (fh->hdr.last_leaf == ph.page->page_id.page_no) {
            fh->hdr.last_leaf = bro_ph.page->page_id.page_no;
        }
        // Go to its parent
        ph = parent_ph;
    }
    return 0;
}

static RC ix_erase_leaf(ix_file_handle_t *fh, ix_page_handle_t *leaf) {
    RC rc;
    assert(leaf->hdr->is_leaf);
    ix_page_handle_t prev;
    rc = ix_fetch_page(fh, leaf->hdr->prev_leaf, &prev);
    if (rc) { return rc; }
    pf_mark_dirty(prev.page);
    prev.hdr->next_leaf = leaf->hdr->next_leaf;

    ix_page_handle_t next;
    rc = ix_fetch_page(fh, leaf->hdr->next_leaf, &next);
    if (rc) { return rc; }
    pf_mark_dirty(next.page);
    next.hdr->prev_leaf = leaf->hdr->prev_leaf;
    return 0;
}

static void ix_release_page(ix_file_handle_t *fh, ix_page_handle_t *ph) {
    ph->hdr->next_free = fh->hdr.first_free;
    fh->hdr.first_free = ph->page->page_id.page_no;
    fh->hdr_dirty = true;
}

RC maintain_child(ix_file_handle_t *fh, ix_page_handle_t *ph, int child_idx) {
    RC rc;
    if (!ph->hdr->is_leaf) {
        // Current node is inner node, load its child and set its parent to current node
        ix_page_handle_t child;
        rc = ix_fetch_page(fh, ph->p_rid[child_idx].page_no, &child);
        if (rc) { return rc; }
        pf_mark_dirty(child.page);
        child.hdr->parent = ph->page->page_id.page_no;
    }
    return 0;
}

RC ix_delete_entry(ix_file_handle_t *fh, input_buffer_t key, const rid_t *rid) {
    RC rc;
    iid_t iid, upper;
    rc = ix_lower_bound(fh, key, &iid);
    if (rc) { return rc; }
    rc = ix_upper_bound(fh, key, &upper);
    if (rc) { return rc; }
    while (!ix_scan_equal(&iid, &upper)) {
        // load btree node
        ix_page_handle_t ph;
        rc = ix_fetch_page(fh, iid.page_no, &ph);
        if (rc) { return rc; }
        assert(ph.hdr->is_leaf);
        rid_t *curr_rid = &ph.p_rid[iid.slot_no];
        if (curr_rid->page_no == rid->page_no && curr_rid->slot_no == rid->slot_no) {
            // Found the entry with the given rid, delete it
            pf_mark_dirty(ph.page);
            ix_page_erase_key(fh, &ph, iid.slot_no);
            ix_page_erase_rid(&ph, iid.slot_no);
            // Update its parent's key to the node's new last key
            rc = maintain_parent(fh, &ph);
            if (rc) { return rc; }
            // Solve underflow
            while (ph.hdr->num_child < (fh->hdr.btree_order + 1) / 2) {
                if (ph.hdr->parent == IX_NO_PAGE) {
                    // If current node is root node, underflow is permitted
                    if (!ph.hdr->is_leaf && ph.hdr->num_key <= 1) {
                        // If root node is not leaf and it is empty, delete the root
                        int new_root_page = ph.p_rid[0].page_no;
                        // Load new root and set its parent to NO_PAGE
                        ix_page_handle_t new_root;
                        rc = ix_fetch_page(fh, new_root_page, &new_root);
                        if (rc) { return rc; }
                        new_root.hdr->parent = IX_NO_PAGE;
                        // Update global root
                        fh->hdr.root_page = new_root_page;
                        fh->hdr_dirty = true;
                        // Free current page
                        ix_release_page(fh, &ph);
                    }
                    break;
                }
                // Load parent node
                ix_page_handle_t parent;
                rc = ix_fetch_page(fh, ph.hdr->parent, &parent);
                if (rc) { return rc; }
                pf_mark_dirty(parent.page);
                // Find the rank of this child in its parent
                int child_idx = ix_get_child_idx(&parent, &ph);
                if (0 < child_idx) {
                    // current node has left brother, load it
                    ix_page_handle_t bro;
                    rc = ix_fetch_page(fh, parent.p_rid[child_idx - 1].page_no, &bro);
                    if (rc) { return rc; }
                    if (bro.hdr->num_child > (fh->hdr.btree_order + 1) / 2) {
                        // If left brother is rich, borrow one node from it
                        pf_mark_dirty(bro.page);
                        ix_page_insert_rid(&ph, 0, &bro.p_rid[bro.hdr->num_key - 1]);
                        ix_page_insert_key(fh, &ph, 0, ix_get_key(fh, &bro, bro.hdr->num_key - 1));
                        ix_page_erase_rid(&bro, bro.hdr->num_key - 1);
                        ix_page_erase_key(fh, &bro, bro.hdr->num_key - 1);
                        // Maintain parent's key as the node's max key
                        rc = maintain_parent(fh, &bro);
                        if (rc) { return rc; }
                        // Maintain first child's parent
                        rc = maintain_child(fh, &ph, 0);
                        if (rc) { return rc; }
                        // underflow is solved
                        break;
                    }
                }
                if (child_idx + 1 < parent.hdr->num_child) {
                    // current node has right brother, load it
                    ix_page_handle_t bro;
                    rc = ix_fetch_page(fh, parent.p_rid[child_idx + 1].page_no, &bro);
                    if (rc) { return rc; }
                    if (bro.hdr->num_child > (fh->hdr.btree_order + 1) / 2) {
                        // If right brother is rich, borrow one node from it
                        pf_mark_dirty(bro.page);
                        ix_page_insert_rid(&ph, ph.hdr->num_key, &bro.p_rid[0]);
                        ix_page_insert_key(fh, &ph, ph.hdr->num_key, bro.p_key);
                        ix_page_erase_rid(&bro, 0);
                        ix_page_erase_key(fh, &bro, 0);
                        // Maintain parent's key as the node's max key
                        rc = maintain_parent(fh, &ph);
                        if (rc) { return rc; }
                        // Maintain last child's parent
                        rc = maintain_child(fh, &ph, ph.hdr->num_child - 1);
                        if (rc) { return rc; }
                        // Underflow is solved
                        break;
                    }
                }
                // neither brothers is rich, need to merge
                if (0 < child_idx) {
                    // merge with left brother, transfer all children of current node to left brother
                    ix_page_handle_t bro;
                    rc = ix_fetch_page(fh, parent.p_rid[child_idx - 1].page_no, &bro);
                    if (rc) { return rc; }
                    pf_mark_dirty(bro.page);
                    ix_page_insert_rids(&bro, bro.hdr->num_key, ph.p_rid, ph.hdr->num_child);
                    ix_page_insert_keys(fh, &bro, bro.hdr->num_key, ph.p_key, ph.hdr->num_key);
                    // Maintain left brother's children
                    for (int i = bro.hdr->num_key - ph.hdr->num_child; i < bro.hdr->num_key; i++) {
                        rc = maintain_child(fh, &bro, i);
                        if (rc) { return rc; }
                    }
                    ix_page_erase_rid(&parent, child_idx);
                    ix_page_erase_key(fh, &parent, child_idx);
                    rc = maintain_parent(fh, &bro);
                    if (rc) { return rc; }

                    // Maintain leaf list
                    if (ph.hdr->is_leaf) {
                        rc = ix_erase_leaf(fh, &ph);
                        if (rc) { return rc; }
                    }
                    // Update global last-leaf
                    if (fh->hdr.last_leaf == ph.page->page_id.page_no) {
                        fh->hdr.last_leaf = bro.page->page_id.page_no;
                        fh->hdr_dirty = true;
                    }
                    // Free current page
                    ix_release_page(fh, &ph);
                } else {
                    assert(child_idx + 1 < parent.hdr->num_child);
                    // merge with right brother, transfer all children of right brother to current node
                    ix_page_handle_t bro;
                    rc = ix_fetch_page(fh, parent.p_rid[child_idx + 1].page_no, &bro);
                    if (rc) { return rc; }
                    pf_mark_dirty(bro.page);
                    // Transfer all right brother's valid rid to current node
                    ix_page_insert_rids(&ph, ph.hdr->num_key, bro.p_rid, bro.hdr->num_key);
                    ix_page_insert_keys(fh, &ph, ph.hdr->num_key, bro.p_key, bro.hdr->num_key);
                    // Maintain current node's children
                    for (int i = ph.hdr->num_key - bro.hdr->num_key; i < ph.hdr->num_key; i++) {
                        rc = maintain_child(fh, &ph, i);
                        if (rc) { return rc; }
                    }
                    ix_page_erase_rid(&parent, child_idx + 1);
                    ix_page_erase_key(fh, &parent, child_idx);
                    // Maintain parent's key as the node's max key
                    rc = maintain_parent(fh, &ph);
                    if (rc) { return rc; }
                    // Maintain leaf list
                    if (bro.hdr->is_leaf) {
                        rc = ix_erase_leaf(fh, &bro);
                        if (rc) { return rc; }
                    }
                    // Update global last leaf
                    if (fh->hdr.last_leaf == bro.page->page_id.page_no) {
                        fh->hdr.last_leaf = ph.page->page_id.page_no;
                        fh->hdr_dirty = true;
                    }
                    // Free right brother page
                    ix_release_page(fh, &bro);
                }
                ph = parent;
            }
            return 0;
        }
        ix_scan_next(fh, &iid);
    }
    return IX_ENTRY_NOT_FOUND;
}

bool ix_scan_equal(const iid_t *x, const iid_t *y) {
    return x->page_no == y->page_no && x->slot_no == y->slot_no;
}

void ix_leaf_begin(const ix_file_handle_t *fh, iid_t *iid) {
    iid->page_no = fh->hdr.first_leaf;
    iid->slot_no = 0;
}

RC ix_leaf_end(const ix_file_handle_t *fh, iid_t *iid) {
    RC rc;
    ix_page_handle_t last;
    rc = ix_fetch_page(fh, fh->hdr.last_leaf, &last);
    if (rc) { return rc; }
    iid->page_no = fh->hdr.last_leaf;
    iid->slot_no = last.hdr->num_key;
    return 0;
}

RC ix_scan_next(const ix_file_handle_t *fh, iid_t *iid) {
    RC rc;
    ix_page_handle_t ph;
    rc = ix_fetch_page(fh, iid->page_no, &ph);
    if (rc) { return rc; }
    assert(ph.hdr->is_leaf);
    // increment slot no
    assert(iid->slot_no < ph.hdr->num_key);
    iid->slot_no++;
    if (iid->page_no != fh->hdr.last_leaf && iid->slot_no == ph.hdr->num_key) {
        // go to next leaf
        iid->slot_no = 0;
        iid->page_no = ph.hdr->next_leaf;
    }
    return 0;
}
