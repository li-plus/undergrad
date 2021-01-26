#include "ix/ix.h"
#include <cassert>
#include <ctime>
#include <map>
#include <queue>
#include <iostream>

void check_tree(const ix_file_handle_t *fh, int root_page) {
    ix_page_handle_t ph;
    assert(ix_fetch_page(fh, root_page, &ph) == 0);
    if (ph.hdr->is_leaf) {
        return;
    }
    for (int i = 0; i < ph.hdr->num_child; i++) {
        ix_page_handle_t child;
        assert(ix_fetch_page(fh, ph.p_rid[i].page_no, &child) == 0);
        // check parent
        assert(child.hdr->parent == root_page);
        // check last key
        assert(memcmp(ph.p_key + fh->hdr.attr_len * i,
                      child.p_key + fh->hdr.attr_len * (child.hdr->num_key - 1),
                      fh->hdr.attr_len) == 0);
        check_tree(fh, ph.p_rid[i].page_no);
    }
    // check leaf list
    int leaf_no = fh->hdr.first_leaf;
    while (leaf_no != IX_LEAF_HEADER_PAGE) {
        ix_page_handle_t curr;
        assert(ix_fetch_page(fh, leaf_no, &curr) == 0);
        ix_page_handle_t prev;
        assert(ix_fetch_page(fh, curr.hdr->prev_leaf, &prev) == 0);
        ix_page_handle_t next;
        assert(ix_fetch_page(fh, curr.hdr->next_leaf, &next) == 0);
        // Ensure prev->next == curr && next->prev == curr
        assert(prev.hdr->next_leaf == leaf_no);
        assert(next.hdr->prev_leaf == leaf_no);
        leaf_no = curr.hdr->next_leaf;
    }
}

void check_equal(const ix_file_handle_t *fh, const std::multimap<int, rid_t> &mock) {
    check_tree(fh, fh->hdr.root_page);
    for (auto &entry: mock) {
        iid_t iid;
        // test lower bound
        uint8_t key_buf[sizeof(int) + 1];
        key_buf[0] = 1;
        *(int *) (key_buf + 1) = entry.first;
        assert(ix_lower_bound(fh, key_buf, &iid) == 0);
        int key;
        rid_t rid;
        assert(ix_get_entry(fh, &iid, key_buf, &rid) == 0);
        key = *(int *) (key_buf + 1);
        assert(key == entry.first);
        // test upper bound
        auto mock_upper = mock.upper_bound(key);
        *(int *) (key_buf + 1) = entry.first;
        ix_upper_bound(fh, key_buf, &iid);
        if (mock_upper == mock.end()) {
            iid_t end;
            assert(ix_leaf_end(fh, &end) == 0);
            assert(ix_scan_equal(&iid, &end));
        } else {
            assert(ix_get_entry(fh, &iid, key_buf, &rid) == 0);
            key = *(int *) (key_buf + 1);
            assert(key == mock_upper->first);
            assert(memcmp(&rid, &mock_upper->second, sizeof(rid_t)) == 0);
        }
    }
    // test btree iterator
    iid_t iid, end;
    ix_leaf_begin(fh, &iid);
    assert(ix_leaf_end(fh, &end) == 0);
    auto it = mock.begin();
    while (!ix_scan_equal(&iid, &end) && it != mock.end()) {
        int key;
        rid_t rid;
        uint8_t key_buf[sizeof(int) + 1];
        assert(ix_get_entry(fh, &iid, key_buf, &rid) == 0);
        key = *(int *) (key_buf + 1);
        assert(key == it->first);
        assert(memcmp(&rid, &it->second, sizeof(rid_t)) == 0);
        assert(ix_scan_next(fh, &iid) == 0);
        it++;
    }
    assert(ix_scan_equal(&iid, &end) && it == mock.end());
}

void print_btree(const ix_file_handle_t *fh, int root_page, int offset) {
    ix_page_handle_t ph;
    assert(ix_fetch_page(fh, root_page, &ph) == 0);
    for (int i = ph.hdr->num_child - 1; i > -1; i--) {
        // print key
        std::cout << std::string(offset, ' ') << ((int *) ph.p_key)[i] << std::endl;
        // print child
        if (!ph.hdr->is_leaf) {
            print_btree(fh, ph.p_rid[i].page_no, offset + 4);
        }
    }
}

void test_ix_insert_delete(int order, int round) {
    char filename[] = "abc";
    int index_no = 0;
    if (ix_exists(filename, index_no)) {
        assert(ix_destroy_index(filename, index_no) == 0);
    }
    assert(ix_create_index(filename, index_no, ATTR_INT, sizeof(int)) == 0);
    ix_file_handle_t fh;
    assert(ix_open_index(filename, index_no, &fh) == 0);
    if (order > 2 && order <= fh.hdr.btree_order) {
        fh.hdr.btree_order = order;
    }
    std::multimap<int, rid_t> mock;
    for (int i = 0; i < round; i++) {
        int rand_key = rand() % round;
        rid_t rand_val = {.page_no = rand(), .slot_no = rand()};
        assert(ix_insert_entry(&fh, (input_buffer_t) &rand_key, &rand_val) == 0);
        mock.insert(std::make_pair(rand_key, rand_val));
        if (round % 500 == 0) {
            assert(ix_close_index(&fh) == 0);
            assert(ix_open_index(filename, index_no, &fh) == 0);
        }
    }
    print_btree(&fh, fh.hdr.root_page, 0);
    std::cout << std::endl;
    check_equal(&fh, mock);
    for (int i = 0; i < round; i++) {
        auto it = mock.begin();
        int key = it->first;
        rid_t rid = it->second;
        assert(ix_delete_entry(&fh, (input_buffer_t) &key, &rid) == 0);
        mock.erase(it);
        if (round % 500 == 0) {
            assert(ix_close_index(&fh) == 0);
            assert(ix_open_index(filename, index_no, &fh) == 0);
        }
    }
    check_equal(&fh, mock);
    assert(pf_clear() == 0);
    assert(ix_close_index(&fh) == 0);
    assert(ix_destroy_index(filename, index_no) == 0);
}

void test_ix(int order, int round) {
    char filename[] = "abc";
    int index_no = 0;
    if (ix_exists(filename, index_no)) {
        assert(ix_destroy_index(filename, index_no) == 0);
    }
    assert(ix_create_index(filename, index_no, ATTR_INT, 1 + sizeof(int)) == 0);
    ix_file_handle_t fh;
    assert(ix_open_index(filename, index_no, &fh) == 0);
    if (order >= 2 && order <= fh.hdr.btree_order) {
        fh.hdr.btree_order = order;
    }
    int add_cnt = 0;
    int del_cnt = 0;
    std::multimap<int, rid_t> mock;
    for (int i = 0; i < round; i++) {
        double dice = rand() * 1. / RAND_MAX;
        double insert_prob = 1. - mock.size() / (0.5 * round);
        if (mock.empty() || dice < insert_prob) {
            // Insert
            int rand_key = rand() % round;
            uint8_t rand_key_buf[sizeof(int) + 1];
            rand_key_buf[0] = 1;
            *(int *) (rand_key_buf + 1) = rand_key;
            rid_t rand_val = {.page_no = rand(), .slot_no = rand()};
            assert(ix_insert_entry(&fh, rand_key_buf, &rand_val) == 0);
            mock.insert(std::make_pair(rand_key, rand_val));
            add_cnt++;
        } else {
            // Delete
            int rand_idx = rand() % mock.size();
            auto it = mock.begin();
            for (int k = 0; k < rand_idx; k++) { it++; }
            int key = it->first;
            rid_t rid = it->second;
            uint8_t key_buf[sizeof(int) + 1];
            key_buf[0] = 1;
            *(int *) (key_buf + 1) = key;
            assert(ix_delete_entry(&fh, key_buf, &rid) == 0);
            mock.erase(it);
            del_cnt++;
        }
        // Randomly re-open file
        if (round % 500 == 0) {
            assert(ix_close_index(&fh) == 0);
            assert(ix_open_index(filename, index_no, &fh) == 0);
        }
    }
    print_btree(&fh, fh.hdr.root_page, 0);
    std::cout << std::endl;
    check_equal(&fh, mock);
    std::cout << "Insert " << add_cnt << '\n'
              << "Delete " << del_cnt << '\n';
    while (!mock.empty()) {
        int rand_idx = rand() % mock.size();
        auto it = mock.begin();
        for (int k = 0; k < rand_idx; k++) { it++; }
        uint8_t key_buf[sizeof(int) + 1];
        key_buf[0] = 1;
        *(int *) (key_buf + 1) = it->first;
        rid_t rid = it->second;
        assert(ix_delete_entry(&fh, key_buf, &rid) == 0);
        mock.erase(it);
        // Randomly re-open file
        if (round % 500 == 0) {
            assert(ix_close_index(&fh) == 0);
            assert(ix_open_index(filename, index_no, &fh) == 0);
        }
    }
    check_equal(&fh, mock);
    assert(pf_clear() == 0);
    assert(ix_close_index(&fh) == 0);
    assert(ix_destroy_index(filename, index_no) == 0);
}

int main() {
    srand((unsigned) time(nullptr));
    // init
    pf_init();
    test_ix(3, 1000);
    test_ix(4, 1000);
    test_ix(-1, 100000);
    return 0;
}