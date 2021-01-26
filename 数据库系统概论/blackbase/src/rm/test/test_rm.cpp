#include "rm/rm.h"
#include <cassert>
#include <ctime>
#include <cstring>
#include <unordered_map>
#include <iostream>

void rand_buf(int size, buffer_t out_buf) {
    for (int i = 0; i < size; i++) {
        out_buf[i] = rand() & 0xff;
    }
}

struct rid_hash_t {
    size_t operator()(const rid_t &rid) const {
        return (rid.page_no << 16) | rid.slot_no;
    }
};

struct rid_equal_t {
    bool operator()(const rid_t &x, const rid_t &y) const {
        return x.page_no == y.page_no && x.slot_no == y.slot_no;
    }
};

void check_equal(rm_file_handle_t *fh,
                 const std::unordered_map<rid_t, std::string, rid_hash_t, rid_equal_t> &mock) {
    static uint8_t read_buffer[PAGE_SIZE];
    for (auto &entry: mock) {
        rid_t rid = entry.first;
        auto mock_buf = (uint8_t *) entry.second.c_str();
        assert(rm_get_record(fh, &rid, read_buffer) == 0);
        assert(memcmp(mock_buf, read_buffer, fh->hdr.record_size) == 0);
    }
    for (int i = 0; i < 10; i++) {
        rid_t rid = {
                .page_no = 1 + rand() % (fh->hdr.num_pages - 1),
                .slot_no = rand() % fh->hdr.num_records_per_page
        };
        bool mock_exist = (mock.count(rid) > 0);
        bool rm_exist = (rm_get_record(fh, &rid, read_buffer) == 0);
        assert(rm_exist == mock_exist);
    }
    rm_record_t rec;
    rm_record_init(&rec, fh);
    assert(rm_scan_init(fh, &rec.rid) == 0);
    int num_records = 0;
    while (!rm_scan_is_end(&rec.rid)) {
        assert(mock.count(rec.rid) > 0);
        assert(rm_get_record(fh, &rec.rid, rec.data) == 0);
        assert(memcmp(rec.data, mock.at(rec.rid).c_str(), fh->hdr.record_size) == 0);
        assert(rm_scan_next(fh, &rec.rid) == 0);
        num_records++;
    }
    assert(num_records == mock.size());
    rm_record_destroy(&rec);
}

std::ostream &operator<<(std::ostream &os, const rid_t &rid) {
    return os << '(' << rid.page_no << ", " << rid.slot_no << ')';
}

int main() {
    srand((unsigned) time(nullptr));

    std::unordered_map<rid_t, std::string, rid_hash_t, rid_equal_t> mock;

    // init
    pf_init();
    char filename[] = "abc.txt";
    rm_file_handle_t fh;

    // test files
    int record_size = 4 + rand() % 256;
    if (pf_exists(filename)) {
        assert(rm_destroy_file(filename) == 0);
    }
    assert(rm_create_file(filename, record_size) == 0);

    assert(rm_open_file(filename, &fh) == 0);
    assert(fh.hdr.record_size == record_size);
    assert(fh.hdr.first_free_page == RM_NO_PAGE);
    assert(fh.hdr.num_pages == 1);
    int max_bytes =
            fh.hdr.record_size * fh.hdr.num_records_per_page + fh.hdr.bitmap_size + (int) sizeof(rm_page_hdr_t);
    assert(max_bytes <= PAGE_SIZE);
    int rand_val = rand();
    fh.hdr.num_pages = rand_val;
    fh.hdr_dirty = true;
    assert(rm_close_file(&fh) == 0);
    // reopen file
    assert(rm_open_file(filename, &fh) == 0);
    assert(fh.hdr.num_pages == rand_val);
    assert(rm_close_file(&fh) == 0);
    assert(rm_destroy_file(filename) == 0);

    // test pages
    assert(rm_create_file(filename, record_size) == 0);
    assert(rm_open_file(filename, &fh) == 0);
    uint8_t write_buffer[PAGE_SIZE];
    int add_cnt = 0;
    int upd_cnt = 0;
    int del_cnt = 0;
    for (int round = 0; round < 10000; round++) {
        double insert_prob = 1. - mock.size() / 2500.;
        double dice = rand() * 1. / RAND_MAX;
        if (mock.empty() || dice < insert_prob) {
            rid_t rid;
            rand_buf(fh.hdr.record_size, write_buffer);
            assert(rm_insert_record(&fh, write_buffer, &rid) == 0);
            mock[rid] = std::string((char *) write_buffer, fh.hdr.record_size);
            add_cnt++;
//            std::cout << "insert " << rid << '\n';
        } else {
            // update or erase random rid
            int rid_idx = rand() % mock.size();
            auto it = mock.begin();
            for (int i = 0; i < rid_idx; i++) {
                it++;
            }
            auto rid = it->first;
            if (rand() % 2 == 0) {
                // update
                rand_buf(fh.hdr.record_size, write_buffer);
                assert(rm_update_record(&fh, &rid, write_buffer) == 0);
                mock[rid] = std::string((char *) write_buffer, fh.hdr.record_size);
                upd_cnt++;
//                std::cout << "update " << rid << '\n';
            } else {
                // erase
                assert(rm_delete_record(&fh, &rid) == 0);
                mock.erase(rid);
                del_cnt++;
//                std::cout << "delete " << rid << '\n';
            }
        }
        // Randomly re-open file
        if (round % 500 == 0) {
            assert(rm_close_file(&fh) == 0);
            assert(rm_open_file(filename, &fh) == 0);
        }
        check_equal(&fh, mock);
    }
    std::cout << "insert " << add_cnt << '\n'
              << "delete " << del_cnt << '\n'
              << "update " << upd_cnt << '\n';
    // clean up
    assert(pf_clear() == 0);
    assert(rm_close_file(&fh) == 0);
    assert(rm_destroy_file(filename) == 0);
    return 0;
}