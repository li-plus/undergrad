#include "pf/pf_hash_table.h"
#include <ctime>
#include <cassert>
#include <unordered_map>
#include <iostream>

struct page_id_hash_t {
    size_t operator()(const pf_page_id_t &page_id) const {
        return (page_id.fd << 16) | page_id.page_no;
    }
};

struct page_id_equal_t {
    bool operator()(const pf_page_id_t &x, const pf_page_id_t &y) const {
        return x.fd == y.fd && x.page_no == y.page_no;
    }
};

std::unordered_map<pf_page_id_t, pf_page_ptr_t, page_id_hash_t, page_id_equal_t> mock;

pf_hash_table_t ht;

int list_size(list_entry_t *header) {
    int size = 0;
    for (list_entry_t *le = list_begin(header); le != list_end(header); le = list_next(le)) {
        size++;
    }
    return size;
}

void check_equal() {
    // test find
    for (auto &entry: mock) {
        assert(pf_hash_table_get(&ht, &entry.first) == entry.second);
    }
    for (int i = 0; i < 10; i++) {
        pf_page_id_t key = {.fd = rand(), .page_no = rand()};
        assert((pf_hash_table_get(&ht, &key) != nullptr) == (mock.count(key) != 0));
    }
    assert(list_size(&ht.free_hashes) == PF_MAX_CACHE_PAGES - mock.size());
}

int main() {
    srand((unsigned) time(nullptr));

    int add_cnt = 0;
    int del_cnt = 0;

    pf_hash_table_init(&ht);
    for (int i = 0; i < 10000; i++) {
        double insert_prob = 1 - mock.size() / (0.5 * PF_MAX_CACHE_PAGES);
        double dice = rand() * 1. / RAND_MAX;
        if (mock.empty() || (dice < insert_prob && mock.size() < PF_MAX_CACHE_PAGES)) {
            // insert
            pf_page_id_t key = {.fd = rand(), .page_no = rand()};
            auto value = (pf_page_ptr_t) (long long) rand();
            if (pf_hash_table_get(&ht, &key) == nullptr) {
                pf_hash_entry_t *entry = pf_hash_table_insert(&ht, &key, value);
                assert(entry->page == value);
            }
            if (mock.count(key) == 0) {
                mock[key] = value;
            }
            add_cnt++;
        } else {
            // erase
            auto key = mock.begin()->first;
            pf_hash_table_erase(&ht, &key);
            mock.erase(key);
            del_cnt++;
        }
        check_equal();
    }
    // clean up
    pf_hash_table_destroy(&ht);
    mock.clear();
    check_equal();
    for (auto &bucket : ht.buckets) {
        assert(list_empty(&bucket));
    }
    std::cout << "insert " << add_cnt << '\n'
              << "delete " << del_cnt << '\n';
    return 0;
}
