#include "pf/list.h"
#include <cassert>
#include <ctime>
#include <vector>
#include <iostream>

typedef struct {
    int val;
    list_entry_t le;
} list_node_t;

#define le2node(ptr) to_struct((ptr), list_node_t, le)

list_entry_t *list_get_pos(list_entry_t *header, int idx) {
    list_entry_t *le = list_begin(header);
    for (int i = 0; i < idx; i++) {
        le = list_next(le);
    }
    return le;
}

void check_equal(list_entry_t *header, const std::vector<int> &mock) {
    auto le = list_begin(header);
    auto it = mock.begin();
    while (le != list_end(header) && it != mock.end()) {
        list_node_t *list_node = le2node(le);
        assert(list_node->val == *it);
        le = list_next(le);
        it++;
    }
    assert(le == list_end(header) && it == mock.end());
}

void rand_insert(list_entry_t *header, std::vector<int> &mock) {
    int idx = mock.empty() ? 0 : (rand() % mock.size());
    int val = rand();
    list_entry_t *pos = list_get_pos(header, idx);
    auto node = new list_node_t;
    node->val = val;
    list_insert(pos, &node->le);
    mock.insert(mock.begin() + idx, val);
}

void rand_erase(list_entry_t *header, std::vector<int> &mock) {
    int idx = rand() % mock.size();
    auto le = list_get_pos(header, idx);
    list_node_t *node = le2node(le);
    list_erase(le);
    delete node;
    mock.erase(mock.begin() + idx);
}

int main() {
    srand((unsigned) time(nullptr));

    list_entry_t header;
    list_init(&header);

    std::vector<int> mock;

    // init
    for (int i = 0; i < 100; i++) {
        rand_insert(&header, mock);
    }
    // random operation
    int add_cnt = 0;
    int del_cnt = 0;
    for (int i = 0; i < 10000; i++) {
        double insert_prob = 1 - mock.size() / 5000.;
        double dice = rand() * 1. / RAND_MAX;
        if (mock.empty() || dice < insert_prob) {
            rand_insert(&header, mock);
            add_cnt++;
        } else {
            rand_erase(&header, mock);
            del_cnt++;
        }
        check_equal(&header, mock);
    }
    std::cout << "insert " << add_cnt << '\n'
              << "delete " << del_cnt << '\n';
    // clean up
    while (!mock.empty()) {
        rand_erase(&header, mock);
        check_equal(&header, mock);
    }
    assert(list_empty(&header));
    return 0;
}
