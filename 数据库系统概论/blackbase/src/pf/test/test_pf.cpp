#include "pf/pf.h"
#include <cassert>
#include <cstring>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <string>

#define MAX_FILES 32
#define MAX_PAGES 128

std::unordered_map<int, buffer_t> mock;     // fd -> buffer

inline buffer_t mock_get_page(int fd, int page_no) {
    return &mock[fd][page_no * PAGE_SIZE];
}

void check_disk(int fd, int page_no) {
    static uint8_t buffer[PAGE_SIZE];
    assert(pf_read_page(fd, page_no, buffer, PAGE_SIZE) == 0);
    buffer_t mock_buf = mock_get_page(fd, page_no);
    assert(memcmp(buffer, mock_buf, PAGE_SIZE) == 0);
}

void check_disk_all() {
    for (auto &file: mock) {
        int fd = file.first;
        for (int page_no = 0; page_no < MAX_PAGES; page_no++) {
            check_disk(fd, page_no);
        }
    }
}

void check_cache(int fd, int page_no) {
    pf_page_ptr_t page;
    assert(pf_fetch_page(fd, page_no, &page) == 0);
    buffer_t mock_buf = mock_get_page(fd, page_no);
    assert(memcmp(page->buffer, mock_buf, PAGE_SIZE) == 0);
}

void check_cache_all() {
    for (auto &file: mock) {
        int fd = file.first;
        for (int page_no = 0; page_no < MAX_PAGES; page_no++) {
            check_cache(fd, page_no);
        }
    }
}

void rand_buffer(int size, output_buffer_t buffer) {
    for (int i = 0; i < size; i++) {
        int rand_ch = rand() & 0xff;
        buffer[i] = rand_ch;
    }
}

int rand_fd() {
    assert(mock.size() == MAX_FILES);
    int fd_idx = rand() % MAX_FILES;
    auto it = mock.begin();
    for (int i = 0; i < fd_idx; i++) {
        it++;
    }
    return it->first;
}

int main() {
    srand((unsigned) time(nullptr));

    // test files
    std::vector<std::string> filenames(MAX_FILES);
    std::unordered_map<int, std::string> fd2name;
    for (int i = 0; i < filenames.size(); i++) {
        auto &filename = filenames[i];
        filename = std::to_string(i) + ".txt";
        if (pf_exists(filename.c_str())) {
            assert(pf_destroy_file(filename.c_str()) == 0);
        }
        int fd;
        // open without create
        assert(pf_open_file(filename.c_str(), &fd) == PF_FILE_NOT_FOUND);
        // create file
        assert(pf_create_file(filename.c_str()) == 0);
        assert(pf_exists(filename.c_str()));
        assert(pf_create_file(filename.c_str()) == PF_FILE_EXISTS);
        // open file
        assert(pf_open_file(filename.c_str(), &fd) == 0);
        mock[fd] = new uint8_t[PAGE_SIZE * MAX_PAGES];
        fd2name[fd] = filename;
    }
    // test cache
    pf_init();
    // init & test alloc_page
    uint8_t init_buffer[PAGE_SIZE];
    pf_page_ptr_t page;
    buffer_t mock_buf;
    for (auto &fh: mock) {
        int fd = fh.first;
        for (int page_no = 0; page_no < MAX_PAGES; page_no++) {
            rand_buffer(PAGE_SIZE, init_buffer);
            assert(pf_create_page(fd, page_no, &page) == 0);
            memcpy(page->buffer, init_buffer, PAGE_SIZE);
            mock_buf = mock_get_page(fd, page_no);
            memcpy(mock_buf, init_buffer, PAGE_SIZE);
        }
    }
    check_cache_all();
    assert(pf_clear() == 0);
    check_disk_all();
    // test get_page
    for (int r = 0; r < 10000; r++) {
        int fd = rand_fd();
        int page_no = rand() % MAX_PAGES;
        // get page
        assert(pf_fetch_page(fd, page_no, &page) == 0);
        mock_buf = mock_get_page(fd, page_no);
        assert(memcmp(page->buffer, mock_buf, PAGE_SIZE) == 0);

        // modify
        rand_buffer(PAGE_SIZE, init_buffer);
        memcpy(page->buffer, init_buffer, PAGE_SIZE);
        memcpy(mock_buf, init_buffer, PAGE_SIZE);
        pf_mark_dirty(page);

        // flush
        if (rand() % 10 == 0) {
            assert(pf_flush_page(page) == 0);
            check_disk(fd, page_no);
        }
        // flush entire file
        if (rand() % 100 == 0) {
            assert(pf_flush_file(fd) == 0);
        }
        // re-open file
        if (rand() % 100 == 0) {
            assert(pf_close_file(fd) == 0);
            auto filename = fd2name[fd];
            buffer_t buf = mock[fd];
            fd2name.erase(fd);
            mock.erase(fd);
            assert(pf_open_file(filename.c_str(), &fd) == 0);
            mock[fd] = buf;
            fd2name[fd] = filename;
        }
        // assert equal in cache
        check_cache(fd, page_no);
    }
    check_cache_all();
    assert(pf_clear() == 0);
    check_disk_all();
    // close and destroy files
    for (auto &entry: fd2name) {
        int fd = entry.first;
        auto &filename = entry.second;
        assert(pf_close_file(fd) == 0);
        assert(pf_destroy_file(filename.c_str()) == 0);
        assert(pf_destroy_file(filename.c_str()) == PF_FILE_NOT_FOUND);
    }
    return 0;
}