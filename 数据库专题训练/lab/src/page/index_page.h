#ifndef THDB_INDEX_PAGE_H_
#define THDB_INDEX_PAGE_H_

#include <cassert>

#include "macros.h"
#include "page/page.h"

namespace thdb {

class IndexPage : public Page {
 public:
  // header offsets
  const PageOffset ROOT_PAGE_OFFSET = 8;
  const PageOffset KEY_TYPE_OFFSET = 12;
  const PageOffset KEY_LEN_OFFSET = 16;
  const PageOffset CAPACITY_OFFSET = 20;

  PageID rootPage;
  FieldType keyType;
  Size keyLen;
  Size capacity;

  IndexPage(PageID rootPage_, FieldType keyType_, Size keyLen_) : Page() {
    rootPage = rootPage_;
    keyType = keyType_;
    keyLen = keyLen_;
    capacity = DATA_SIZE / (keyLen_ + sizeof(PageSlotID)) - 1;
  }

  IndexPage(PageID nPageID) : Page(nPageID) { Load(); }

  ~IndexPage() override { Store(); }

 private:
  void Load() {
    // load headers
    GetHeader((uint8_t*)&rootPage, sizeof(PageID), ROOT_PAGE_OFFSET);
    GetHeader((uint8_t*)&keyType, sizeof(int), KEY_TYPE_OFFSET);
    GetHeader((uint8_t*)&keyLen, sizeof(Size), KEY_LEN_OFFSET);
    GetHeader((uint8_t*)&capacity, sizeof(Size), CAPACITY_OFFSET);
  }

  void Store() {
    // store headers
    SetHeader((uint8_t*)&rootPage, sizeof(PageID), ROOT_PAGE_OFFSET);
    SetHeader((uint8_t*)&keyType, sizeof(int), KEY_TYPE_OFFSET);
    SetHeader((uint8_t*)&keyLen, sizeof(Size), KEY_LEN_OFFSET);
    SetHeader((uint8_t*)&capacity, sizeof(Size), CAPACITY_OFFSET);
  }
};

}  // namespace thdb

#endif