#include "page/page.h"

#include "macros.h"
#include "minios/os.h"

namespace thdb {

const PageOffset DATA_BEGIN_OFFSET = HEADER_SIZE;

Page::Page() {
  this->_bModified = true;
  this->_nPageID = MiniOS::GetOS()->NewPage();
}

Page::Page(PageID nPageID) {
  this->_bModified = false;
  this->_nPageID = nPageID;
}

Page::~Page() {}

uint32_t Page::GetPageID() const { return _nPageID; }

void Page::SetPageID(PageID nPageID) {
  this->_nPageID = nPageID;
  this->_bModified = true;
}

void Page::GetHeader(uint8_t *dst, PageOffset nSize, PageOffset nOffset) const {
  MiniOS::GetOS()->ReadPage(_nPageID, dst, nSize, nOffset);
}

void Page::SetHeader(const uint8_t *src, PageOffset nSize, PageOffset nOffset) {
  MiniOS::GetOS()->WritePage(_nPageID, src, nSize, nOffset);
  this->_bModified = true;
}

void Page::GetData(uint8_t *dst, PageOffset nSize, PageOffset nOffset) const {
  MiniOS::GetOS()->ReadPage(_nPageID, dst, nSize, nOffset + DATA_BEGIN_OFFSET);
}

void Page::SetData(const uint8_t *src, PageOffset nSize, PageOffset nOffset) {
  MiniOS::GetOS()->WritePage(_nPageID, src, nSize, nOffset + DATA_BEGIN_OFFSET);
  this->_bModified = true;
}

}  // namespace thdb
