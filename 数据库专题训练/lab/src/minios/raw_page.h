#ifndef MINIOS_PAGE_H_
#define MINIOS_PAGE_H_

#include "defines.h"

namespace thdb {

class RawPage {
 public:
  RawPage();
  ~RawPage();

  void Read(uint8_t* dst, PageOffset nSize, PageOffset nOffset = 0);
  void Write(const uint8_t* src, PageOffset nSize, PageOffset nOffset = 0);

 private:
  uint8_t* _pData;
};

}  // namespace thdb

#endif  // MINIOS_PAGE_H_
