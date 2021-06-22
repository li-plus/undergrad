#ifndef THDB_OS_H_
#define THDB_OS_H_

#include "defines.h"
#include "minios/raw_page.h"
#include "utils/bitmap.h"

namespace thdb {

class Page;
class Bitmap;

class MiniOS {
 public:
  static MiniOS *GetOS();
  static void WriteBack();

  PageID NewPage();
  void DeletePage(PageID pid);
  void ReadPage(PageID pid, uint8_t *dst, PageOffset nSize,
                PageOffset nOffset = 0);
  void WritePage(PageID pid, const uint8_t *src, PageOffset nSize,
                 PageOffset nOffset = 0);
  Size GetUsedSize() const;
  void Reload() { LoadPages(); }

 private:
  MiniOS();
  ~MiniOS();

  void LoadBitmap();
  void LoadPages();

  void StoreBitmap();
  void StorePages();

  RawPage **_pMemory;
  Bitmap *_pUsed;
  Size _nClock;

  static MiniOS *os;
};

}  // namespace thdb

#endif  // MINIOS_OS_H_
