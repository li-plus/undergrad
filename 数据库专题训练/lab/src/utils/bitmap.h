#ifndef THDB_BITMAP_H_
#define THDB_BITMAP_H_

#include <cstdint>
#include <cstdlib>

#include "defines.h"

namespace thdb {

class Bitmap {
 public:
  Bitmap(Size size);
  ~Bitmap();

  void Set(Size pos);
  void Unset(Size pos);
  bool Get(Size pos) const;
  Size GetSize() const;
  Size GetUsed() const;
  bool Full() const;
  bool Empty() const;
  void Load(const uint8_t *pBits);
  void Store(uint8_t *pBits);

 private:
  uint8_t *_pBits;
  Size _nSize;
  Size _nUsed;
};

}  // namespace thdb

#endif  // MINIOS_BITMAP_H_
