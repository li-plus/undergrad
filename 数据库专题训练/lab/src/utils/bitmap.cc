#include "utils/bitmap.h"

#include <cstring>

namespace thdb {

Bitmap::Bitmap(Size size) {
  _pBits = new uint8_t[(size - 1) / 8 + 1];
  memset(_pBits, 0, (size - 1) / 8 + 1);
  _nSize = size;
  _nUsed = 0;
}

Bitmap::~Bitmap() { delete[] _pBits; }

void Bitmap::Set(Size pos) {
  if (!Get(pos)) {
    _pBits[pos >> 3] |= (1 << (pos & 7));
    ++_nUsed;
  }
}

void Bitmap::Unset(Size pos) {
  if (Get(pos)) {
    _pBits[pos >> 3] &= ~(1 << (pos & 7));
    --_nUsed;
  }
}

bool Bitmap::Get(Size pos) const { return _pBits[pos >> 3] & (1 << (pos & 7)); }

Size Bitmap::GetSize() const { return _nSize; }

Size Bitmap::GetUsed() const { return _nUsed; }

bool Bitmap::Empty() const { return _nUsed == 0; }

bool Bitmap::Full() const { return _nUsed == _nSize; }

void Bitmap::Load(const uint8_t *pBits) {
  memcpy(_pBits, pBits, (_nSize - 1) / 8 + 1);
  for (Size i = 0; i < _nSize; ++i)
    if (Get(i)) ++_nUsed;
}

void Bitmap::Store(uint8_t *pBits) {
  memcpy(pBits, _pBits, (_nSize - 1) / 8 + 1);
}

}  // namespace thdb
