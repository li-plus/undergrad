#include "int_field.h"

#include <cassert>
#include <cstring>

namespace thdb {

IntField::IntField(const int &nData) : _nData(nData) {}

void IntField::SetData(const uint8_t *src, Size nSize) {
  assert(nSize == 4);
  memcpy((uint8_t *)&_nData, src, nSize);
}

void IntField::GetData(uint8_t *dst, Size nSize) const {
  assert(nSize == 4);
  memcpy(dst, (uint8_t *)&_nData, nSize);
}

FieldType IntField::GetType() const { return FieldType::INT_TYPE; }

String IntField::ToString() const { return std::to_string(_nData); }

int IntField::GetIntData() const { return _nData; }

Field *IntField::Copy() const { return new IntField(_nData); }

bool operator==(const IntField &a, const IntField &b) {
  return a.GetIntData() == b.GetIntData();
}

bool operator<(const IntField &a, const IntField &b) {
  return a.GetIntData() < b.GetIntData();
}

bool operator<=(const IntField &a, const IntField &b) {
  return (a < b) || (a == b);
}

bool operator>(const IntField &a, const IntField &b) { return !(a <= b); }

bool operator>=(const IntField &a, const IntField &b) { return !(a < b); }

bool operator!=(const IntField &a, const IntField &b) { return !(a == b); }

}  // namespace thdb
