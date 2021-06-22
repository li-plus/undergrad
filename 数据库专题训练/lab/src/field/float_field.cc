#include "float_field.h"

#include <cassert>
#include <cstring>
#include <sstream>

namespace thdb {

FloatField::FloatField(const double &fData) : _fData(fData) {}

void FloatField::SetData(const uint8_t *src, Size nSize) {
  assert(nSize == 8);
  memcpy((uint8_t *)&_fData, src, nSize);
}

void FloatField::GetData(uint8_t *dst, Size nSize) const {
  assert(nSize == 8);
  memcpy(dst, (uint8_t *)&_fData, nSize);
}

FieldType FloatField::GetType() const { return FieldType::FLOAT_TYPE; }

String FloatField::ToString() const {
  std::ostringstream strs;
  strs << _fData;
  return strs.str();
}

double FloatField::GetFloatData() const { return _fData; }

Field *FloatField::Copy() const { return new FloatField(_fData); }

bool operator==(const FloatField &a, const FloatField &b) {
  return a.GetFloatData() == b.GetFloatData();
}

bool operator<(const FloatField &a, const FloatField &b) {
  return a.GetFloatData() < b.GetFloatData();
}

bool operator<=(const FloatField &a, const FloatField &b) {
  return (a < b) || (a == b);
}

bool operator>(const FloatField &a, const FloatField &b) { return !(a <= b); }

bool operator>=(const FloatField &a, const FloatField &b) { return !(a < b); }

bool operator!=(const FloatField &a, const FloatField &b) { return !(a == b); }

}  // namespace thdb
