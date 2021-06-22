#include "none_field.h"

#include <cassert>

namespace thdb {

void NoneField::SetData(const uint8_t *src, Size nSize) { assert(nSize == 0); }

void NoneField::GetData(uint8_t *dst, Size nSize) const { assert(nSize == 0); }

FieldType NoneField::GetType() const { return FieldType::NONE_TYPE; }

String NoneField::ToString() const { return ""; }

Field *NoneField::Copy() const { return new NoneField(); }

}  // namespace thdb
