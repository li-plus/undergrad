#ifndef THDB_NONE_FIELD_H_
#define THDB_NONE_FIELD_H_

#include "field.h"

namespace thdb {

class NoneField : public Field {
 public:
  NoneField() = default;
  ~NoneField() = default;

  void SetData(const uint8_t *src, Size nSize) override;
  void GetData(uint8_t *dst, Size nSize) const override;

  FieldType GetType() const override;

  Field *Copy() const override;

  String ToString() const override;
};

}  // namespace thdb

#endif  // THDB_NONE_FIELD_H_
