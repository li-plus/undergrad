#ifndef THDB_FIELD_H_
#define THDB_FIELD_H_

#include "defines.h"

namespace thdb {

enum class FieldType {
  NONE_TYPE = 0,
  INT_TYPE = 1,
  FLOAT_TYPE = 2,
  STRING_TYPE = 3
};

class Field {
 public:
  virtual void SetData(const uint8_t *src, Size nSize) = 0;
  virtual void GetData(uint8_t *dst, Size nSize) const = 0;
  virtual FieldType GetType() const = 0;
  virtual String ToString() const = 0;
  virtual Field *Copy() const = 0;

  virtual ~Field() = default;
};

String toString(FieldType iType);

}  // namespace thdb

#endif  // THDB_FIELD_H_
