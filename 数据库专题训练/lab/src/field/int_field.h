#ifndef THDB_INT_FIELD_H_
#define THDB_INT_FIELD_H_

#include "field.h"

namespace thdb {

class IntField : public Field {
 public:
  IntField() = default;
  IntField(const int &nData);
  ~IntField() = default;

  void SetData(const uint8_t *src, Size nSize) override;
  void GetData(uint8_t *dst, Size nSize) const override;

  FieldType GetType() const override;

  std::string ToString() const override;

  Field *Copy() const override;

  int GetIntData() const;

 private:
  int _nData;
};

bool operator==(const IntField &a, const IntField &b);

bool operator<(const IntField &a, const IntField &b);

bool operator<=(const IntField &a, const IntField &b);

bool operator>(const IntField &a, const IntField &b);

bool operator>=(const IntField &a, const IntField &b);

bool operator!=(const IntField &a, const IntField &b);

}  // namespace thdb

#endif  // THDB_INT_FIELD_H_
