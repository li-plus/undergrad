#ifndef THDB_FLOAT_FIELD_H_
#define THDB_FLOAT_FIELD_H_

#include "field.h"

namespace thdb {

class FloatField : public Field {
 public:
  FloatField() = default;
  FloatField(const double &fData);
  ~FloatField() = default;

  void SetData(const uint8_t *src, Size nSize) override;
  void GetData(uint8_t *dst, Size nSize) const override;

  FieldType GetType() const override;

  String ToString() const override;

  Field *Copy() const override;

  double GetFloatData() const;

 private:
  double _fData;
};

bool operator==(const FloatField &a, const FloatField &b);

bool operator<(const FloatField &a, const FloatField &b);

bool operator<=(const FloatField &a, const FloatField &b);

bool operator>(const FloatField &a, const FloatField &b);

bool operator>=(const FloatField &a, const FloatField &b);

bool operator!=(const FloatField &a, const FloatField &b);

}  // namespace thdb

#endif  // THDB_FLOAT_FIELD_H_
