#ifndef THDB_STRING_FIELD_H_
#define THDB_STRING_FIELD_H_

#include <cstring>
#include <string>

#include "field.h"

namespace thdb {

class StringField : public Field {
 public:
  StringField(Size nSize);
  StringField(const String &sData);
  ~StringField() = default;

  void SetData(const uint8_t *src, Size nSize) override;
  void GetData(uint8_t *dst, Size nSize) const override;

  FieldType GetType() const override;

  Field *Copy() const override;

  String ToString() const override;

  String GetString() const;

 private:
  String _sData;
};

bool operator==(const StringField &a, const StringField &b);

bool operator<(const StringField &a, const StringField &b);

bool operator<=(const StringField &a, const StringField &b);

bool operator>(const StringField &a, const StringField &b);

bool operator>=(const StringField &a, const StringField &b);

bool operator!=(const StringField &a, const StringField &b);

}  // namespace thdb

#endif  // THDB_STRING_FIELD_H_
