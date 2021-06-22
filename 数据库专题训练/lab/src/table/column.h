#ifndef THDB_COLUMN_H_
#define THDB_COLUMN_H_

#include "defines.h"
#include "field/field.h"

namespace thdb {

class Column {
 public:
  Column(const String &sName, FieldType iType);
  Column(const String &sName, FieldType iType, Size nSize);
  ~Column() = default;

  String GetName() const;
  FieldType GetType() const;
  Size GetSize() const;

 private:
  String _sName;
  FieldType _iType;
  Size _nSize;
};

}  // namespace thdb

#endif