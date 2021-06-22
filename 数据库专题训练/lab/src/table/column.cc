#include "column.h"

namespace thdb {

Column::Column(const String &sName, FieldType iType)
    : _sName(sName), _iType(iType) {
  switch (iType) {
    case FieldType::INT_TYPE:
      _nSize = 4;
      break;
    case FieldType::FLOAT_TYPE:
      _nSize = 8;
      break;
    default:
      _nSize = 0;
      break;
  }
}

Column::Column(const String &sName, FieldType iType, Size nSize)
    : _sName(sName), _iType(iType), _nSize(nSize) {}

String Column::GetName() const { return _sName; }

FieldType Column::GetType() const { return _iType; }

Size Column::GetSize() const { return _nSize; }

}  // namespace thdb
