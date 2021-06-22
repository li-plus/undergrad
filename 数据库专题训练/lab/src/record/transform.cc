#include "record/transform.h"

#include "exception/exceptions.h"

namespace thdb {

Transform::Transform(FieldID nFieldID, FieldType iType, const String &sRaw)
    : _nFieldID(nFieldID), _iType(iType), _sRaw(sRaw) {}

FieldID Transform::GetPos() const { return _nFieldID; }

Field *Transform::GetField() const {
  Field *pField = nullptr;
  if (_sRaw == "NULL") {
    pField = new NoneField();
  }
  if (_iType == FieldType::INT_TYPE) {
    pField = new IntField(std::stoi(_sRaw));
  } else if (_iType == FieldType::FLOAT_TYPE) {
    pField = new FloatField(std::stod(_sRaw));
  } else if (_iType == FieldType::STRING_TYPE) {
    pField = new StringField(_sRaw.substr(1, _sRaw.size() - 2));
  } else {
    throw Exception();
  }
  return pField;
}

}  // namespace thdb
