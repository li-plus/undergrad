#include "field/field.h"

namespace thdb {
String toString(FieldType iType) {
  if (iType == FieldType::INT_TYPE) {
    return "Integer";
  } else if (iType == FieldType::FLOAT_TYPE) {
    return "Float";
  } else if (iType == FieldType::STRING_TYPE) {
    return "String";
  } else if (iType == FieldType::NONE_TYPE) {
    return "None";
  } else {
    return "Error";
  }
}
}  // namespace thdb
