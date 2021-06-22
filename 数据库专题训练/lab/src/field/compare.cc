#include <assert.h>

#include "field/fields.h"

namespace thdb {

bool Less(Field *pA, Field *pB, FieldType iType) {
  if (iType == FieldType::INT_TYPE) {
    IntField *pIntA = dynamic_cast<IntField *>(pA);
    IntField *pIntB = dynamic_cast<IntField *>(pB);
    return *pIntA < *pIntB;
  } else if (iType == FieldType::FLOAT_TYPE) {
    FloatField *pFloatA = dynamic_cast<FloatField *>(pA);
    FloatField *pFloatB = dynamic_cast<FloatField *>(pB);
    return *pFloatA < *pFloatB;
  } else {
    return false;
  }
}

bool Equal(Field *pA, Field *pB, FieldType iType) {
  if (iType == FieldType::INT_TYPE) {
    IntField *pIntA = dynamic_cast<IntField *>(pA);
    IntField *pIntB = dynamic_cast<IntField *>(pB);
    return *pIntA == *pIntB;
  } else if (iType == FieldType::FLOAT_TYPE) {
    FloatField *pFloatA = dynamic_cast<FloatField *>(pA);
    FloatField *pFloatB = dynamic_cast<FloatField *>(pB);
    return *pFloatA == *pFloatB;
  } else {
    return false;
  }
}

bool Greater(Field *pA, Field *pB, FieldType iType) {
  if (iType == FieldType::INT_TYPE) {
    IntField *pIntA = dynamic_cast<IntField *>(pA);
    IntField *pIntB = dynamic_cast<IntField *>(pB);
    return *pIntA > *pIntB;
  } else if (iType == FieldType::FLOAT_TYPE) {
    FloatField *pFloatA = dynamic_cast<FloatField *>(pA);
    FloatField *pFloatB = dynamic_cast<FloatField *>(pB);
    return *pFloatA > *pFloatB;
  } else {
    return false;
  }
}

}  // namespace thdb