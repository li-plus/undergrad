#include "condition/range_condition.h"

#include <assert.h>

#include <cmath>

#include "field/fields.h"

namespace thdb {
RangeCondition::RangeCondition(FieldID nPos, const double &fMin,
                               const double &fMax)
    : _nPos(nPos), _fMin(fMin), _fMax(fMax) {}

bool RangeCondition::Match(const Record &iRecord) const {
  Field *pField = iRecord.GetField(_nPos);
  if (pField->GetType() == FieldType::NONE_TYPE) return false;
  if (pField->GetType() == FieldType::INT_TYPE) {
    auto pIntField = dynamic_cast<IntField *>(pField);
    int dData = pIntField->GetIntData();
    int fMin = (_fMin < INT32_MIN) ? INT32_MIN : (ceil(_fMin));
    int fMax = (_fMax > INT32_MAX) ? INT32_MAX : (ceil(_fMax));
    return (dData >= fMin) && (dData < fMax);
  } else if (pField->GetType() == FieldType::FLOAT_TYPE) {
    auto pFloatField = dynamic_cast<FloatField *>(pField);
    double fData = pFloatField->GetFloatData();
    return (fData >= _fMin) && (fData < _fMax);
  } else {
    // TODO: Throw Error Here
    assert(false);
  }
}

}  // namespace thdb
