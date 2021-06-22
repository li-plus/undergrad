#include "condition/index_condition.h"

#include <assert.h>
#include <math.h>

namespace thdb {

IndexCondition::IndexCondition(const String &sTableName, const String &sColName,
                               double fMin, double fMax, FieldType iType)
    : _sTableName(sTableName), _sColName(sColName) {
  if (iType == FieldType::INT_TYPE) {
    int dMin = (fMin < INT32_MIN) ? INT32_MIN : (ceil(fMin));
    int dMax = (fMax > INT32_MAX) ? INT32_MAX : (ceil(fMax));
    _pLow = new IntField(dMin);
    _pHigh = new IntField(dMax);
  } else if (iType == FieldType::FLOAT_TYPE) {
    _pLow = new FloatField(fMin);
    _pHigh = new FloatField(fMax);
  } else {
    assert(false);
  }
}

IndexCondition::~IndexCondition() {
  delete _pLow;
  delete _pHigh;
}

bool IndexCondition::Match(const Record &iRecord) const { return true; }

ConditionType IndexCondition::GetType() const {
  return ConditionType::INDEX_TYPE;
}

std::pair<String, String> IndexCondition::GetIndexName() const {
  return {_sTableName, _sColName};
}

std::pair<Field *, Field *> IndexCondition::GetIndexRange() const {
  return {_pLow, _pHigh};
}

}  // namespace thdb
