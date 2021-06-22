#include "condition/not_condition.h"

namespace thdb {
NotCondition::NotCondition(Condition *pCond) : _pCond(pCond) {}

NotCondition::~NotCondition() { delete _pCond; }

bool NotCondition::Match(const Record &iRecord) const {
  return !_pCond->Match(iRecord);
}

}  // namespace thdb
