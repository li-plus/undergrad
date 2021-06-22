#include "condition/or_condition.h"

namespace thdb {

OrCondition::OrCondition(const std::vector<Condition *> &iCondVec)
    : _iCondVec(iCondVec) {}

OrCondition::~OrCondition() {
  for (const auto &pCond : _iCondVec) delete pCond;
}

void OrCondition::PushBack(Condition *pCond) { _iCondVec.push_back(pCond); }

bool OrCondition::Match(const Record &iRecord) const {
  for (auto it = _iCondVec.begin(); it != _iCondVec.end(); ++it) {
    if ((*it)->Match(iRecord))
      return true;
    else
      continue;
  }
  return false;
}

}  // namespace thdb
