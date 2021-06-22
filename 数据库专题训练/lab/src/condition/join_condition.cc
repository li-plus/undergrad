#include "condition/join_condition.h"

namespace thdb {

JoinCondition::JoinCondition(const String &sTableA, const String &sColA,
                             const String &sTableB, const String &sColB) {
  this->sTableA = sTableA;
  this->sTableB = sTableB;
  this->sColA = sColA;
  this->sColB = sColB;
}

bool JoinCondition::Match(const Record &iRecord) const { return true; }

ConditionType JoinCondition::GetType() const {
  return ConditionType::JOIN_TYPE;
}

}  // namespace thdb
