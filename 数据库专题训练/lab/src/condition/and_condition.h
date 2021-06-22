#ifndef THDB_AND_CONDITION_H_
#define THDB_AND_CONDITION_H_

#include "condition/condition.h"
#include "defines.h"

namespace thdb {

class AndCondition : public Condition {
 public:
  AndCondition(const std::vector<Condition *> &iCondVec);
  ~AndCondition();
  bool Match(const Record &iRecord) const override;
  void PushBack(Condition *pCond);

 private:
  std::vector<Condition *> _iCondVec;
};

}  // namespace thdb

#endif