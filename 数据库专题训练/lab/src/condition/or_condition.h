#ifndef THDB_OR_CONDITION_H_
#define THDB_OR_CONDITION_H_

#include "condition/condition.h"

namespace thdb {

class OrCondition : public Condition {
 public:
  OrCondition(const std::vector<Condition *> &iCondVec);
  ~OrCondition();
  bool Match(const Record &iRecord) const override;
  void PushBack(Condition *pCond);

 private:
  std::vector<Condition *> _iCondVec;
};

}  // namespace thdb

#endif