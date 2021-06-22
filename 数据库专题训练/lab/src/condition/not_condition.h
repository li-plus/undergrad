#ifndef THDB_NOT_CONDITION_H_
#define THDB_NOT_CONDITION_H_

#include "condition/condition.h"

namespace thdb {
class NotCondition : public Condition {
 public:
  NotCondition(Condition *pCond);
  ~NotCondition();
  bool Match(const Record &iRecord) const override;

 private:
  Condition *_pCond;
};
}  // namespace thdb

#endif