#ifndef THDB_CONDITION_H_
#define THDB_CONDITION_H_

#include "record/record.h"

namespace thdb {

enum class ConditionType { SIMPLE_TYPE = 0, JOIN_TYPE = 1, INDEX_TYPE = 2 };

/**
 * @brief 条件检索的条件
 *
 */
class Condition {
 public:
  virtual ~Condition() = default;
  /**
   * @brief 判断记录是否符合当前条件
   *
   * @param iRecord 记录
   * @return true 符合
   * @return false 不符合
   */
  virtual bool Match(const Record &iRecord) const = 0;
  virtual ConditionType GetType() const;
};

}  // namespace thdb

#endif
