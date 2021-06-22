#ifndef THDB_INDEX_CONDITION_H_
#define THDB_INDEX_CONDITION_H_

#include "condition/condition.h"
#include "field/fields.h"

namespace thdb {

class IndexCondition : public Condition {
 public:
  IndexCondition(const String &sTableName, const String &sColName, double fMin,
                 double fMax, FieldType iType);
  ~IndexCondition();

  bool Match(const Record &iRecord) const override;
  ConditionType GetType() const override;

  std::pair<String, String> GetIndexName() const;
  std::pair<Field *, Field *> GetIndexRange() const;

 private:
  String _sTableName, _sColName;
  Field *_pLow, *_pHigh;
};

}  // namespace thdb

#endif
