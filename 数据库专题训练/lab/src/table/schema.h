#ifndef THDB_SCHEMA_H_
#define THDB_SCHEMA_H_

#include "defines.h"
#include "table/column.h"

namespace thdb {

class Schema {
 public:
  Schema(const std::vector<Column> &iColVec);
  ~Schema() = default;

  Size GetSize() const;
  Column GetColumn(Size nPos) const;

 private:
  std::vector<Column> _iColVec;
};

}  // namespace thdb

#endif