#include "schema.h"

namespace thdb {

Schema::Schema(const std::vector<Column> &iColVec) : _iColVec(iColVec) {}

Size Schema::GetSize() const { return _iColVec.size(); }

Column Schema::GetColumn(Size nPos) const { return _iColVec[nPos]; }

}  // namespace thdb
