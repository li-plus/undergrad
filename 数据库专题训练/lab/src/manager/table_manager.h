#ifndef THDB_TABLE_MANAGER_H_
#define THDB_TABLE_MANAGER_H_

#include "defines.h"
#include "macros.h"
#include "table/schema.h"
#include "table/table.h"

namespace thdb {

class TableManager {
 public:
  TableManager();
  ~TableManager();

  Table *GetTable(const String &sTableName);
  Table *AddTable(const String &sTableName, const Schema &iSchema);
  void DropTable(const String &sTableName);

  std::vector<String> GetTableNames() const;
  std::vector<String> GetColumnNames(const String &sTableName);

 private:
  std::map<String, Table *> _iTableMap;
  std::map<String, PageID> _iTableIDMap;

  void Store();
  void Load();
};

}  // namespace thdb

#endif