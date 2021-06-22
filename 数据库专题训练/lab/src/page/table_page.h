#ifndef THDB_TABLE_PAGE_H_
#define THDB_TABLE_PAGE_H_

#include "defines.h"
#include "field/field.h"
#include "page/page.h"
#include "table/schema.h"

namespace thdb {

class Table;

class TablePage : public Page {
 public:
  TablePage(const Schema &iSchema);
  TablePage(PageID nPageID);
  ~TablePage();

  Size GetFieldSize() const;
  FieldID GetFieldID(const String &sColName) const;
  std::vector<FieldType> GetTypeVec() const;
  std::vector<Size> GetSizeVec() const;
  Size GetTotalSize() const;

  PageID GetHeadID() const;
  PageID GetTailID() const;
  void SetHeadID(PageID nHeadID);
  void SetTailID(PageID nTailID);

  FieldID GetPos(const String &sCol);
  FieldType GetType(const String &sCol);
  Size GetSize(const String &sCol);

 private:
  void Store();
  void Load();

  std::map<String, FieldID> _iColMap;
  std::vector<FieldType> _iTypeVec;
  std::vector<Size> _iSizeVec;
  PageID _nHeadID, _nTailID;
  bool _bModified = false;

  friend class Table;
};

}  // namespace thdb

#endif