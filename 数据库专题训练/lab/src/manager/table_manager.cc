#include "manager/table_manager.h"

#include <assert.h>

#include <cstring>

#include "exception/exceptions.h"
#include "field/fields.h"
#include "minios/os.h"
#include "page/record_page.h"
#include "record/fixed_record.h"

namespace thdb {

TableManager::TableManager() {
  _iTableMap = {};
  Load();
}

TableManager::~TableManager() {
  Store();
  for (const auto &iPair : _iTableMap)
    if (iPair.second) delete iPair.second;
}

Table *TableManager::GetTable(const String &sTableName) {
  if (_iTableMap.find(sTableName) == _iTableMap.end()) {
    if (_iTableIDMap.find(sTableName) == _iTableIDMap.end())
      return nullptr;
    else {
      _iTableMap[sTableName] = new Table(_iTableIDMap[sTableName]);
      return _iTableMap[sTableName];
    }
  }
  return _iTableMap[sTableName];
}

Table *TableManager::AddTable(const String &sTableName, const Schema &iSchema) {
  if (GetTable(sTableName) != nullptr) throw TableExistException(sTableName);
  TablePage *pPage = new TablePage(iSchema);
  PageID nTableID = pPage->GetPageID();
  delete pPage;
  Table *pTable = new Table(nTableID);
  _iTableMap[sTableName] = pTable;
  _iTableIDMap[sTableName] = nTableID;
  return pTable;
}

void TableManager::DropTable(const String &sTableName) {
  if (GetTable(sTableName) == nullptr) throw TableNotExistException(sTableName);
  Table *pTable = GetTable(sTableName);
  pTable->Clear();
  delete pTable;
  PageID nTableID = _iTableIDMap[sTableName];
  MiniOS::GetOS()->DeletePage(nTableID);
  _iTableIDMap.erase(sTableName);
  _iTableMap.erase(sTableName);
}

void TableManager::Store() {
  RecordPage *pPage = new RecordPage(TABLE_MANAGER_PAGEID);
  pPage->Clear();
  FixedRecord *pRecord = new FixedRecord(
      2, {FieldType::STRING_TYPE, FieldType::INT_TYPE}, {TABLE_NAME_SIZE, 4});
  for (const auto &iPair : _iTableIDMap) {
    StringField *pString = new StringField(iPair.first);
    IntField *pInt = new IntField(iPair.second);
    pRecord->SetField(0, pString);
    pRecord->SetField(1, pInt);
    uint8_t pData[TABLE_NAME_SIZE + 4];
    pRecord->Store(pData);
    pPage->InsertRecord(pData);
  }
  delete pRecord;
  delete pPage;
}

void TableManager::Load() {
  RecordPage *pPage = new RecordPage(TABLE_MANAGER_PAGEID);
  FixedRecord *pRecord = new FixedRecord(
      2, {FieldType::STRING_TYPE, FieldType::INT_TYPE}, {TABLE_NAME_SIZE, 4});
  for (Size i = 0; i < pPage->GetCap(); ++i) {
    if (!pPage->HasRecord(i)) break;
    uint8_t *pData = pPage->GetRecord(i);
    pRecord->Load(pData);
    StringField *pString = dynamic_cast<StringField *>(pRecord->GetField(0));
    IntField *pInt = dynamic_cast<IntField *>(pRecord->GetField(1));
    _iTableIDMap[pString->GetString()] = pInt->GetIntData();
    delete[] pData;
  }
  delete pRecord;
  delete pPage;
}

std::vector<String> TableManager::GetTableNames() const {
  std::vector<String> iVec;
  for (const auto &it : _iTableIDMap) iVec.push_back(it.first);
  return iVec;
}

std::vector<String> TableManager::GetColumnNames(const String &sTableName) {
  Table *pTable = GetTable(sTableName);
  return pTable->GetColumnNames();
}

}  // namespace thdb
