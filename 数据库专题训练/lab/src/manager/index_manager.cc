#include "manager/index_manager.h"

#include <algorithm>

#include "assert.h"
#include "exception/index_exception.h"
#include "macros.h"
#include "minios/os.h"
#include "page/record_page.h"
#include "record/fixed_record.h"

namespace thdb {

String GetIndexName(const String &sTableName, const String &sColName) {
  return sTableName + " " + sColName;
}

std::pair<String, String> SplitIndexName(const String &sName) {
  auto nPos = sName.find(' ');
  assert(nPos != String::npos);
  return {sName.substr(0, nPos), sName.substr(nPos + 1)};
}

IndexManager::IndexManager() {
  Load();
  Init();
}

IndexManager::~IndexManager() {
  Store();
  for (const auto &iPair : _iIndexMap)
    if (iPair.second) delete iPair.second;
}

Index *IndexManager::GetIndex(const String &sTableName,
                              const String &sColName) {
  String sIndexName = GetIndexName(sTableName, sColName);
  if (_iIndexMap.find(sIndexName) == _iIndexMap.end()) {
    if (_iIndexIDMap.find(sIndexName) == _iIndexIDMap.end())
      return nullptr;
    else {
      _iIndexMap[sIndexName] = new Index(_iIndexIDMap[sIndexName]);
      return _iIndexMap[sIndexName];
    }
  }
  return _iIndexMap[sIndexName];
}

Index *IndexManager::AddIndex(const String &sTableName, const String &sColName,
                              FieldType iType) {
  if (IsIndex(sTableName, sColName)) throw IndexException();
  String sIndexName = GetIndexName(sTableName, sColName);
  Index *pIndex = new Index(iType);
  PageID nRoot = pIndex->GetRootID();
  delete pIndex;
  pIndex = new Index(nRoot);
  _iIndexIDMap[sIndexName] = nRoot;
  _iIndexMap[sIndexName] = pIndex;
  if (_iTableIndexes.find(sTableName) == _iTableIndexes.end()) {
    _iTableIndexes[sTableName] = std::vector<String>{sColName};
  } else {
    _iTableIndexes[sTableName].push_back(sColName);
  }
  return pIndex;
}

void IndexManager::DropIndex(const String &sTableName, const String &sColName) {
  if (!IsIndex(sTableName, sColName)) throw IndexException();
  String sIndexName = GetIndexName(sTableName, sColName);
  Index *pIndex = GetIndex(sTableName, sColName);
  pIndex->Clear();
  delete pIndex;
  PageID nRoot = _iIndexIDMap[sIndexName];
  MiniOS::GetOS()->DeletePage(nRoot);
  _iIndexIDMap.erase(sIndexName);
  _iIndexMap.erase(sIndexName);
  assert(_iTableIndexes.find(sTableName) != _iTableIndexes.end());
  _iTableIndexes[sTableName].erase(std::find(_iTableIndexes[sTableName].begin(),
                                             _iTableIndexes[sTableName].end(),
                                             sColName));
  if (_iTableIndexes[sTableName].size() == 0) _iTableIndexes.erase(sTableName);
}

std::vector<String> IndexManager::GetTableIndexes(
    const String &sTableName) const {
  if (_iTableIndexes.find(sTableName) == _iTableIndexes.end()) return {};
  return _iTableIndexes.find(sTableName)->second;
}

bool IndexManager::IsIndex(const String &sTableName, const String &sColName) {
  String sIndexName = GetIndexName(sTableName, sColName);
  return _iIndexIDMap.find(sIndexName) != _iIndexIDMap.end();
}

bool IndexManager::HasIndex(const String &sTableName) const {
  if (_iTableIndexes.find(sTableName) == _iTableIndexes.end()) return false;
  return _iTableIndexes.find(sTableName)->second.size() > 0;
}

void IndexManager::Store() {
  RecordPage *pPage = new RecordPage(INDEX_MANAGER_PAGEID);
  pPage->Clear();
  FixedRecord *pRecord = new FixedRecord(
      2, {FieldType::STRING_TYPE, FieldType::INT_TYPE}, {INDEX_NAME_SIZE, 4});
  for (const auto &iPair : _iIndexIDMap) {
    StringField *pString = new StringField(iPair.first);
    IntField *pInt = new IntField(iPair.second);
    pRecord->SetField(0, pString);
    pRecord->SetField(1, pInt);
    uint8_t pData[INDEX_NAME_SIZE + 4];
    pRecord->Store(pData);
    pPage->InsertRecord(pData);
  }
  delete pRecord;
  delete pPage;
}

void IndexManager::Load() {
  RecordPage *pPage = new RecordPage(INDEX_MANAGER_PAGEID);
  FixedRecord *pRecord = new FixedRecord(
      2, {FieldType::STRING_TYPE, FieldType::INT_TYPE}, {INDEX_NAME_SIZE, 4});
  for (Size i = 0; i < pPage->GetCap(); ++i) {
    if (!pPage->HasRecord(i)) break;
    uint8_t *pData = pPage->GetRecord(i);
    pRecord->Load(pData);
    StringField *pString = dynamic_cast<StringField *>(pRecord->GetField(0));
    IntField *pInt = dynamic_cast<IntField *>(pRecord->GetField(1));
    _iIndexIDMap[pString->GetString()] = pInt->GetIntData();
    delete[] pData;
  }
  delete pRecord;
  delete pPage;
}

void IndexManager::Init() {
  for (const auto &it : _iIndexIDMap) {
    auto iPair = SplitIndexName(it.first);
    if (_iTableIndexes.find(iPair.first) == _iTableIndexes.end()) {
      _iTableIndexes[iPair.first] = std::vector<String>{iPair.second};
    } else {
      _iTableIndexes[iPair.first].push_back(iPair.second);
    }
  }
}

std::vector<std::pair<String, String>> IndexManager::GetIndexInfos() const {
  std::vector<std::pair<String, String>> iInfos{};
  for (const auto &it : _iIndexIDMap)
    iInfos.push_back(SplitIndexName(it.first));
  return iInfos;
}

}  // namespace thdb
