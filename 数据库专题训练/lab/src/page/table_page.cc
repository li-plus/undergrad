#include "page/table_page.h"

#include <assert.h>

#include <algorithm>

#include "exception/exceptions.h"
#include "page/record_page.h"

namespace thdb {

const PageOffset COLUMN_LEN_OFFSET = 16;
const PageOffset COLUMN_NAME_LEN_OFFSET = 20;
const PageOffset HEAD_PAGE_OFFSET = 24;
const PageOffset TAIL_PAGE_OFFSET = 28;

const PageOffset COLUMN_TYPE_OFFSET = 0;
const PageOffset COLUMN_SIZE_OFFSET = 64;
const PageOffset COLUMN_NAME_OFFSET = 192;

TablePage::TablePage(const Schema &iSchema) : Page() {
  for (Size i = 0; i < iSchema.GetSize(); ++i) {
    Column iCol = iSchema.GetColumn(i);
    _iColMap[iCol.GetName()] = i;
    _iTypeVec.push_back(iCol.GetType());
    _iSizeVec.push_back(iCol.GetSize());
  }
  assert(_iColMap.size() == _iTypeVec.size());
  RecordPage *pPage = new RecordPage(GetTotalSize(), true);
  _nHeadID = _nTailID = pPage->GetPageID();
  delete pPage;
  _bModified = true;
}

TablePage::TablePage(PageID nPageID) : Page(nPageID) {
  Load();
  _bModified = false;
}

TablePage::~TablePage() {
  if (_bModified) Store();
}

Size TablePage::GetFieldSize() const { return _iTypeVec.size(); }

FieldID TablePage::GetFieldID(const String &sColName) const {
  if (_iColMap.find(sColName) == _iColMap.end()) throw TableException();
  return _iColMap.find(sColName)->second;
}

std::vector<FieldType> TablePage::GetTypeVec() const { return _iTypeVec; }

std::vector<Size> TablePage::GetSizeVec() const { return _iSizeVec; }

Size TablePage::GetTotalSize() const {
  Size nTotal = 0;
  for (const auto &nSize : _iSizeVec) nTotal += nSize;
  return nTotal;
}

PageID TablePage::GetHeadID() const { return _nHeadID; }

PageID TablePage::GetTailID() const { return _nTailID; }

void TablePage::SetHeadID(PageID nHeadID) {
  _nHeadID = nHeadID;
  _bModified = true;
}

void TablePage::SetTailID(PageID nTailID) {
  _nTailID = nTailID;
  _bModified = true;
}

bool CmpByValue(const std::pair<String, FieldID> &a,
                const std::pair<String, FieldID> &b) {
  return a.second < b.second;
}

String BuildColumnsString(const std::map<String, FieldID> &iColMap) {
  if (iColMap.size() == 0) return "";
  std::vector<std::pair<String, FieldID>> iTempVec{iColMap.begin(),
                                                   iColMap.end()};
  std::sort(iTempVec.begin(), iTempVec.end(), CmpByValue);
  String sColumnsName = "";
  for (const auto &iPair : iTempVec) {
    sColumnsName += iPair.first;
    sColumnsName += "%";
  }
  return sColumnsName.substr(0, sColumnsName.size());
}

std::map<String, FieldID> LoadColumnsString(const String &sName) {
  std::map<String, FieldID> iColMap;
  size_t nBegin = 0, nEnd = 0;
  Size nPos = 0;
  nEnd = sName.find('%', nBegin);
  while (nEnd != std::string::npos) {
    String sKey = sName.substr(nBegin, nEnd - nBegin);
    iColMap[sKey] = nPos++;
    nBegin = nEnd + 1;
    nEnd = sName.find('%', nBegin);
  }
  return iColMap;
}

void TablePage::Store() {
  SetHeader((uint8_t *)&_nHeadID, 4, HEAD_PAGE_OFFSET);
  SetHeader((uint8_t *)&_nTailID, 4, TAIL_PAGE_OFFSET);
  FieldID iFieldSize = _iSizeVec.size();
  SetHeader((uint8_t *)&iFieldSize, 4, COLUMN_LEN_OFFSET);
  for (Size i = 0; i < iFieldSize; ++i) {
    int nType = (int)_iTypeVec[i];
    SetData((uint8_t *)&nType, 1, COLUMN_TYPE_OFFSET + i);
  }
  for (Size i = 0; i < iFieldSize; ++i) {
    Size nSize = _iSizeVec[i];
    SetData((uint8_t *)&nSize, 2, COLUMN_SIZE_OFFSET + 2 * i);
  }
  String sColumnsName = BuildColumnsString(_iColMap);
  Size sColNameLen = sColumnsName.size();
  SetHeader((uint8_t *)&sColNameLen, 4, COLUMN_NAME_LEN_OFFSET);
  SetData((uint8_t *)sColumnsName.c_str(), sColNameLen, COLUMN_NAME_OFFSET);
}

void TablePage::Load() {
  GetHeader((uint8_t *)&_nHeadID, 4, HEAD_PAGE_OFFSET);
  GetHeader((uint8_t *)&_nTailID, 4, TAIL_PAGE_OFFSET);
  FieldID iFieldSize = 0;
  GetHeader((uint8_t *)&iFieldSize, 4, COLUMN_LEN_OFFSET);
  for (Size i = 0; i < iFieldSize; ++i) {
    int nType = 0;
    GetData((uint8_t *)&nType, 1, COLUMN_TYPE_OFFSET + i);
    _iTypeVec.push_back(FieldType(nType));
  }
  for (Size i = 0; i < iFieldSize; ++i) {
    Size nSize = 0;
    GetData((uint8_t *)&nSize, 2, COLUMN_SIZE_OFFSET + 2 * i);
    _iSizeVec.push_back(nSize);
  }
  Size sColNameLen = 0;
  GetHeader((uint8_t *)&sColNameLen, 4, COLUMN_NAME_LEN_OFFSET);
  char *pTemp = new char[sColNameLen + 1];
  pTemp[sColNameLen] = '\0';
  GetData((uint8_t *)pTemp, sColNameLen, COLUMN_NAME_OFFSET);
  String sName{pTemp};
  _iColMap = LoadColumnsString(sName);
  delete[] pTemp;
}

FieldID TablePage::GetPos(const String &sCol) { return _iColMap[sCol]; }

FieldType TablePage::GetType(const String &sCol) {
  return _iTypeVec[GetPos(sCol)];
}

Size TablePage::GetSize(const String &sCol) { return _iSizeVec[GetPos(sCol)]; }

}  // namespace thdb
