#include "system/instance.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>

#include "condition/conditions.h"
#include "exception/exceptions.h"
#include "manager/table_manager.h"
#include "minios/os.h"
#include "record/fixed_record.h"

namespace thdb {

Instance::Instance() {
  MiniOS::GetOS()->Reload();
  _pTableManager = new TableManager();
  _pIndexManager = new IndexManager();
  _pTransactionManager = new TransactionManager(this);
  _pRecoveryManager = new RecoveryManager(this);
}

Instance::~Instance() {
  delete _pTableManager;
  delete _pIndexManager;
  delete _pTransactionManager;
  delete _pRecoveryManager;
}

Table *Instance::GetTable(const String &sTableName) const {
  return _pTableManager->GetTable(sTableName);
}

bool Instance::CreateTable(const String &sTableName, const Schema &iSchema,
                           bool useTxn) {
  _pRecoveryManager->LogCreateTable(sTableName, iSchema, useTxn);
  auto schema = iSchema;
  if (useTxn) {
    std::vector<Column> cols;
    for (Size i = 0; i < iSchema.GetSize(); i++) {
      cols.push_back(iSchema.GetColumn(i));
    }
    cols.emplace_back("DB_TXN_ID", FieldType::INT_TYPE);
    schema = Schema(cols);
  }
  _pTableManager->AddTable(sTableName, schema);
  return true;
}

bool Instance::DropTable(const String &sTableName) {
  for (const auto &sColName : _pIndexManager->GetTableIndexes(sTableName))
    _pIndexManager->DropIndex(sTableName, sColName);
  _pTableManager->DropTable(sTableName);
  return true;
}

FieldID Instance::GetColID(const String &sTableName,
                           const String &sColName) const {
  Table *pTable = GetTable(sTableName);
  if (pTable == nullptr) throw TableException();
  return pTable->GetPos(sColName);
}

FieldType Instance::GetColType(const String &sTableName,
                               const String &sColName) const {
  Table *pTable = GetTable(sTableName);
  if (pTable == nullptr) throw TableException();
  return pTable->GetType(sColName);
}

Size Instance::GetColSize(const String &sTableName,
                          const String &sColName) const {
  Table *pTable = GetTable(sTableName);
  if (pTable == nullptr) throw TableException();
  return pTable->GetSize(sColName);
}

bool CmpPageSlotID(const PageSlotID &iA, const PageSlotID &iB) {
  if (iA.first == iB.first) return iA.second < iB.second;
  return iA.first < iB.first;
}

std::vector<PageSlotID> Intersection(std::vector<PageSlotID> iA,
                                     std::vector<PageSlotID> iB) {
  std::sort(iA.begin(), iA.end(), CmpPageSlotID);
  std::sort(iB.begin(), iB.end(), CmpPageSlotID);
  std::vector<PageSlotID> iRes{};
  std::set_intersection(iA.begin(), iA.end(), iB.begin(), iB.end(),
                        std::back_inserter(iRes));
  return iRes;
}

std::vector<PageSlotID> Instance::Search(
    const String &sTableName, Condition *pCond,
    const std::vector<Condition *> &iIndexCond, Transaction *txn) {
  Table *pTable = GetTable(sTableName);
  if (pTable == nullptr) throw TableException();
  std::vector<PageSlotID> iRes;
  if (iIndexCond.size() > 0) {
    IndexCondition *pIndexCond = dynamic_cast<IndexCondition *>(iIndexCond[0]);
    assert(pIndexCond != nullptr);
    auto iName = pIndexCond->GetIndexName();
    auto iRange = pIndexCond->GetIndexRange();
    iRes =
        GetIndex(iName.first, iName.second)->Range(iRange.first, iRange.second);
    for (Size i = 1; i < iIndexCond.size(); ++i) {
      IndexCondition *pIndexCond =
          dynamic_cast<IndexCondition *>(iIndexCond[i]);
      auto iName = pIndexCond->GetIndexName();
      auto iRange = pIndexCond->GetIndexRange();
      iRes = Intersection(iRes, GetIndex(iName.first, iName.second)
                                    ->Range(iRange.first, iRange.second));
    }
  } else {
    iRes = pTable->SearchRecord(pCond);
  }
  // Check visibility in read view
  if (txn) {
    std::vector<PageSlotID> txnRes;
    txnRes.reserve(iRes.size());
    for (auto &rid : iRes) {
      auto rec = pTable->GetRecord(rid.first, rid.second);
      TxnID txnId = dynamic_cast<IntField *>(rec->GetField(rec->GetSize() - 1))
                        ->GetIntData();
      if (txnId <= txn->txnId && txn->activeTxns.count(txnId) == 0) {
        txnRes.push_back(rid);
      }
    }
    iRes = txnRes;
  }
  return iRes;
}

PageSlotID Instance::Insert(const String &sTableName,
                            const std::vector<String> &iRawVec,
                            Transaction *txn) {
  TxnID logTxnId = txn ? txn->txnId : -1;
  _pRecoveryManager->LogInsert(sTableName, iRawVec, logTxnId);

  auto rawVec = iRawVec;
  if (txn) {
    rawVec.push_back(std::to_string(txn->txnId));
  }
  Table *pTable = GetTable(sTableName);
  if (pTable == nullptr) throw TableException();
  Record *pRecord = pTable->EmptyRecord();
  pRecord->Build(rawVec);
  PageSlotID iPair = pTable->InsertRecord(pRecord);
  // Handle Insert on Index
  if (_pIndexManager->HasIndex(sTableName)) {
    auto iColNames = _pIndexManager->GetTableIndexes(sTableName);
    for (const auto &sCol : iColNames) {
      FieldID nPos = pTable->GetPos(sCol);
      Field *pKey = pRecord->GetField(nPos);
      _pIndexManager->GetIndex(sTableName, sCol)->Insert(pKey, iPair);
    }
  }

  delete pRecord;
  return iPair;
}

uint32_t Instance::Delete(const String &sTableName, Condition *pCond,
                          const std::vector<Condition *> &iIndexCond,
                          Transaction *txn) {
  auto iResVec = Search(sTableName, pCond, iIndexCond);
  Table *pTable = GetTable(sTableName);
  bool bHasIndex = _pIndexManager->HasIndex(sTableName);
  for (const auto &iPair : iResVec) {
    // Handle Delete on Index
    if (bHasIndex) {
      Record *pRecord = pTable->GetRecord(iPair.first, iPair.second);
      auto iColNames = _pIndexManager->GetTableIndexes(sTableName);
      for (const auto &sCol : iColNames) {
        FieldID nPos = pTable->GetPos(sCol);
        Field *pKey = pRecord->GetField(nPos);
        _pIndexManager->GetIndex(sTableName, sCol)->Delete(pKey, iPair);
      }
      delete pRecord;
    }

    pTable->DeleteRecord(iPair.first, iPair.second);
  }
  return iResVec.size();
}

uint32_t Instance::Update(const String &sTableName, Condition *pCond,
                          const std::vector<Condition *> &iIndexCond,
                          const std::vector<Transform> &iTrans,
                          Transaction *txn) {
  auto iResVec = Search(sTableName, pCond, iIndexCond);
  Table *pTable = GetTable(sTableName);
  bool bHasIndex = _pIndexManager->HasIndex(sTableName);
  for (const auto &iPair : iResVec) {
    // Handle Delete on Index
    if (bHasIndex) {
      Record *pRecord = pTable->GetRecord(iPair.first, iPair.second);
      auto iColNames = _pIndexManager->GetTableIndexes(sTableName);
      for (const auto &sCol : iColNames) {
        FieldID nPos = pTable->GetPos(sCol);
        Field *pKey = pRecord->GetField(nPos);
        _pIndexManager->GetIndex(sTableName, sCol)->Delete(pKey, iPair);
      }
      delete pRecord;
    }

    pTable->UpdateRecord(iPair.first, iPair.second, iTrans);

    // Handle Delete on Index
    if (bHasIndex) {
      Record *pRecord = pTable->GetRecord(iPair.first, iPair.second);
      auto iColNames = _pIndexManager->GetTableIndexes(sTableName);
      for (const auto &sCol : iColNames) {
        FieldID nPos = pTable->GetPos(sCol);
        Field *pKey = pRecord->GetField(nPos);
        _pIndexManager->GetIndex(sTableName, sCol)->Insert(pKey, iPair);
      }
      delete pRecord;
    }
  }
  return iResVec.size();
}

Record *Instance::GetRecord(const String &sTableName, const PageSlotID &iPair,
                            Transaction *txn) const {
  Table *pTable = GetTable(sTableName);
  auto rec = pTable->GetRecord(iPair.first, iPair.second);
  if (txn) {
    rec->Remove(rec->GetSize() - 1);
  }
  return rec;
}

std::vector<Record *> Instance::GetTableInfos(const String &sTableName) const {
  std::vector<Record *> iVec{};
  for (const auto &sName : GetColumnNames(sTableName)) {
    FixedRecord *pDesc = new FixedRecord(
        3,
        {FieldType::STRING_TYPE, FieldType::STRING_TYPE, FieldType::INT_TYPE},
        {COLUMN_NAME_SIZE, 10, 4});
    pDesc->SetField(0, new StringField(sName));
    pDesc->SetField(1,
                    new StringField(toString(GetColType(sTableName, sName))));
    pDesc->SetField(2, new IntField(GetColSize(sTableName, sName)));
    iVec.push_back(pDesc);
  }
  return iVec;
}
std::vector<String> Instance::GetTableNames() const {
  return _pTableManager->GetTableNames();
}
std::vector<String> Instance::GetColumnNames(const String &sTableName) const {
  return _pTableManager->GetColumnNames(sTableName);
}

bool Instance::IsIndex(const String &sTableName, const String &sColName) const {
  return _pIndexManager->IsIndex(sTableName, sColName);
}

Index *Instance::GetIndex(const String &sTableName,
                          const String &sColName) const {
  return _pIndexManager->GetIndex(sTableName, sColName);
}

std::vector<Record *> Instance::GetIndexInfos() const {
  std::vector<Record *> iVec{};
  for (const auto &iPair : _pIndexManager->GetIndexInfos()) {
    FixedRecord *pInfo =
        new FixedRecord(4,
                        {FieldType::STRING_TYPE, FieldType::STRING_TYPE,
                         FieldType::STRING_TYPE, FieldType::INT_TYPE},
                        {TABLE_NAME_SIZE, COLUMN_NAME_SIZE, 10, 4});
    pInfo->SetField(0, new StringField(iPair.first));
    pInfo->SetField(1, new StringField(iPair.second));
    pInfo->SetField(
        2, new StringField(toString(GetColType(iPair.first, iPair.second))));
    pInfo->SetField(3, new IntField(GetColSize(iPair.first, iPair.second)));
    iVec.push_back(pInfo);
  }
  return iVec;
}

bool Instance::CreateIndex(const String &sTableName, const String &sColName,
                           FieldType iType) {
  auto iAll = Search(sTableName, nullptr, {});
  _pIndexManager->AddIndex(sTableName, sColName, iType);
  Table *pTable = GetTable(sTableName);
  // Handle Exists Data
  for (const auto &iPair : iAll) {
    FieldID nPos = pTable->GetPos(sColName);
    Record *pRecord = pTable->GetRecord(iPair.first, iPair.second);
    Field *pKey = pRecord->GetField(nPos);
    _pIndexManager->GetIndex(sTableName, sColName)->Insert(pKey, iPair);
    delete pRecord;
  }
  return true;
}

bool Instance::DropIndex(const String &sTableName, const String &sColName) {
  auto iAll = Search(sTableName, nullptr, {});
  Table *pTable = GetTable(sTableName);
  for (const auto &iPair : iAll) {
    FieldID nPos = pTable->GetPos(sColName);
    Record *pRecord = pTable->GetRecord(iPair.first, iPair.second);
    Field *pKey = pRecord->GetField(nPos);
    _pIndexManager->GetIndex(sTableName, sColName)->Delete(pKey, iPair);
    delete pRecord;
  }
  _pIndexManager->DropIndex(sTableName, sColName);
  return true;
}

std::pair<std::vector<String>, std::vector<Record *>> Instance::Join(
    std::map<String, std::vector<PageSlotID>> &iResultMap,
    std::vector<Condition *> &iJoinConds) {
  // LAB3 BEGIN
  // TODO:实现正确且高效的表之间JOIN过程

  // ALERT:由于实现临时表存储具有一定难度，所以允许JOIN过程中将中间结果保留在内存中，不需要存入临时表
  // ALERT:一定要注意，存在JOIN字段值相同的情况，需要特别重视
  // ALERT:针对于不同的JOIN情况（此处只需要考虑数据量和是否为索引列），可以选择使用不同的JOIN算法
  // ALERT:JOIN前已经经过了Filter过程
  // ALERT:建议不要使用不经过优化的NestedLoopJoin算法

  // TIPS:JoinCondition中保存了JOIN两方的表名和列名
  // TIPS:利用GetTable(TableName)的方式可以获得Table*指针，之后利用lab1中的Table::GetRecord获得初始Record*数据
  // TIPs:利用Table::GetColumnNames可以获得Table初始的列名，与初始Record*顺序一致
  // TIPS:Record对象添加了Copy,Sub,Add,Remove函数，方便同学们对于Record进行处理
  // TIPS:利用GetColID/Type/Size三个函数可以基于表名和列名获得列的信息
  // TIPS:利用IsIndex可以判断列是否存在索引
  // TIPS:利用GetIndex可以获得索引Index*指针

  // EXTRA:JOIN的表的数量超过2时，所以需要先计算一个JOIN执行计划（不要求复杂算法）,有兴趣的同学可以自行实现
  // EXTRA:在多表JOIN时，可以采用并查集或执行树来确定执行JOIN的数据内容

  // LAB3 END
  if (iResultMap.empty()) {
    return {};
  }

  std::vector<JoinCondition *> joinConds;
  for (auto &cond : iJoinConds) {
    auto joinCond = dynamic_cast<JoinCondition *>(cond);
    assert(joinCond);
    joinConds.push_back(joinCond);
  }

  auto resMapPos = iResultMap.begin();
  auto &firstTabName = resMapPos->first;
  auto &firstRids = resMapPos->second;

  auto firstTab = GetTable(firstTabName);
  std::vector<Record *> firstRecs;
  for (auto &firstRid : firstRids) {
    auto rec = firstTab->GetRecord(firstRid.first, firstRid.second);
    firstRecs.push_back(rec);
  }

  std::vector<String> joinTabs = {firstTabName};
  std::vector<Record *> joinRecs = std::move(firstRecs);

  resMapPos++;
  while (resMapPos != iResultMap.end()) {
    auto &tabName = resMapPos->first;
    auto &tabRids = resMapPos->second;
    // get all records of current table
    std::vector<Record *> tabRecs;
    auto tab = GetTable(tabName);
    for (auto &rid : tabRids) {
      auto tabRec = tab->GetRecord(rid.first, rid.second);
      tabRecs.push_back(tabRec);
    }
    // get all conditions for current join
    std::vector<JoinCondition *> currJoinConds;
    auto joinCondPos = joinConds.begin();
    while (joinCondPos != joinConds.end()) {
      auto joinCond = *joinCondPos;
      if (joinCond->sTableA == tabName) {
        std::swap(joinCond->sTableA, joinCond->sTableB);
        std::swap(joinCond->sColA, joinCond->sColB);
      }
      if (joinCond->sTableB == tabName &&
          std::find(joinTabs.begin(), joinTabs.end(), joinCond->sTableA) !=
              joinTabs.end()) {
        // if rhs is current table and lhs is joined before, pop and use it
        currJoinConds.push_back(joinCond);
        joinCondPos = joinConds.erase(joinCondPos);
      } else {
        joinCondPos++;
      }
    }
    // join them
    auto joinRes =
        JoinPair(joinTabs, joinRecs, tabName, tabRecs, currJoinConds);

    for (auto &joinRec : joinRecs) {
      delete joinRec;
    }
    joinRecs = std::move(joinRes.second);
    joinTabs = std::move(joinRes.first);

    resMapPos++;
  }
  assert(joinConds.empty());
  return std::make_pair(joinTabs, joinRecs);
}

std::pair<std::vector<String>, std::vector<Record *>> Instance::JoinPair(
    const std::vector<String> &leftTabs, std::vector<Record *> leftRecs,
    const String &rightTab, std::vector<Record *> rightRecs,
    const std::vector<JoinCondition *> &joinConds) const {
  std::vector<String> tabs = leftTabs;
  tabs.push_back(rightTab);

  std::vector<Record *> joinRecs;

  if (joinConds.empty()) {
    // no condition, complete cross product
    for (auto &leftRec : leftRecs) {
      for (auto &rightRec : rightRecs) {
        auto joinRec = leftRec->Copy();
        joinRec->Add(rightRec);
        joinRecs.push_back(joinRec);
      }
    }
  } else {
    // use 1st condition for merge join
    auto joinCond = joinConds.front();
    assert(joinCond->sTableB == rightTab);
    std::vector<JoinCondition *> remainConds(joinConds.begin() + 1,
                                             joinConds.end());

    // sort right records by key
    auto rightColId = GetColID(joinCond->sTableB, joinCond->sColB);
    auto rightColType = GetColType(joinCond->sTableB, joinCond->sColB);
    std::sort(rightRecs.begin(), rightRecs.end(), [&](Record *x, Record *y) {
      return Less(x->GetField(rightColId), y->GetField(rightColId),
                  rightColType);
    });

    // sort left records by key
    auto leftColId =
        GetRecordColumnID(leftTabs, joinCond->sTableA, joinCond->sColA);
    auto leftColType = GetColType(joinCond->sTableA, joinCond->sColA);
    std::sort(leftRecs.begin(), leftRecs.end(), [&](Record *x, Record *y) {
      return Less(x->GetField(leftColId), y->GetField(leftColId), leftColType);
    });

    size_t rightStart = 0;
    for (auto &leftRec : leftRecs) {
      while (rightStart < rightRecs.size() &&
             Greater(leftRec->GetField(leftColId),
                     rightRecs[rightStart]->GetField(rightColId),
                     leftColType)) {
        rightStart++;
      }
      if (rightStart == rightRecs.size()) {
        break;
      }
      for (size_t k = rightStart; k < rightRecs.size(); k++) {
        auto rightRec = rightRecs[k];
        if (!Equal(leftRec->GetField(leftColId), rightRec->GetField(rightColId),
                   leftColType)) {
          break;
        }
        bool isMatch = std::all_of(
            remainConds.begin(), remainConds.end(), [&](JoinCondition *cond) {
              assert(cond->sTableB == rightTab);
              auto filterLeftColId =
                  GetRecordColumnID(leftTabs, cond->sTableA, cond->sColA);
              auto filterRightColId = GetColID(cond->sTableB, cond->sColB);
              auto filterType = GetColType(cond->sTableB, cond->sColB);
              return Equal(leftRec->GetField(filterLeftColId),
                           rightRec->GetField(filterRightColId), filterType);
            });
        if (isMatch) {
          auto joinRec = leftRec->Copy();
          joinRec->Add(rightRec);
          joinRecs.push_back(joinRec);
        }
      }
    }
  }

  return std::make_pair(tabs, joinRecs);
}

FieldID Instance::GetRecordColumnID(const std::vector<String> &tabs,
                                    const String &tab,
                                    const String &col) const {
  FieldID colId = GetColID(tab, col);
  auto tabPos = std::find(tabs.begin(), tabs.end(), tab);
  assert(tabPos != tabs.end());
  for (auto it = tabs.begin(); it != tabPos; it++) {
    colId += GetColumnNames(*it).size();
  }
  return colId;
}

}  // namespace thdb
