#include "manager/recovery_manager.h"

#include "system/instance.h"

namespace thdb {

RecoveryManager::RecoveryManager(Instance* instance)
    : _instance(instance),
      _logger("THDB_LOG", std::fstream::out | std::fstream::app) {}

void RecoveryManager::Redo() {
  // Redo from last checkpoint
  std::ifstream ifs("THDB_LOG");
  auto txnMgr = _instance->GetTransactionManager();
  std::string logType;
  while (ifs >> logType) {
    if (logType == "BEGIN") {
      TxnID txnId;
      ifs >> txnId;
      txnMgr->RecoverBegin(txnId);
    } else if (logType == "COMMIT") {
      TxnID txnId;
      ifs >> txnId;
      txnMgr->RecoverCommit(txnId);
    } else if (logType == "ABORT") {
      TxnID txnId;
      ifs >> txnId;
      txnMgr->RecoveryAbort(txnId);
    } else if (logType == "INSERT") {
      std::string tableName;
      ifs >> tableName;
      TxnID txnId;
      ifs >> txnId;
      auto txn = txnMgr->GetTxn(txnId);
      Size size;
      ifs >> size;
      std::vector<std::string> rawVec(size);
      for (auto& rec : rawVec) {
        ifs >> rec;
      }
      _instance->Insert(tableName, rawVec, txn);
    } else if (logType == "CREATE_TABLE") {
      std::string tableName;
      ifs >> tableName;
      Size size;
      ifs >> size;
      std::vector<Column> cols;
      for (Size colId = 0; colId < size; colId++) {
        std::string colName;
        ifs >> colName;
        FieldType colType;
        int tmpColType;
        ifs >> tmpColType;
        colType = (FieldType)tmpColType;
        Size colSize;
        ifs >> colSize;
        cols.emplace_back(colName, colType, colSize);
      }
      Schema schema(cols);
      bool useTxn;
      ifs >> useTxn;
      _instance->CreateTable(tableName, schema, useTxn);
    } else {
      throw std::runtime_error("Unexpected log type" + logType);
    }
  }
}

void RecoveryManager::Undo() {
  // Undo uncommitted transactions
  auto txnMgr = _instance->GetTransactionManager();
  for (auto& entry : txnMgr->GetActiveTxns()) {
    auto txn = entry.second;
    txnMgr->Abort(txn);
  }
}

void RecoveryManager::LogInsert(const std::string &tableName,
               const std::vector<std::string> &rawVec, TxnID txnId) {
  _logger << "INSERT\n";
  _logger << tableName << "\n";
  _logger << txnId << "\n";
  _logger << rawVec.size() << "\n";
  for (auto &line : rawVec) {
    _logger << line << "\n";
  }
}

void RecoveryManager::LogBegin(TxnID txnId) {
  _logger << "BEGIN\n";
  _logger << txnId << "\n";
}

void RecoveryManager::LogCommit(TxnID txnId) {
  _logger << "COMMIT\n";
  _logger << txnId << "\n";
}

void RecoveryManager::LogAbort(TxnID txnId) {
  _logger << "ABORT\n";
  _logger << txnId << "\n";
}

void RecoveryManager::LogCreateTable(const std::string &tableName, const Schema &schema,
                    bool useTxn) {
  _logger << "CREATE_TABLE\n";
  _logger << tableName << "\n";
  _logger << schema.GetSize() << "\n";
  for (Size i = 0; i < schema.GetSize(); i++) {
    auto col = schema.GetColumn(i);
    _logger << col.GetName() << "\n";
    _logger << (int)col.GetType() << "\n";
    _logger << col.GetSize() << "\n";
  }
  _logger << useTxn << "\n";
}

}  // namespace thdb
