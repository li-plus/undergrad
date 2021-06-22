#ifndef RECOVERY_MANAGER_H_
#define RECOVERY_MANAGER_H_

#include <fstream>

#include "defines.h"
#include "table/table.h"

namespace thdb {

class Instance;

class RecoveryManager {
 public:
  RecoveryManager(Instance *instance);
  ~RecoveryManager() = default;

  void Redo();
  void Undo();

  void LogInsert(const std::string &tableName,
                 const std::vector<std::string> &rawVec, TxnID txnId);
  void LogBegin(TxnID txnId);
  void LogCommit(TxnID txnId);
  void LogAbort(TxnID txnId);
  void LogCreateTable(const std::string &tableName, const Schema &schema,
                      bool useTxn);

 private:
  Instance *_instance;
  std::ofstream _logger;
};

}  // namespace thdb

#endif  // RECOVERY_MANAGER_H_
