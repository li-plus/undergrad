#ifndef TRANSACTION_MANAGER_H_
#define TRANSACTION_MANAGER_H_

#include <set>
#include <unordered_map>

#include "defines.h"
#include "transaction/transaction.h"

namespace thdb {

class Instance;

class TransactionManager {
 public:
  TransactionManager(Instance *instance) : _txnId(0), _instance(instance) {}
  ~TransactionManager() = default;

  Transaction *GetTxn(TxnID txnId) const { return _activeTxns.at(txnId); }
  std::map<TxnID, Transaction *> GetActiveTxns() { return _activeTxns; }

  Transaction *Begin();
  void Commit(Transaction *txn);
  void Abort(Transaction *txn);

  Transaction *RecoverBegin(TxnID txnId);
  void RecoverCommit(TxnID txnId);
  void RecoveryAbort(TxnID txnId);

 private:
  TxnID _txnId;
  std::map<TxnID, Transaction *> _activeTxns;
  Instance *_instance;
};

}  // namespace thdb

#endif  // TRANSACTION_MANAGER_H_
