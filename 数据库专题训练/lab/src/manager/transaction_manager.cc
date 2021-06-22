#include "manager/transaction_manager.h"

#include "system/instance.h"

namespace thdb {

Transaction *TransactionManager::Begin() {
  _instance->GetRecoveryManager()->LogBegin(_txnId);
  auto txn = new Transaction(_txnId, _activeTxns);
  _activeTxns[_txnId] = txn;
  _txnId++;
  return txn;
}

void TransactionManager::Commit(Transaction *txn) {
  _instance->GetRecoveryManager()->LogCommit(txn->txnId);
  _activeTxns.erase(txn->txnId);
}

void TransactionManager::Abort(Transaction *txn) {
  _instance->GetRecoveryManager()->LogAbort(txn->txnId);
}

Transaction *TransactionManager::RecoverBegin(TxnID txnId) {
  auto txn = new Transaction(txnId, _activeTxns);
  _activeTxns[_txnId] = txn;
  _txnId = std::max(txnId + 1, _txnId);
  return txn;
}

void TransactionManager::RecoverCommit(TxnID txnId) {
  _activeTxns.erase(txnId);
}

void TransactionManager::RecoveryAbort(TxnID txnId) {}

}  // namespace thdb
