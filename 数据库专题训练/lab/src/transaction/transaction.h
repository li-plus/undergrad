#ifndef TRANSACTION_H_
#define TRANSACTION_H_

#include <map>

#include "defines.h"

namespace thdb {

class Transaction {
 public:
  TxnID txnId;
  std::map<TxnID, Transaction*> activeTxns;

 public:
  Transaction(TxnID txnId_, std::map<TxnID, Transaction*> activeTxns_)
      : txnId(txnId_), activeTxns(std::move(activeTxns_)) {}
};

}  // namespace thdb

#endif  // TRANSACTION_H_
