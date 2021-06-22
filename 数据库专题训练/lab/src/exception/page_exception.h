#ifndef THDB_PAGE_EXCEPT_H_
#define THDB_PAGE_EXCEPT_H_

#include "defines.h"
#include "exception/exception.h"
#include "field/field.h"

namespace thdb {

class PageException : public Exception {
  virtual const char* what() const throw() { return "Page Exception"; }
};

class RecordPageException : public PageException {
 public:
  RecordPageException(SlotID nSlotID) : _nSlotID(nSlotID) {
    _msg = "Slot " + std::to_string(_nSlotID) + " is not used";
  }
  virtual const char* what() const throw() { return _msg.c_str(); }
 private:
  SlotID _nSlotID;
  String _msg;
};

class RecordTypeException : public PageException {
 public:
  virtual const char* what() const throw() { return "Unknown record type";; }
};

}  // namespace thdb

#endif
