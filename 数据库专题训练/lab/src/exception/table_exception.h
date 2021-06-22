#ifndef THDB_TABLE_EXCEPT_H_
#define THDB_TABLE_EXCEPT_H_

#include "defines.h"
#include "exception/exception.h"

namespace thdb {

class TableException : public Exception {
  virtual const char* what() const throw() { return "Table Exception"; }
};

class TableNotExistException : public TableException {
 public:
  TableNotExistException(const String& table) : _table(table) {
    _msg = "table " + _table + " does not exist";
  }

  virtual const char* what() const throw() { return _msg.c_str(); }

 private:
  String _table;
  String _msg;
};

class TableExistException : public TableException {
 public:
  TableExistException(const String& table) : _table(table) {
    _msg = "table " + _table + " already exists";
  }

  virtual const char* what() const throw() { return _msg.c_str(); }

 private:
  String _table;
  String _msg;
};

}  // namespace thdb

#endif
