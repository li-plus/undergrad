#ifndef THDB_OS_EXCEPT_H_
#define THDB_OS_EXCEPT_H_

#include "defines.h"
#include "exception/exception.h"
#include "macros.h"

namespace thdb {

class OsException : public Exception {
 public:
  virtual const char* what() const throw() { return "Os Exception"; }
};

class NewPageException : public OsException {
 public:
  virtual const char* what() const throw() { return "Cannot allocate new page"; }
};

class PageNotInitException : public OsException {
 public:
  PageNotInitException(int pid) : _pid(pid) {
    _msg = "Page " + std::to_string(_pid) + " does not initialize";
  }
  virtual const char* what() const throw() { return _msg.c_str(); }

 private:
  PageID _pid;
  String _msg;
};

class PageOutOfSizeException : public OsException {
 public:
  virtual const char* what() const throw() { return "Page out of size"; }
};

}  // namespace thdb

#endif
