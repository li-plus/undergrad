#ifndef THDB_SPECIAL_EXCEPTION_H_
#define THDB_SPECIAL_EXCEPTION_H_

#include "exception/exception.h"

namespace thdb {

class SpecialException : public Exception {
 public:
  virtual const char* what() const throw() { return "Special Exception"; }
};

}  // namespace thdb

#endif