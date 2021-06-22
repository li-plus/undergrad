#ifndef THDB_EXCEPT_H_
#define THDB_EXCEPT_H_

#include <exception>

namespace thdb {

class Exception : public std::exception {
 public:
  virtual const char* what() const throw() { return "Exception"; }
};

}  // namespace thdb

#endif