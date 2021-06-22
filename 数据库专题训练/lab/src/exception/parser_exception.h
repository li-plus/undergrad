#ifndef THDB_PARSER_EXCEPTION_H_
#define THDB_PARSER_EXCEPTION_H_

#include "defines.h"
#include "exception/exception.h"

namespace thdb {
class ParserException : public Exception {
 public:
  ParserException(const String& sMsg) : _sMsg(sMsg) {}
  const char* what() const throw() { return _sMsg.c_str(); }

 private:
  String _sMsg;
};
}  // namespace thdb

#endif