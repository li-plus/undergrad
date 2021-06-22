#ifndef THDB_ERROR_LISTENER_H_
#define THDB_ERROR_LISTENER_H_

#include "antlr4-runtime.h"

namespace antlr4 {
class SyntaxErrorListener : public BaseErrorListener {
  void syntaxError(Recognizer *recognizer, Token *offendingSymbol, size_t line,
                   size_t charPositionInLine, const std::string &msg,
                   std::exception_ptr e) override;
};
}  // namespace antlr4

#endif