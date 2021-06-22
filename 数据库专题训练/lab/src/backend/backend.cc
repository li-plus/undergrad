#include "backend/backend.h"

#include <fstream>

#include "antlr4-runtime.h"
#include "exception/exceptions.h"
#include "macros.h"
#include "minios/os.h"
#include "page/record_page.h"
#include "parser/ErrorListener.h"
#include "parser/SQLLexer.h"
#include "parser/SQLParser.h"
#include "parser/SystemVisitor.h"

namespace thdb {

using namespace antlr4;

bool Exists() {
  std::ifstream fin{"THDB_BITMAP"};
  if (fin)
    return true;
  else
    return false;
}

void Init() {
  if (Exists()) return;
  printf("Database Init.\n");

  RecordPage *pNotUsed1 = new RecordPage(256, true);
  RecordPage *pNotUsed2 = new RecordPage(256, true);
  RecordPage *pTableManagerPage = new RecordPage(TABLE_NAME_SIZE + 4, true);
  RecordPage *pIndexManagerPage = new RecordPage(128, true);

  printf("Build Finish.\n");

  delete pNotUsed1;
  delete pNotUsed2;
  delete pTableManagerPage;
  delete pIndexManagerPage;

  MiniOS::WriteBack();
}

void Close() {
  // Close System
  MiniOS::WriteBack();
}

void Clear() {
  if (!Exists()) return;
  std::remove("THDB_BITMAP");
  std::remove("THDB_PAGE");
  std::remove("THDB_LOG");
}

void Help() { printf("Sorry, Help Tips is developing.\n"); }

std::vector<Result *> Execute(Instance *pDB, const String &sSQL) {
  ANTLRInputStream sInputStream(sSQL);
  SQLLexer iLexer(&sInputStream);
  CommonTokenStream sTokenStream(&iLexer);
  SQLParser iParser(&sTokenStream);
  iParser.removeErrorListeners();
  SyntaxErrorListener *pListener = new SyntaxErrorListener();
  iParser.addErrorListener(pListener);
  auto iTree = iParser.program();
  delete pListener;
  SystemVisitor iVisitor{pDB};
  return iVisitor.visit(iTree);
}

}  // namespace thdb
