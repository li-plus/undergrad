#include "display.h"

namespace thdb {

void PrintTable(std::vector<Result *> &results) {
  for (auto result : results) {
    result->Display();
    delete result;
  }
}

}  // namespace thdb
