#ifndef THDB_BACKEND_H_
#define THDB_BACKEND_H_

#include "defines.h"
#include "result/results.h"
#include "system/instance.h"

namespace thdb {

bool Exists();
void Init();
void Close();
void Clear();
void Help();
std::vector<Result *> Execute(Instance *pDB, const String &sSQL);

}  // namespace thdb

#endif