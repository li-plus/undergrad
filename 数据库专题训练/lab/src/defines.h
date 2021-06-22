#ifndef THDB_DEFINE_H_
#define THDB_DEFINE_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <map>
#include <string>
#include <vector>

namespace thdb {

typedef uint32_t PageID;
typedef uint16_t PageOffset;
typedef uint16_t SlotID;
typedef uint16_t FieldID;
typedef std::string String;
typedef uint32_t Size;

typedef std::pair<PageID, SlotID> PageSlotID;

typedef uint32_t TxnID;

}  // namespace thdb

#endif
