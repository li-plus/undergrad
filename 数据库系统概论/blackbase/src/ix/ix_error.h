#pragma once

#include "ix_defs.h"

#define IX_BAD_FILENAME         (IX_ERROR_START + 0)
#define IX_ENTRY_NOT_FOUND      (IX_ERROR_START + 1)
#define IX_ERROR_END            (IX_ERROR_START + 2)

//#define IX_EOF                  (IX_ERROR_START + 10)// End of index file
//
//#define IX_BADINDEXSPEC         (IX_ERROR_START + 0) // Bad Specification for Index File
//#define IX_BADINDEXNAME         (IX_ERROR_START + 1) // Bad index name
//#define IX_INVALIDINDEXHANDLE   (IX_ERROR_START + 2) // FileHandle used is invalid
//#define IX_INVALIDINDEXFILE     (IX_ERROR_START + 3) // Bad index file
//#define IX_NODEFULL             (IX_ERROR_START + 4) // A node in the file is full
//
//#define IX_INVALIDBUCKET        (IX_ERROR_START + 6) // Bucket trying to access is invalid
//#define IX_DUPLICATEENTRY       (IX_ERROR_START + 7) // Trying to enter a duplicate entry
//#define IX_INVALIDSCAN          (IX_ERROR_START + 8) // Invalid IX_Indexscsan

const char* ix_str_error(RC rc);
