#pragma once

#include "pf_defs.h"

#define PF_UNIX                 (PF_ERROR_START + 0)
#define PF_FILE_EXISTS          (PF_ERROR_START + 1)
#define PF_FILE_NOT_FOUND       (PF_ERROR_START + 2)
#define PF_INCOMPLETE_READ      (PF_ERROR_START + 3)
#define PF_INCOMPLETE_WRITE     (PF_ERROR_START + 4)
#define PF_ERROR_END            (PF_ERROR_START + 5)

//// PF warnings:
//#define PF_EOF              1   // end of file
//#define PF_PAGEPINNED       2   // page pinned in buffer
//#define PF_PAGENOTINBUF     3   // page to be unpinned is not in buffer
//#define PF_PAGEUNPINNED     4   // page already unpinned
//#define PF_PAGEFREE         5   // page already free
//#define PF_INVALIDPAGE      6   // invalid page number
//#define PF_FILEOPEN         7   // file handle already open
//#define PF_CLOSEDFILE       8   // file is closed
//
//// PF errors:
//#define PF_NOMEM            -1  // out of memory
//#define PF_NOBUF            -2  // out of buffer space
//
//// Internal PF errors:
//#define PF_PAGEINBUF        -7  // new allocated page already in buffer
//#define PF_HASHNOTFOUND     -8  // hash table entry not found
//#define PF_HASHPAGEEXIST    -9  // page already exists in hash table
//#define PF_INVALIDNAME      -10 // invalid file name

const char *pf_str_error(RC rc);
