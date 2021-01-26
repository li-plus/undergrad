#pragma once

#include "rm_defs.h"

#define RM_RECORD_NOT_FOUND     (RM_ERROR_START + 0)
#define RM_ERROR_END            (RM_ERROR_START + 1)
//
//#define RM_INVALIDRID           (RM_ERROR_START + 0) // invalid RID
//#define RM_BADRECORDSIZE        (RM_ERROR_START + 1) // record size is invalid
//#define RM_INVALID_RECORD       (RM_ERROR_START + 2) // invalid record
//#define RM_INVALIDBITOPERATION  (RM_ERROR_START + 3) // invalid page header bit ops
//#define RM_PAGEFULL             (RM_ERROR_START + 4) // no more free slots on page
//#define RM_INVALIDFILE          (RM_ERROR_START + 5) // file is corrupt/not there
//#define RM_INVALIDFILEHANDLE    (RM_ERROR_START + 6) // filehandle is improperly set up
//#define RM_INVALIDSCAN          (RM_ERROR_START + 7) // scan is improperly set up
//#define RM_ENDOFPAGE            (RM_ERROR_START + 8) // end of a page
//#define RM_EOF                  (RM_ERROR_START + 9) // end of file
//#define RM_BADFILENAME          (RM_ERROR_START + 10)

const char* rm_str_error(RC rc);
