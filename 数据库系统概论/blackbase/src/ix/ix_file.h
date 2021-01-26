#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "ix_defs.h"
#include "defs.h"
#include <stdio.h>

char *ix_get_filename(const char *filename, int index_no);

bool ix_exists(const char *filename, int index_no);

RC ix_create_index(const char *filename, int index_no, attr_type_t attr_type, int attr_len);

RC ix_destroy_index(const char *filename, int index_no);

RC ix_open_index(const char *filename, int index_no, ix_file_handle_t *fh);

RC ix_close_index(ix_file_handle_t *ih);

#ifdef __cplusplus
}
#endif
