#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "rm_defs.h"
#include "defs.h"

RC rm_create_file(const char *filename, int record_size);

RC rm_destroy_file(const char *filename);

RC rm_open_file(const char *filename, rm_file_handle_t *fh);

RC rm_close_file(rm_file_handle_t *fh);

#ifdef __cplusplus
}
#endif
