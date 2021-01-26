#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "defs.h"

int yyparse();

typedef struct yy_buffer_state *YY_BUFFER_STATE;

YY_BUFFER_STATE yy_scan_string(const char *str);

void yy_delete_buffer(YY_BUFFER_STATE buffer);

#define PARSER_NODE_POOL_SIZE   1024
#define PARSER_STR_POOL_SIZE    65536

#ifdef __cplusplus
}
#endif
