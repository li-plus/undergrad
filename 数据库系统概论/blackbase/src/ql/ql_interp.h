#pragma once

#include "ql_defs.h"
#include "parser/parser.h"

attr_type_t interp_sv_type(sv_type_t sv_type);

RC exec_sql(sv_node_t *root);
