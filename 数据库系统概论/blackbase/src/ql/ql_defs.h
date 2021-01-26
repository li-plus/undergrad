#pragma once

#include "defs.h"

typedef struct {
    char root_dir[SM_MAX_NAME_LEN];
    char curr_db[SM_MAX_NAME_LEN];
} sm_config_t;

typedef struct {
    char *name;
    attr_type_t type;
    int len;
    bool nullable;
    int is_primary;
    bool has_frgn_key;
    char *frgn_tab;
    char *frgn_col;
} field_info_t;

typedef struct {
    char tbname[SM_MAX_NAME_LEN];
    char name[SM_MAX_FIELD_NAME_LEN];
    attr_type_t type;
    int len;
    int nullable;
    int index;
    int offset;
    int is_primary;
    int has_frgn_key;
    char frgn_tab[SM_MAX_NAME_LEN];
    char frgn_col[SM_MAX_FIELD_NAME_LEN];
} field_meta_t;

typedef struct {
    char name[SM_MAX_NAME_LEN];
    int num_fields;
    field_meta_t fields[SM_MAX_FIELDS];
} table_meta_t;

typedef struct {
    int num_tables;
    table_meta_t tables[SM_MAX_TABLES];
} db_meta_t;
