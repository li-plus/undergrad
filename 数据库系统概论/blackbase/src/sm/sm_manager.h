#pragma once

#include "ql/ql_defs.h"
#include "sm_defs.h"
#include "rm/rm.h"
#include "ix/ix.h"
#include <vector>

extern db_meta_t db_meta;

extern rm_file_handle_t sm_fhs[SM_MAX_TABLES];

void sm_init(const char *root_dir);

RC sm_destroy();

int sm_find_table(const char *tbname);

int sm_find_field(int tbidx, const char *field_name);

RC sm_show_databases();

bool sm_database_exists(const char *dbname);

RC sm_create_database(const char *dbname);

RC sm_drop_database(const char *dbname);

RC sm_use_database(const char *dbname);

RC sm_show_tables();

RC sm_create_table(const char *tbname, int num_fields, field_info_t *fields);

RC sm_drop_table(const char *tbname);

RC sm_desc_table(const char *tbname);

RC sm_create_index(const char *tbname, const char *colname);

RC sm_drop_index(const char *tbname, const char *colname);

RC sm_add_primary_key(const char *tbname, const std::vector<std::string> &colnames);

RC sm_drop_primary_key(const char *tbname);

RC sm_rename_table(const char *tbname, const char *new_tbname);

RC sm_add_foreign_key(const char *tbname, const char *fkname, const std::vector<std::string> &colnames,
                      const char *ref_tbname, const std::vector<std::string> &ref_colnames);

RC sm_drop_foreign_key(const char *tbname, const char *fkname);

RC sm_drop_col(const char *tbname, const char *colname);

RC sm_add_col(const char *tbname, char *colname, bool nullable, attr_type_t type, int len);
