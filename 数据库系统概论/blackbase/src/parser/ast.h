#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "parser_defs.h"

typedef enum {
    SV_TYPE_INT,
    SV_TYPE_FLOAT,
    SV_TYPE_STRING
} sv_type_t;

typedef struct {
    sv_type_t type;
    int len;
} sv_type_len_t;

typedef enum {
    SV_EQ,
    SV_LT,
    SV_GT,
    SV_NE,
    SV_LE,
    SV_GE,
} sv_comp_op_t;

typedef enum {
    // sysStmt
    N_SHOW_DATABASES,
    // dbStmt
    N_CREATE_DATABASE,
    N_DROP_DATABASE,
    N_USE_DATABASE,
    N_SHOW_TABLES,
    // tbStmt
    N_CREATE_TABLE,
    N_DROP_TABLE,
    N_DESC_TABLE,
    N_INSERT,
    N_DELETE,
    N_UPDATE,
    N_SELECT_FROM,

    N_SET_CLAUSE,

    N_COLUMN_DEF,
    N_PRIMARY_KEY,
    N_COL,
    N_FOREIGN_KEY,

    // idxStmt
    N_CREATE_INDEX,
    N_DROP_INDEX,

    // expr
    N_BINARY_EXPR,

    // misc
    N_LIST,
    // values
    N_IDENTIFIER,
    N_VALUE_INT,
    N_VALUE_FLOAT,
    N_VALUE_STRING,
    N_SQLNULL,

    // alter stmt
    N_ALTER_ADD_FIELD,
    N_ALTER_DROP_COL,
    N_ALTER_CHANGE_COL,
    N_ALTER_RENAME_TABLE,
    N_ALTER_DROP_PRIMARY_KEY,
    N_ALTER_ADD_PRIMARY_KEY,
    N_ALTER_ADD_FOREIGN_KEY,
    N_ALTER_DROP_FOREIGN_KEY,
} sv_node_kind_t;

typedef struct sv_node {
    sv_node_kind_t kind;

    union {
        /* database statement nodes */
        struct {
            char *dbname;
        } create_database;

        struct {
            char *dbname;
        } drop_database;

        struct {
            char *dbname;
        } use_database;

        /* table statement nodes */
        struct {
            char *tbname;
            struct sv_node *attrlist;
        } create_table;

        struct {
            char *tbname;
        } drop_table;

        struct {
            char *tbname;
        } desc_table;

        struct {
            char *tbname;
            struct sv_node *value_lists;
        } insert;

        struct {
            char *tbname;
            struct sv_node *where_clause;
        } delete_from;

        struct {
            char *tbname;
            struct sv_node *set_clauses;
            struct sv_node *where_clause;
        } update_set;

        struct {
            struct sv_node *col_list;
            struct sv_node *tb_list;
            struct sv_node *where_clause;
        } select_from;

        struct {
            char *colname;
            struct sv_node *value;
        } set_clause;

        struct {
            char *colname;
            sv_type_len_t type_len;
            bool nullable;
        } column_def;

        struct {
            char *tbname;
            char *colname;
        } col;

        struct {
            struct sv_node *colnamelist;
        } primary_key;

        struct {
            struct sv_node *local_colnames;
            char *tbname;
            struct sv_node *colnames;
        } foreign_key;

        struct {
            struct sv_node *left;
            struct sv_node *right;
            sv_comp_op_t op;
        } binary_expr;

        /* identifier */
        char *identifier;

        /* values */
        int value_int;
        float value_float;
        char *value_string;

        /* index statement nodes */
        struct {
            char *idxname;
            char *tbname;
            struct sv_node *colnames;
        } create_index;

        struct {
            char *idxname;
        } drop_index;

        /* alter statement nodes */
        struct {
            char *tbname;
            struct sv_node *field;
        } add_key;

        struct {
            char *tbname;
            char *colname;
        } drop_col;

        struct {
            char *tbname;
            char *colname;
            struct sv_node *field;
        } change_col;

        struct {
            char *tbname;
            char *new_tbname;
        } rename_table;

        struct {
            char *tbname;
        } drop_primary_key;

        struct {
            char *tbname;
            struct sv_node *colnames;
        } add_primary_key;

        struct {
            char *tbname;
            char *fkname;
            struct sv_node *colnames;
            char *ref_tbname;
            struct sv_node *ref_colnames;
        } add_foreign_key;

        struct {
            char *tbname;
            char *fkname;
        } drop_foreign_key;

        /* list node */
        struct {
            struct sv_node *curr;
            struct sv_node *next;
        } list;
    };
} sv_node_t;

extern sv_node_t *parse_tree;

void parser_init();

char *alloc_str(int size);

// sysStmt
sv_node_t *alloc_show_databases();

// dbStmt
sv_node_t *alloc_create_database(char *dbname);
sv_node_t *alloc_drop_database(char *dbname);
sv_node_t *alloc_use_database(char *dbname);
sv_node_t *alloc_show_tables();

// tbStmt
sv_node_t *alloc_create_table(char *tbname, sv_node_t *attrlist);
sv_node_t *alloc_drop_table(char *tbname);
sv_node_t *alloc_desc_table(char *tbname);
sv_node_t *alloc_insert(char *tbname, sv_node_t *value_lists);
sv_node_t *alloc_delete_from(char *tbname, sv_node_t *where_clause);
sv_node_t *alloc_update_set(char *tbname, sv_node_t *set_clauses, sv_node_t *where_clause);
sv_node_t *alloc_select_from(sv_node_t *collist, sv_node_t *tb_list, sv_node_t *where_clause);

sv_node_t *alloc_set_clause(char *colname, sv_node_t *value);
sv_node_t *alloc_column_def(char *colname, sv_type_len_t type_len, bool nullable);
sv_node_t *alloc_primary_key(sv_node_t *colnamelist);
sv_node_t *alloc_col(char *tbname, char *colname);
sv_node_t *alloc_foreign_key(sv_node_t *local_colnames, char *tbname, sv_node_t *colnames);

// expr
sv_node_t *alloc_binary_expr(sv_node_t *left, sv_comp_op_t op, sv_node_t *right);

// values
sv_node_t *alloc_identifier(char *identifier);
sv_node_t *alloc_value_int(int i);
sv_node_t *alloc_value_float(float f);
sv_node_t *alloc_value_string(char *s);
sv_node_t *alloc_sqlnull();

// list
sv_node_t *alloc_list(sv_node_t *curr);
sv_node_t *sv_list_push_front(sv_node_t *head, sv_node_t *curr);

// idx stmt
sv_node_t *alloc_create_index(char *idxname, char *tbname, sv_node_t *colnames);
sv_node_t *alloc_drop_index(char *idxname);

// alter stmt
sv_node_t *alloc_alter_add_field(char *tbname, sv_node_t *field);
sv_node_t *alloc_alter_drop_col(char *tbname, char *colname);
sv_node_t *alloc_alter_change_col(char *tbname, char *colname, sv_node_t *field);
sv_node_t *alloc_alter_rename_table(char *tbname, char *new_tbname);
sv_node_t *alloc_alter_drop_primary_key(char *tbname);
sv_node_t *alloc_alter_add_primary_key(char *tbname, sv_node_t *colnames);
sv_node_t *alloc_alter_add_foreign_key(char *tbname, char *fkname, sv_node_t *colnames, char *ref_tbname,
                                       sv_node_t *ref_colnames);
sv_node_t *alloc_alter_drop_foreign_key(char *tbname, char *fkname);

#ifdef __cplusplus
}
#endif
