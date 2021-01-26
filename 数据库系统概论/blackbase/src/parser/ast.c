#include "ast.h"
#include <assert.h>
#include <stdio.h>

sv_node_t *parse_tree;
sv_node_t node_pool[PARSER_NODE_POOL_SIZE];
int node_pool_top;
char str_pool[PARSER_STR_POOL_SIZE];
int str_pool_top;

void parser_init() {
    parse_tree = NULL;
    node_pool_top = 0;
    str_pool_top = 0;
}

static sv_node_t *alloc_node(sv_node_kind_t kind) {
    assert(node_pool_top < PARSER_NODE_POOL_SIZE);
    sv_node_t *node = &node_pool[node_pool_top++];
    node->kind = kind;
    return node;
}

char *alloc_str(int size) {
    char *str = str_pool + str_pool_top;
    str_pool_top += size;
    assert(str_pool_top <= PARSER_STR_POOL_SIZE);
    return str;
}

sv_node_t *alloc_show_databases() {
    sv_node_t *node = alloc_node(N_SHOW_DATABASES);
    return node;
}

sv_node_t *alloc_create_database(char *dbname) {
    sv_node_t *node = alloc_node(N_CREATE_DATABASE);
    node->create_database.dbname = dbname;
    return node;
}

sv_node_t *alloc_drop_database(char *dbname) {
    sv_node_t *node = alloc_node(N_DROP_DATABASE);
    node->drop_database.dbname = dbname;
    return node;
}

sv_node_t *alloc_use_database(char *dbname) {
    sv_node_t *node = alloc_node(N_USE_DATABASE);
    node->use_database.dbname = dbname;
    return node;
}

sv_node_t *alloc_show_tables() {
    sv_node_t *node = alloc_node(N_SHOW_TABLES);
    return node;
}

sv_node_t *alloc_create_table(char *tbname, sv_node_t *attrlist) {
    sv_node_t *node = alloc_node(N_CREATE_TABLE);
    node->create_table.tbname = tbname;
    node->create_table.attrlist = attrlist;
    return node;
}

sv_node_t *alloc_drop_table(char *tbname) {
    sv_node_t *node = alloc_node(N_DROP_TABLE);
    node->drop_table.tbname = tbname;
    return node;
}

sv_node_t *alloc_desc_table(char *tbname) {
    sv_node_t *node = alloc_node(N_DESC_TABLE);
    node->desc_table.tbname = tbname;
    return node;
}

sv_node_t *alloc_insert(char *tbname, sv_node_t *value_lists) {
    sv_node_t *node = alloc_node(N_INSERT);
    node->insert.tbname = tbname;
    node->insert.value_lists = value_lists;
    return node;
}

sv_node_t *alloc_delete_from(char *tbname, sv_node_t *where_clause) {
    sv_node_t *node = alloc_node(N_DELETE);
    node->delete_from.tbname = tbname;
    node->delete_from.where_clause = where_clause;
    return node;
}

sv_node_t *alloc_update_set(char *tbname, sv_node_t *set_clauses, sv_node_t *where_clause) {
    sv_node_t *node = alloc_node(N_UPDATE);
    node->update_set.tbname = tbname;
    node->update_set.set_clauses = set_clauses;
    node->update_set.where_clause = where_clause;
    return node;
}

sv_node_t *alloc_select_from(sv_node_t *collist, sv_node_t *tb_list, sv_node_t *where_clause) {
    sv_node_t *node = alloc_node(N_SELECT_FROM);
    node->select_from.col_list = collist;
    node->select_from.tb_list = tb_list;
    node->select_from.where_clause = where_clause;
    return node;
}

sv_node_t *alloc_set_clause(char *colname, sv_node_t *value) {
    sv_node_t *node = alloc_node(N_SET_CLAUSE);
    node->set_clause.colname = colname;
    node->set_clause.value = value;
    return node;
}

sv_node_t *alloc_column_def(char *colname, sv_type_len_t type_len, bool nullable) {
    sv_node_t *node = alloc_node(N_COLUMN_DEF);
    node->column_def.colname = colname;
    node->column_def.type_len = type_len;
    node->column_def.nullable = nullable;
    return node;
}

sv_node_t *alloc_primary_key(sv_node_t *colnamelist) {
    sv_node_t *node = alloc_node(N_PRIMARY_KEY);
    node->primary_key.colnamelist = colnamelist;
    return node;
}

sv_node_t *alloc_foreign_key(sv_node_t *local_colnames, char *tbname, sv_node_t *colnames) {
    sv_node_t *node = alloc_node(N_FOREIGN_KEY);
    node->foreign_key.local_colnames = local_colnames;
    node->foreign_key.tbname = tbname;
    node->foreign_key.colnames = colnames;
    return node;
}

sv_node_t *alloc_col(char *tbname, char *colname) {
    sv_node_t *node = alloc_node(N_COL);
    node->col.tbname = tbname;
    node->col.colname = colname;
    return node;
}

sv_node_t *alloc_binary_expr(sv_node_t *left, sv_comp_op_t op, sv_node_t *right) {
    sv_node_t *node = alloc_node(N_BINARY_EXPR);
    node->binary_expr.left = left;
    node->binary_expr.op = op;
    node->binary_expr.right = right;
    return node;
}

sv_node_t *alloc_identifier(char *identifier) {
    sv_node_t *node = alloc_node(N_IDENTIFIER);
    node->identifier = identifier;
    return node;
}

sv_node_t *alloc_value_int(int i) {
    sv_node_t *node = alloc_node(N_VALUE_INT);
    node->value_int = i;
    return node;
}

sv_node_t *alloc_value_float(float f) {
    sv_node_t *node = alloc_node(N_VALUE_FLOAT);
    node->value_float = f;
    return node;
}

sv_node_t *alloc_value_string(char *s) {
    sv_node_t *node = alloc_node(N_VALUE_STRING);
    node->value_string = s;
    return node;
}

sv_node_t *alloc_sqlnull() {
    sv_node_t *node = alloc_node(N_SQLNULL);
    return node;
}

sv_node_t *alloc_list(sv_node_t *curr) {
    sv_node_t *node = alloc_node(N_LIST);
    node->list.curr = curr;
    node->list.next = NULL;
    return node;
}

sv_node_t *sv_list_push_front(sv_node_t *head, sv_node_t *curr) {
    sv_node_t *node = alloc_node(N_LIST);
    node->list.curr = curr;
    node->list.next = head;
    return node;
}

sv_node_t *alloc_create_index(char *idxname, char *tbname, sv_node_t *colnames) {
    sv_node_t *node = alloc_node(N_CREATE_INDEX);
    node->create_index.idxname = idxname;
    node->create_index.tbname = tbname;
    node->create_index.colnames = colnames;
    return node;
}

sv_node_t *alloc_drop_index(char *idxname) {
    sv_node_t *node = alloc_node(N_DROP_INDEX);
    node->drop_index.idxname = idxname;
    return node;
}

sv_node_t *alloc_alter_add_field(char *tbname, sv_node_t *field) {
    sv_node_t *node = alloc_node(N_ALTER_ADD_FIELD);
    node->add_key.tbname = tbname;
    node->add_key.field = field;
    return node;
}

sv_node_t *alloc_alter_drop_col(char *tbname, char *colname) {
    sv_node_t *node = alloc_node(N_ALTER_DROP_COL);
    node->drop_col.tbname = tbname;
    node->drop_col.colname = colname;
    return node;
}

sv_node_t *alloc_alter_change_col(char *tbname, char *colname, sv_node_t *field) {
    sv_node_t *node = alloc_node(N_ALTER_CHANGE_COL);
    node->change_col.tbname = tbname;
    node->change_col.colname = colname;
    node->change_col.field = field;
    return node;
}

sv_node_t *alloc_alter_rename_table(char *tbname, char *new_tbname) {
    sv_node_t *node = alloc_node(N_ALTER_RENAME_TABLE);
    node->rename_table.tbname = tbname;
    node->rename_table.new_tbname = new_tbname;
    return node;
}

sv_node_t *alloc_alter_drop_primary_key(char *tbname) {
    sv_node_t *node = alloc_node(N_ALTER_DROP_PRIMARY_KEY);
    node->drop_primary_key.tbname = tbname;
    return node;
}

sv_node_t *alloc_alter_add_primary_key(char *tbname, sv_node_t *colnames) {
    sv_node_t *node = alloc_node(N_ALTER_ADD_PRIMARY_KEY);
    node->add_primary_key.tbname = tbname;
    node->add_primary_key.colnames = colnames;
    return node;
}

sv_node_t *alloc_alter_add_foreign_key(char *tbname, char *fkname, sv_node_t *colnames, char *ref_tbname,
                                       sv_node_t *ref_colnames) {
    sv_node_t *node = alloc_node(N_ALTER_ADD_FOREIGN_KEY);
    node->add_foreign_key.tbname = tbname;
    node->add_foreign_key.fkname = fkname;
    node->add_foreign_key.colnames = colnames;
    node->add_foreign_key.ref_tbname = ref_tbname;
    node->add_foreign_key.ref_colnames = ref_colnames;
    return node;
}

sv_node_t *alloc_alter_drop_foreign_key(char *tbname, char *fkname) {
    sv_node_t *node = alloc_node(N_ALTER_DROP_FOREIGN_KEY);
    node->drop_foreign_key.tbname = tbname;
    node->drop_foreign_key.fkname = fkname;
    return node;
}
