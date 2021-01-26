#include "printer.h"
#include <stdio.h>
#include <assert.h>

void print_offset(int offset) {
    printf("%*s", offset, "");
}

void print_string(const char *str, int offset) {
    print_offset(offset);
    printf("%s\n", str);
}

void print_type_len(sv_type_len_t type_len, int offset) {
    print_offset(offset);
    switch (type_len.type) {
        case SV_TYPE_INT:
            printf("INT(%d)\n", type_len.len);
            break;
        case SV_TYPE_FLOAT:
            printf("FLOAT(%d)\n", type_len.len);
            break;
        case SV_TYPE_STRING:
            printf("STRING(%d)\n", type_len.len);
            break;
        default:
            assert(0);
    }
}

void print_comp_op(sv_comp_op_t op, int offset) {
    print_offset(offset);
    switch (op) {
        case SV_EQ:
            printf("==\n");
            break;
        case SV_LT:
            printf("<\n");
            break;
        case SV_GT:
            printf(">\n");
            break;
        case SV_NE:
            printf("!=\n");
            break;
        case SV_LE:
            printf("<=\n");
            break;
        case SV_GE:
            printf(">=\n");
            break;
        default:
            assert(0);
    }
}

void print_tree(sv_node_t *node, int offset) {
    print_offset(offset);
    offset += 2;
    if (node == NULL) {
        printf("EMPTY_NODE\n");
        return;
    }
    switch (node->kind) {
        case N_SHOW_DATABASES:
            printf("SHOW_DATABASES\n");
            break;
        case N_CREATE_DATABASE:
            printf("CREATE_DATABASE\n");
            print_string(node->create_database.dbname, offset);
            break;
        case N_DROP_DATABASE:
            printf("DROP_DATABASE\n");
            print_string(node->drop_database.dbname, offset);
            break;
        case N_USE_DATABASE:
            printf("USE_DATABASE\n");
            print_string(node->use_database.dbname, offset);
            break;
        case N_SHOW_TABLES:
            printf("SHOW_TABLES\n");
            break;
        case N_CREATE_TABLE:
            printf("CREATE_TABLE\n");
            print_string(node->create_table.tbname, offset);
            print_tree(node->create_table.attrlist, offset);
            break;
        case N_DROP_TABLE:
            printf("DROP_TABLE\n");
            print_string(node->drop_table.tbname, offset);
            break;
        case N_DESC_TABLE:
            printf("DESC_TABLE\n");
            print_string(node->desc_table.tbname, offset);
            break;
        case N_INSERT:
            printf("INSERT\n");
            print_string(node->insert.tbname, offset);
            print_tree(node->insert.value_lists, offset);
            break;
        case N_DELETE:
            printf("DELETE\n");
            print_string(node->delete_from.tbname, offset);
            print_tree(node->delete_from.where_clause, offset);
            break;
        case N_UPDATE:
            printf("UPDATE\n");
            print_string(node->update_set.tbname, offset);
            print_tree(node->update_set.set_clauses, offset);
            print_tree(node->update_set.where_clause, offset);
            break;
        case N_SELECT_FROM:
            printf("SELECT_FROM\n");
            print_tree(node->select_from.col_list, offset);
            print_tree(node->select_from.tb_list, offset);
            print_tree(node->select_from.where_clause, offset);
            break;
        case N_SET_CLAUSE:
            printf("SET_CLAUSE\n");
            print_string(node->set_clause.colname, offset);
            print_tree(node->set_clause.value, offset);
            break;
        case N_COLUMN_DEF:
            printf("COLUMN_DEF\n");
            print_string(node->column_def.colname, offset);
            print_type_len(node->column_def.type_len, offset);
            break;
        case N_PRIMARY_KEY:
            printf("PRIMARY_KEY\n");
            print_tree(node->primary_key.colnamelist, offset);
            break;
        case N_COL:
            printf("COL\n");
            print_string(node->col.tbname, offset);
            print_string(node->col.colname, offset);
            break;
        case N_FOREIGN_KEY:
            printf("FOREIGN_KEY\n");
            print_tree(node->foreign_key.local_colnames, offset);
            print_string(node->foreign_key.tbname, offset);
            print_tree(node->foreign_key.colnames, offset);
            break;
        case N_BINARY_EXPR:
            printf("BINARY_EXPR\n");
            print_comp_op(node->binary_expr.op, offset);
            print_tree(node->binary_expr.left, offset);
            print_tree(node->binary_expr.right, offset);
            break;
        case N_SQLNULL:
            printf("NULL\n");
            break;
        case N_IDENTIFIER:
            printf("IDENTIFIER\n");
            print_string(node->identifier, offset);
            break;
        case N_VALUE_INT:
            printf("%d\n", node->value_int);
            break;
        case N_VALUE_FLOAT:
            printf("%f\n", node->value_float);
            break;
        case N_VALUE_STRING:
            printf("%s\n", node->value_string);
            break;
        case N_LIST:
            printf("LIST\n");
            for (sv_node_t *pos = node; pos != NULL; pos = pos->list.next) {
                print_tree(pos->list.curr, offset);
            }
            break;
        case N_CREATE_INDEX:
            printf("CREATE_INDEX\n");
            print_string(node->create_index.idxname, offset);
            print_string(node->create_index.tbname, offset);
            print_tree(node->create_index.colnames, offset);
            break;
        case N_DROP_INDEX:
            printf("DROP_INDEX\n");
            print_string(node->drop_index.idxname, offset);
            break;
        case N_ALTER_ADD_FIELD:
            printf("ADD_KEY\n");
            print_string(node->add_key.tbname, offset);
            print_tree(node->add_key.field, offset);
            break;
        case N_ALTER_DROP_COL:
            printf("DROP_COL\n");
            print_string(node->drop_col.tbname, offset);
            print_string(node->drop_col.colname, offset);
            break;
        case N_ALTER_CHANGE_COL:
            printf("CHANGE_COL\n");
            print_string(node->change_col.tbname, offset);
            print_string(node->change_col.colname, offset);
            print_tree(node->change_col.field, offset);
            break;
        case N_ALTER_RENAME_TABLE:
            printf("RENAME_TABLE\n");
            print_string(node->rename_table.tbname, offset);
            print_string(node->rename_table.new_tbname, offset);
            break;
        case N_ALTER_DROP_PRIMARY_KEY:
            printf("DROP_PRIMARY_KEY\n");
            print_string(node->drop_primary_key.tbname, offset);
            break;
        case N_ALTER_ADD_PRIMARY_KEY:
            printf("ADD_PRIMARY_KEY\n");
            print_string(node->add_primary_key.tbname, offset);
            print_tree(node->add_primary_key.colnames, offset);
            break;
        case N_ALTER_ADD_FOREIGN_KEY:
            printf("ADD_FOREIGN_KEY\n");
            print_string(node->add_foreign_key.tbname, offset);
            print_string(node->add_foreign_key.fkname, offset);
            print_tree(node->add_foreign_key.colnames, offset);
            print_string(node->add_foreign_key.ref_tbname, offset);
            print_tree(node->add_foreign_key.ref_colnames, offset);
            break;
        case N_ALTER_DROP_FOREIGN_KEY:
            printf("DROP_FOREIGN_KEY\n");
            print_string(node->drop_foreign_key.tbname, offset);
            print_string(node->drop_foreign_key.fkname, offset);
            break;
        default:
            assert(0);
    }
}
