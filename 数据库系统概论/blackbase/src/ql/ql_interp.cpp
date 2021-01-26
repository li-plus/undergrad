#include "ql_interp.h"
#include "ql_manager.h"
#include "ql_error.h"
#include "sm/sm.h"
#include "defs.h"
#include <assert.h>
#include <string.h>
#include <string>

attr_type_t interp_sv_type(sv_type_t sv_type) {
    switch (sv_type) {
        case SV_TYPE_INT:
            return ATTR_INT;
        case SV_TYPE_FLOAT:
            return ATTR_FLOAT;
        case SV_TYPE_STRING:
            return ATTR_STRING;
        default:
            assert(0);
    }
}

comp_op_t interp_sv_comp_op(sv_comp_op_t sv_op) {
    switch (sv_op) {
        case SV_EQ:
            return OP_EQ;
        case SV_LT:
            return OP_LT;
        case SV_GT:
            return OP_GT;
        case SV_NE:
            return OP_NE;
        case SV_LE:
            return OP_LE;
        case SV_GE:
            return OP_GE;
        default:
            assert(0);
    }
}

std::vector<condition_t> parse_where_clause(sv_node_t *where_clause) {
    if (where_clause == NULL) {
        return {};
    }
    std::vector<condition_t> conds;
    for (sv_node_t *node = where_clause; node != NULL; node = node->list.next) {
        assert(node->kind == N_LIST);
        sv_node_t *curr = node->list.curr;
        assert(curr->kind == N_BINARY_EXPR);
        sv_node_t *lhs = curr->binary_expr.left;
        assert(lhs->kind == N_COL);
        sv_node_t *rhs = curr->binary_expr.right;

        condition_t cond;
        cond.lhs_attr = {
                .table = lhs->col.tbname,
                .field = lhs->col.colname,
        };
        cond.op = interp_sv_comp_op(curr->binary_expr.op);
        if (rhs->kind == N_COL) {
            cond.is_rhs_attr = true;
            cond.rhs_attr = {.table = rhs->col.tbname, .field = rhs->col.colname};
        } else {
            cond.is_rhs_attr = false;
            cond.rhs_value.is_null = false;
            if (rhs->kind == N_SQLNULL) {
                cond.rhs_value.is_null = true;
            } else if (rhs->kind == N_VALUE_INT) {
                cond.rhs_value.data = (buffer_t) &rhs->value_int;
                cond.rhs_value.type = ATTR_INT;
            } else if (rhs->kind == N_VALUE_FLOAT) {
                cond.rhs_value.data = (buffer_t) &rhs->value_float;
                cond.rhs_value.type = ATTR_FLOAT;
            } else if (rhs->kind == N_VALUE_STRING) {
                cond.rhs_value.data = (buffer_t) rhs->value_string;
                cond.rhs_value.type = ATTR_STRING;
            } else {
                assert(0);
            }
        }
        conds.push_back(cond);
    }
    return conds;
}

value_t parse_value(sv_node_t *sv_val) {
    value_t val;
    switch (sv_val->kind) {
        case N_SQLNULL:
            val.is_null = true;
            break;
        case N_VALUE_INT:
            val.type = ATTR_INT;
            val.data = (buffer_t) &sv_val->value_int;
            val.is_null = false;
            break;
        case N_VALUE_FLOAT:
            val.type = ATTR_FLOAT;
            val.data = (buffer_t) &sv_val->value_float;
            val.is_null = false;
            break;
        case N_VALUE_STRING:
            val.type = ATTR_STRING;
            val.data = (buffer_t) sv_val->value_string;
            val.is_null = false;
            break;
        default:
            assert(0);
    }
    return val;
}

RC exec_sql(sv_node_t *root) {
    RC rc = 0;
    switch (root->kind) {
        case N_SHOW_DATABASES:
            rc = sm_show_databases();
            break;
        case N_CREATE_DATABASE:
            rc = sm_create_database(root->create_database.dbname);
            break;
        case N_DROP_DATABASE:
            rc = sm_drop_database(root->drop_database.dbname);
            break;
        case N_USE_DATABASE:
            rc = sm_use_database(root->use_database.dbname);
            break;
        case N_SHOW_TABLES:
            rc = sm_show_tables();
            break;
        case N_CREATE_TABLE: {
            const char *tbname = root->create_table.tbname;
            field_info_t fields[SM_MAX_FIELDS];
            int num_fields = 0;
            // Build columns
            for (sv_node_t *node = root->create_table.attrlist; node != NULL; node = node->list.next) {
                assert(node->kind == N_LIST);
                sv_node_t *curr = node->list.curr;
                switch (curr->kind) {
                    case N_COLUMN_DEF: {
                        field_info_t *field = &fields[num_fields++];
                        field->name = curr->column_def.colname;
                        field->type = interp_sv_type(curr->column_def.type_len.type);
                        field->len = curr->column_def.type_len.len;
                        field->nullable = curr->column_def.nullable;
                        field->is_primary = false;
                        field->has_frgn_key = false;
                        break;
                    }
                    case N_FOREIGN_KEY:
                    case N_PRIMARY_KEY:
                        break;
                    default:
                        assert(0);
                }
            }
            // Build primary keys
            bool is_prim_solved = false;
            for (sv_node_t *node = root->create_table.attrlist; node != NULL; node = node->list.next) {
                sv_node_t *curr = node->list.curr;
                if (curr->kind == N_PRIMARY_KEY) {
                    if (is_prim_solved) {
                        return QL_MULTIPLE_PRIMARY_KEY_DEFINED;
                    }
                    is_prim_solved = true;
                    for (sv_node_t *name_node = curr->primary_key.colnamelist;
                         name_node != NULL; name_node = name_node->list.next) {
                        assert(name_node->kind == N_LIST);
                        sv_node_t *name_curr = name_node->list.curr;
                        assert(name_curr->kind == N_IDENTIFIER);
                        // Find the column idx of primary key
                        int prim_idx = 0;
                        while (prim_idx < num_fields) {
                            if (strcmp(fields[prim_idx].name, name_curr->identifier) == 0) {
                                break;
                            }
                            prim_idx++;
                        }
                        if (prim_idx == num_fields) { return QL_PRIMARY_KEY_NOT_FOUND; }
                        fields[prim_idx].is_primary = true;
                    }
                }
            }
            bool is_foreign_solved = false;
            for (sv_node_t *node = root->create_table.attrlist; node != NULL; node = node->list.next) {
                sv_node_t *curr = node->list.curr;
                if (curr->kind == N_FOREIGN_KEY) {
                    if (is_foreign_solved) { return QL_MULTIPLE_FOREIGN_KEY_DEFINED; }
                    is_foreign_solved = true;
                    sv_node_t *local_col_node = curr->foreign_key.local_colnames;
                    sv_node_t *frgn_col_node = curr->foreign_key.colnames;
                    while (local_col_node != NULL && frgn_col_node != NULL) {
                        assert(frgn_col_node->kind == N_LIST);
                        sv_node_t *frgn_col_curr = frgn_col_node->list.curr;
                        assert(frgn_col_curr->kind == N_IDENTIFIER);

                        assert(local_col_node->kind == N_LIST);
                        sv_node_t *local_col_curr = local_col_node->list.curr;
                        assert(local_col_curr->kind == N_IDENTIFIER);
                        // Check local key
                        int local_idx = 0;
                        while (local_idx < num_fields) {
                            if (strcmp(fields[local_idx].name, local_col_curr->identifier) == 0) {
                                break;
                            }
                            local_idx++;
                        }
                        if (local_idx == num_fields) { return QL_LOCAL_FIELD_NOT_FOUND; }
                        // Check foreign key
                        int frgn_tbidx = sm_find_table(curr->foreign_key.tbname);
                        if (frgn_tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
                        auto &frgn_table = db_meta.tables[frgn_tbidx];
                        int frgn_colidx = sm_find_field(frgn_tbidx, frgn_col_curr->identifier);
                        if (frgn_colidx == frgn_table.num_fields) { return QL_FOREIGN_FIELD_NOT_FOUND; }
                        if (!frgn_table.fields[frgn_colidx].is_primary) { return QL_FOREIGN_KEY_NOT_PRIMARY; }

                        fields[local_idx].has_frgn_key = true;
                        fields[local_idx].frgn_tab = curr->foreign_key.tbname;
                        fields[local_idx].frgn_col = frgn_col_curr->identifier;

                        local_col_node = local_col_node->list.next;
                        frgn_col_node = frgn_col_node->list.next;
                    }
                    if (!(local_col_node == NULL && frgn_col_node == NULL)) {
                        return QL_INVALID_FOREIGN_KEY_COUNT;
                    }
                }
            }
            rc = sm_create_table(tbname, num_fields, fields);
            break;
        }
        case N_DESC_TABLE:
            rc = sm_desc_table(root->desc_table.tbname);
            break;
        case N_DROP_TABLE:
            rc = sm_drop_table(root->drop_table.tbname);
            break;
        case N_INSERT: {
            char *tbname = root->insert.tbname;
            for (sv_node_t *val_list = root->insert.value_lists; val_list != NULL; val_list = val_list->list.next) {
                std::vector<value_t> values;
                assert(val_list->kind == N_LIST);
                for (sv_node_t *val = val_list->list.curr; val != NULL; val = val->list.next) {
                    assert(val->kind == N_LIST);
                    values.push_back(parse_value(val->list.curr));
                }
                rc = ql_insert(tbname, values);
                if (rc) { return rc; }
            }
            break;
        }
        case N_SELECT_FROM: {
            std::vector<table_field_t> sel_attrs;
            std::vector<std::string> tables;
            // Parse selector
            for (sv_node_t *node = root->select_from.col_list; node != NULL; node = node->list.next) {
                assert(node->kind == N_LIST);
                sv_node_t *curr = node->list.curr;
                assert(curr->kind == N_COL);
                table_field_t relattr = {
                        .table = curr->col.tbname,
                        .field = curr->col.colname,
                };
                sel_attrs.push_back(relattr);
            }
            // Parse tables
            for (sv_node_t *node = root->select_from.tb_list; node != NULL; node = node->list.next) {
                assert(node->kind == N_LIST);
                sv_node_t *curr = node->list.curr;
                assert(curr->kind == N_IDENTIFIER);
                tables.emplace_back(curr->identifier);
            }
            // where clause
            auto conds = parse_where_clause(root->select_from.where_clause);
            rc = ql_select(sel_attrs, tables, conds);
            break;
        }
        case N_DELETE: {
            auto conds = parse_where_clause(root->delete_from.where_clause);
            rc = ql_delete(root->delete_from.tbname, conds);
            break;
        }
        case N_UPDATE: {
            char *tabname = root->update_set.tbname;
            // set clause
            std::vector<table_field_t> set_fields;
            std::vector<value_t> rhs_values;
            for (sv_node_t *node = root->update_set.set_clauses; node != NULL; node = node->list.next) {
                assert(node->kind == N_LIST);
                sv_node_t *set_clause = node->list.curr;
                assert(set_clause->kind == N_SET_CLAUSE);
                table_field_t set_field = {.table = tabname, .field = set_clause->set_clause.colname};
                set_fields.push_back(set_field);
                value_t rhs_value = parse_value(set_clause->set_clause.value);
                rhs_values.push_back(rhs_value);
            }
            // where clause
            auto conds = parse_where_clause(root->update_set.where_clause);
            rc = ql_update(tabname, set_fields, rhs_values, conds);
            break;
        }
        case N_CREATE_INDEX: {
            char *tbname = root->create_index.tbname;
            std::vector<std::string> colnames;
            for (sv_node_t *node = root->create_index.colnames; node != NULL; node = node->list.next) {
                assert(node->kind == N_LIST);
                sv_node_t *curr = node->list.curr;
                assert(curr->kind == N_IDENTIFIER);
                colnames.emplace_back(curr->identifier);
            }
            for (auto &colname: colnames) {
                rc = sm_create_index(tbname, colname.c_str());
                if (rc) { return rc; }
            }
            break;
        }
        case N_DROP_INDEX:
            // ignore, we don't drop index
            break;
        case N_ALTER_ADD_FIELD:
            if (root->add_key.field->kind == N_COLUMN_DEF) {
                sv_node_t *field = root->add_key.field;
                rc = sm_add_col(root->add_key.tbname,
                                field->column_def.colname,
                                field->column_def.nullable,
                                interp_sv_type(field->column_def.type_len.type),
                                field->column_def.type_len.len);
            } else {
//                assert(0);
            }

//            assert(root->add_key.field->kind == N_COL);

//            for (sv_node_t *node = root->add_primary_key.colnames; node != NULL; node = node->list.next) {
//            rc = sm_add_col(root->add_key);
            break;
        case N_ALTER_DROP_COL:
            rc = sm_drop_col(root->drop_col.tbname, root->drop_col.colname);
            break;
        case N_ALTER_ADD_PRIMARY_KEY: {
            char *tbname = root->add_primary_key.tbname;
            std::vector<std::string> colnames;
            for (sv_node_t *node = root->add_primary_key.colnames; node != NULL; node = node->list.next) {
                assert(node->kind == N_LIST);
                sv_node_t *curr = node->list.curr;
                assert(curr->kind == N_IDENTIFIER);
                colnames.emplace_back(curr->identifier);
            }
            rc = sm_add_primary_key(tbname, colnames);
            break;
        }
        case N_ALTER_DROP_PRIMARY_KEY: {
            char *tbname = root->drop_primary_key.tbname;
            rc = sm_drop_primary_key(tbname);
            break;
        }
        case N_ALTER_RENAME_TABLE:
            rc = sm_rename_table(root->rename_table.tbname, root->rename_table.new_tbname);
            break;
        case N_ALTER_CHANGE_COL:
            break;
        case N_ALTER_ADD_FOREIGN_KEY: {
            char *tbname = root->add_foreign_key.tbname;
            std::vector<std::string> colnames, ref_colnames;
            char *ref_tbname = root->add_foreign_key.ref_tbname;
            char *fkname = root->add_foreign_key.fkname;
            for (sv_node_t *node = root->add_foreign_key.colnames; node != NULL; node = node->list.next) {
                assert(node->kind == N_LIST);
                sv_node_t *curr = node->list.curr;
                assert(curr->kind == N_IDENTIFIER);
                colnames.emplace_back(curr->identifier);
            }
            for (sv_node_t *node = root->add_foreign_key.ref_colnames; node != NULL; node = node->list.next) {
                assert(node->kind == N_LIST);
                sv_node_t *curr = node->list.curr;
                assert(curr->kind == N_IDENTIFIER);
                ref_colnames.emplace_back(curr->identifier);
            }
            rc = sm_add_foreign_key(tbname, fkname, colnames, ref_tbname, ref_colnames);
            break;
        }
        case N_ALTER_DROP_FOREIGN_KEY:
            rc = sm_drop_foreign_key(root->drop_foreign_key.tbname, root->drop_foreign_key.fkname);
            break;
        default:
            assert(0);
    }
    return rc;
}