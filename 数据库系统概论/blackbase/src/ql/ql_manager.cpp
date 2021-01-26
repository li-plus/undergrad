#include "ql_manager.h"
#include "ql_error.h"
#include "sm/sm.h"
#include "ix/ix_compare.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>


// First byte indicates if this value is non-null
// Format: [ non-null(1) | data ]
static RC make_value(value_t *val, int field_len) {
    buffer_t val_buf = (buffer_t) malloc(field_len + 1);
    if (val->is_null) {
        memset(val_buf, 0, field_len + 1);
    } else {
        val_buf[0] = 1;
        if (val->type == ATTR_STRING) {
            int str_len = strlen((char *) val->data);
            if (str_len > field_len) { return QL_VALUE_STRING_TOO_LONG; }
            memset(val_buf + 1, 0, field_len);
            memcpy(val_buf + 1, val->data, str_len);
        } else {
            memcpy(val_buf + 1, val->data, field_len);
        }
    }
    val->data = val_buf;
    return 0;
}

static void free_value(value_t *val) {
    free(val->data);
}

bool eval_field(input_buffer_t lhs, attr_type_t lhs_type, comp_op_t op,
                input_buffer_t rhs, attr_type_t rhs_type, int len) {
    // convert to common type
    assert(lhs_type == rhs_type);
    attr_type_t type = lhs_type;
    if (rhs[0] == 0) {
        // rhs is null. condition is "is null" or "is not null".
        if (op == OP_EQ) {
            return lhs[0] == 0;
        } else if (op == OP_NE) {
            return lhs[0] == 1;
        } else {
            assert(0);
        }
    } else {
        // rhs is not null.
        if (lhs[0] == 0) {
            // if lhs is null, then cannot satisfy condition
            return false;
        }
    }
    assert(lhs[0] == 1 && rhs[0] == 1);
    if (op == OP_EQ) {
        return ix_compare(lhs, rhs, type, len) == 0;
    } else if (op == OP_LT) {
        return ix_compare(lhs, rhs, type, len) < 0;
    } else if (op == OP_GT) {
        return ix_compare(lhs, rhs, type, len) > 0;
    } else if (op == OP_LE) {
        return ix_compare(lhs, rhs, type, len) <= 0;
    } else if (op == OP_GE) {
        return ix_compare(lhs, rhs, type, len) >= 0;
    } else if (op == OP_NE) {
        return ix_compare(lhs, rhs, type, len) != 0;
    } else {
        assert(0);
    }
}

bool eval_cond(int tbidx, rm_record_t *record, const condition_t &cond) {
    assert(tbidx < db_meta.num_tables);
    auto &table = db_meta.tables[tbidx];
    int lhs_idx = sm_find_field(tbidx, cond.lhs_attr.field);
    assert(lhs_idx < table.num_fields);
    field_meta_t lhs_field = table.fields[lhs_idx];
    buffer_t lhs = record->data + lhs_field.offset;
    buffer_t rhs;
    attr_type_t rhs_type;
    if (cond.is_rhs_attr) {
        // rhs is attr
        int rhs_idx = sm_find_field(tbidx, cond.rhs_attr.field);
        assert(rhs_idx < table.num_fields);
        field_meta_t rhs_field = table.fields[rhs_idx];
        rhs = record->data + rhs_field.offset;
        rhs_type = rhs_field.type;
        assert(rhs_field.type == lhs_field.type);
    } else {
        // rhs is value
        rhs_type = cond.rhs_value.type;
        rhs = cond.rhs_value.data;
    }
    assert(rhs_type == lhs_field.type);
    bool ret = eval_field(lhs, lhs_field.type, cond.op, rhs, rhs_type, lhs_field.len);
    return ret;
}

bool eval_conds(int tbidx, rm_record_t *record, const std::vector<condition_t> &conds) {
    bool result = true;
    for (auto &cond: conds) {
        result &= eval_cond(tbidx, record, cond);
        if (!result) { return false; }
    }
    return true;
}

static int find_field(const table_field_t &tab_col, const std::vector<field_meta_t> &fields) {
    field_meta_t lhs_field;
    for (int i = 0; i < fields.size(); i++) {
        auto &field = fields[i];
        if (strcmp(field.tbname, tab_col.table) == 0 && strcmp(field.name, tab_col.field) == 0) {
            lhs_field = field;
            return i;
        }
    }
    return fields.size();
}

bool eval_multi_tab_cond(buffer_t buf, const std::vector<field_meta_t> &fields, const condition_t &cond) {
    int lhs_idx = find_field(cond.lhs_attr, fields);
    assert(lhs_idx < fields.size());
    field_meta_t lhs_field = fields[lhs_idx];
    buffer_t lhs = buf + lhs_field.offset;
    buffer_t rhs;
    attr_type_t rhs_type;
    if (cond.is_rhs_attr) {
        // rhs is attr
        int rhs_idx = find_field(cond.rhs_attr, fields);
        assert(rhs_idx < fields.size());
        field_meta_t rhs_field = fields[rhs_idx];
        rhs = buf + rhs_field.offset;
        rhs_type = rhs_field.type;
        if (*(int *) (rhs + 1) == 3691 && *(int *) (lhs + 1) == 1) {
            int cc = 0;
        }
        assert(rhs_field.type == lhs_field.type);
    } else {
        // rhs is value
        rhs_type = cond.rhs_value.type;
        rhs = cond.rhs_value.data;
    }
    assert(rhs_type == lhs_field.type);
    bool ret = eval_field(lhs, lhs_field.type, cond.op, rhs, rhs_type, lhs_field.len);
    return ret;
}

bool eval_multi_tab_conds(buffer_t buf,
                          const std::vector<field_meta_t> &fields,
                          const std::vector<condition_t> &conds) {
    for (auto &cond: conds) {
        if (!eval_multi_tab_cond(buf, fields, cond)) {
            return false;
        }
    }
    return true;
}

class ql_node_t {
public:
    int _size;
    buffer_t _buf;
    std::vector<field_meta_t> _fields;

    virtual ~ql_node_t() {}

    virtual RC next_record() = 0;

    virtual bool is_end() = 0;

    virtual void reset() = 0;
};

class ql_node_join_t : public ql_node_t {
public:
    ql_node_t *_left;
    ql_node_t *_right;

    ql_node_join_t(ql_node_t *left, ql_node_t *right) : _left(left), _right(right) {
        _size = left->_size + right->_size;
        _buf = new uint8_t[_size];
        _fields = left->_fields;
        for (auto r_field: right->_fields) {
            r_field.offset += _left->_size;
            _fields.push_back(r_field);
        }
        if (_right->is_end()) {
            while (!_left->is_end()) {
                _left->next_record();
            }
        }
        if (!is_end()) {
            memcpy(_buf, _left->_buf, _left->_size);
            memcpy(_buf + _left->_size, _right->_buf, _right->_size);
        }
    }

    RC next_record() override {
//        RC rc;
        if (is_end()) { return QL_NO_MORE_RECORD; }
        _right->next_record();
        if (_right->is_end()) {
            _right->reset();
            _left->next_record();
        }
        if (!is_end()) {
            memcpy(_buf, _left->_buf, _left->_size);
            memcpy(_buf + _left->_size, _right->_buf, _right->_size);
        }
        return 0;
    }

    bool is_end() override {
        return _left->is_end();
    }

    void reset() override {
        _left->reset();
        _right->reset();
        memcpy(_buf, _left->_buf, _left->_size);
        memcpy(_buf + _left->_size, _right->_buf, _right->_size);
    }
};

class ql_node_proj_t : public ql_node_t {
public:
    ql_node_t *_node;
    std::vector<int> _colidx;
    std::vector<condition_t> _conds;

    ql_node_proj_t(ql_node_t *node,
                   const std::vector<table_field_t> &sel_fields,
                   const std::vector<condition_t> &conds) : _node(node) {
        // Build selected column indices
        _conds = conds;
        int curr_offset = 0;
        for (auto &sel_field: sel_fields) {
            int i;
            for (i = 0; i < _node->_fields.size(); i++) {
                if (strcmp(_node->_fields[i].tbname, sel_field.table) == 0 &&
                    strcmp(_node->_fields[i].name, sel_field.field) == 0) {
                    break;
                }
            }
            assert(i < node->_fields.size());
            _colidx.push_back(i);
            auto field = _node->_fields[i];
            field.offset = curr_offset;
            _fields.push_back(field);
            curr_offset += field.len + 1;
        }
        _size = curr_offset;
        _buf = new uint8_t[_size];
        assert(_colidx.size() == sel_fields.size());
        // TODO: use pure next like python, raise an exception when stopping iteration
        while (!is_end()) {
            if (!_node->is_end()) {
                if (eval_multi_tab_conds(_node->_buf, _node->_fields, _conds)) {
                    // projection
                    for (int i = 0; i < _colidx.size(); i++) {
                        int col = _colidx[i];
                        int offset = _fields[i].offset;
                        memcpy(_buf + offset, _node->_buf + _node->_fields[col].offset, _node->_fields[col].len + 1);
                    }
                    break;
                }
                _node->next_record();
            }
        }
    }

    RC next_record() override {
        if (is_end()) { return QL_NO_MORE_RECORD; }
        while (!is_end()) {
            _node->next_record();
            if (!_node->is_end()) {
                if (eval_multi_tab_conds(_node->_buf, _node->_fields, _conds)) {
                    // projection
                    for (int i = 0; i < _colidx.size(); i++) {
                        int col = _colidx[i];
                        int offset = _fields[i].offset;
                        memcpy(_buf + offset, _node->_buf + _node->_fields[col].offset, _node->_fields[col].len + 1);
                    }
                    return 0;
                }
            }
        }
        return 0;
    }

    bool is_end() override {
        return _node->is_end();
    }

    void reset() override {
        assert(0);
        _node->reset();
    }
};

class ql_node_table_t : public ql_node_t {
    table_scan_t *_scan;
    rm_record_t _rec;
    int _tbidx;
    std::vector<condition_t> _conds;
public:
    ql_node_table_t(int tbidx, const std::vector<condition_t> &conds) {
        _scan = new table_scan_t(tbidx, conds, &_rec);
        _size = _rec.size;
        _tbidx = tbidx;
        _conds = conds;
        auto &tbmeta = db_meta.tables[tbidx];
        for (int i = 0; i < tbmeta.num_fields; i++) {
            _fields.push_back(tbmeta.fields[i]);
        }
        _buf = new uint8_t[_size];
        memcpy(_buf, _rec.data, _size);
    }

    ~ql_node_table_t() override {
        _scan->close_scan(&_rec);
        delete _scan;
    }

    RC next_record() override {
        if (is_end()) { return QL_NO_MORE_RECORD; }
        _scan->get_next_record(&_rec);
        memcpy(_buf, _rec.data, _size);
        return 0;
    }

    bool is_end() override {
        return _scan->is_end(&_rec);
    }

    void reset() override {
        delete _scan;
        _scan = new table_scan_t(_tbidx, _conds, &_rec);
        memcpy(_buf, _rec.data, _size);
    }
};

void print_record(const rm_record_t &record, const std::vector<field_meta_t> &fields) {
    printf("|");
    for (const auto &field : fields) {
        if (record.data[field.offset] == 0) {
            printf(" %10s |", "NULL");
        } else {
            buffer_t field_buf = record.data + field.offset + 1;
            if (field.type == ATTR_INT) {
                printf(" %10d |", *(int *) field_buf);
            } else if (field.type == ATTR_FLOAT) {
                printf(" %10.2f |", *(float *) field_buf);
            } else if (field.type == ATTR_STRING) {
                printf(" %10.*s |", field.len < 10 ? field.len : 10, (char *) field_buf);
            }
        }
    }
    printf("\n");
}

static RC make_cond(int tbidx, condition_t &cond) {
    RC rc;
    assert(tbidx < db_meta.num_tables);
    auto &tbmeta = db_meta.tables[tbidx];
    int field_idx = sm_find_field(tbidx, cond.lhs_attr.field);
    if (field_idx == tbmeta.num_fields) { return SM_FIELD_NOT_FOUND; }
    auto &field_meta = tbmeta.fields[field_idx];
    if (!cond.is_rhs_attr) {
        rc = make_value(&cond.rhs_value, field_meta.len);
        if (rc) { return rc; }
    }
    return 0;
}

static RC make_where_clause(int tbidx, std::vector<condition_t> &conditions) {
    RC rc;
    for (auto &cond : conditions) {
        rc = make_cond(tbidx, cond);
        if (rc) { return rc; }
    }
    return 0;
}

static void free_where_clause(std::vector<condition_t> &conditions) {
    for (auto &cond: conditions) {
        if (!cond.is_rhs_attr) {
            free_value(&cond.rhs_value);
        }
    }
}

static RC check_where_clause(int tbidx, const std::vector<condition_t> &conds) {
    assert(tbidx < db_meta.num_tables);
    auto &table = db_meta.tables[tbidx];
    for (auto &cond: conds) {
        int field_idx = sm_find_field(tbidx, cond.lhs_attr.field);
        if (field_idx == table.num_fields) { return SM_FIELD_NOT_FOUND; }
        auto &field = table.fields[field_idx];
        if (!cond.is_rhs_attr && !cond.rhs_value.is_null && cond.rhs_value.type != field.type) {
            return QL_INCOMPATIBLE_TYPE;
        }
    }
    return 0;
}

RC make_tab_name(table_field_t &tab_col, const std::vector<std::string> &tables) {
    // If table is not specified, infer from field
    bool is_solved = false;
    for (auto &tb: tables) {
        int tab_idx = sm_find_table(tb.c_str());
        if (tab_idx == db_meta.num_tables) { continue; }
        int col_idx = sm_find_field(tab_idx, tab_col.field);
        if (col_idx == db_meta.tables[tab_idx].num_fields) { continue; }
        if (is_solved) { return QL_AMBIGUOUS_FIELD; }
        is_solved = true;
        tab_col.table = (char *) malloc(strlen(db_meta.tables[tab_idx].name));
        strcpy(tab_col.table, db_meta.tables[tab_idx].name);
    }
    if (!is_solved) { return SM_FIELD_NOT_FOUND; }
    return 0;
}

RC ql_select(std::vector<table_field_t> &sel_fields,    // selectors
             const std::vector<std::string> &tables,    // tables after FROM clause
             const std::vector<condition_t> &conditions_) {
    auto conditions = conditions_;
    RC rc;
    // Infer table name in where clause
    for (auto &cond: conditions) {
        if (strlen(cond.lhs_attr.table) == 0) {
            rc = make_tab_name(cond.lhs_attr, tables);
            if (rc) { return rc; }
        }
        if (cond.is_rhs_attr && strlen(cond.rhs_attr.table) == 0) {
            rc = make_tab_name(cond.rhs_attr, tables);
            if (rc) { return rc; }
        }
        // make lhs
        int tbidx = sm_find_table(cond.lhs_attr.table);
        rc = make_cond(tbidx, cond);
        if (rc) { return rc; }
        // Check where clause
        rc = check_where_clause(tbidx, {cond});
        if (rc) { return rc; }
    }
    // Prepare to scan table
    if (sel_fields.size() == 1 && strcmp(sel_fields[0].field, "*") == 0) {
        // Expand to all fields
        sel_fields.clear();
        for (auto &tb: tables) {
            int tab_idx = sm_find_table(tb.c_str());
            if (tab_idx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
            for (int i = 0; i < db_meta.tables[tab_idx].num_fields; i++) {
                table_field_t tab_col = {
                        .table = db_meta.tables[tab_idx].name,
                        .field = db_meta.tables[tab_idx].fields[i].name
                };
                sel_fields.push_back(tab_col);
            }
        }
    } else {
        // Infer columns in selector
        for (auto &sel_field: sel_fields) {
            if (strlen(sel_field.table) == 0) {
                // If table is not specified, infer from field
                bool is_solved = false;
                for (auto &tb: tables) {
                    int tab_idx = sm_find_table(tb.c_str());
                    if (tab_idx == db_meta.num_tables) { continue; }
                    int col_idx = sm_find_field(tab_idx, sel_field.field);
                    if (col_idx == db_meta.tables[tab_idx].num_fields) { continue; }
                    if (is_solved) { return QL_AMBIGUOUS_FIELD; }
                    is_solved = true;
                    sel_field.table = (char *) malloc(strlen(db_meta.tables[tab_idx].name));
                    strcpy(sel_field.table, db_meta.tables[tab_idx].name);
                }
                if (!is_solved) { return SM_FIELD_NOT_FOUND; }
            } else {
                int tab_idx = sm_find_table(sel_field.table);
                if (tab_idx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
                int col_idx = sm_find_field(tab_idx, sel_field.field);
                if (col_idx == db_meta.tables[tab_idx].num_fields) { return SM_FIELD_NOT_FOUND; }
            }
        }
    }
    ql_node_t *ql_node = nullptr;
    for (int i = tables.size() - 1; i > -1; i--) {
        int tbidx = sm_find_table(tables[i].c_str());
        if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
        std::vector<condition_t> tbconds;
        for (auto &cond: conditions) {
            if (strcmp(cond.lhs_attr.table, db_meta.tables[tbidx].name) == 0 && !cond.is_rhs_attr) {
                tbconds.push_back(cond);
            }
        }
        if (ql_node == nullptr) {
            ql_node = new ql_node_table_t(tbidx, tbconds);
        } else {
            ql_node_t *left_tbnode = new ql_node_table_t(tbidx, tbconds);
            ql_node_t *join_node = new ql_node_join_t(left_tbnode, ql_node);
            ql_node = join_node;
        }
    }
    ql_node = new ql_node_proj_t(ql_node, sel_fields, conditions);
    // print header
    putchar('+');
    for (int i = 0; i < sel_fields.size() * 13; i++) {
        if ((i + 1) % 13 == 0) {
            putchar('+');
        } else {
            putchar('-');
        }
    }
    printf("\n");
    printf("|");
    for (auto &sel_field : sel_fields) {
        char tab_col[256];
        sprintf(tab_col, "%s.%s", sel_field.table, sel_field.field);
        printf(" %10s |", tab_col);
    }
    printf("\n");
    putchar('+');
    for (int i = 0; i < sel_fields.size() * 13; i++) {
        if ((i + 1) % 13 == 0) {
            putchar('+');
        } else {
            putchar('-');
        }
    }
    printf("\n");
    // print records
    int cnt_rec = 0;
    rm_record_t rec;
    rec.size = ql_node->_size;
    rec.data = new uint8_t[ql_node->_size];
    while (!ql_node->is_end()) {
        memcpy(rec.data, ql_node->_buf, ql_node->_size);
        print_record(rec, ql_node->_fields);
        cnt_rec++;
        rc = ql_node->next_record();
        if (rc) { return rc; }
    }
    // print footer
    putchar('+');
    for (int i = 0; i < sel_fields.size() * 13; i++) {
        if ((i + 1) % 13 == 0) {
            putchar('+');
        } else {
            putchar('-');
        }
    }
    printf("\n");
    printf("Got %d record(s)\n", cnt_rec);
//    table_scan.close_scan(&rec);
    // project TODO
    // Clean up values in where clause
    free_where_clause(conditions);
    return 0;
}

std::vector<std::pair<int, std::vector<int>>> get_refs(int dst_tbidx) {
    // tb1, col11, col12, ...
    // tb2, col21, col22, ...
    std::vector<std::pair<int, std::vector<int>>> frgn_refs;
    assert(dst_tbidx < db_meta.num_tables);
    for (int src_tbidx = 0; src_tbidx < db_meta.num_tables; src_tbidx++) {
        if (src_tbidx != dst_tbidx) {
            auto &src_tb = db_meta.tables[src_tbidx];
            bool has_frgn_key = false;
            std::vector<int> frgn_keys;
            for (int src_colidx = 0; src_colidx < src_tb.num_fields; src_colidx++) {
                auto &src_field = src_tb.fields[src_colidx];
                if (src_field.has_frgn_key && sm_find_table(src_field.frgn_tab) == dst_tbidx) {
                    has_frgn_key = true;
                    frgn_keys.push_back(src_colidx);
                }
            }
            if (has_frgn_key) {
                frgn_refs.emplace_back(src_tbidx, frgn_keys);
            }
        }
    }
    return frgn_refs;
}

namespace std {
    template<>
    struct std::hash<rid_t> {
        size_t operator()(const rid_t &rid) const {
            return (rid.page_no << 16) | (rid.slot_no);
        }
    };

    template<>
    struct std::equal_to<rid_t> {
        bool operator()(const rid_t &x, const rid_t &y) const {
            return x.page_no == y.page_no && x.slot_no == y.slot_no;
        }
    };
}

std::vector<rid_t> intersect(const std::vector<rid_t> &a, const std::vector<rid_t> &b) {
    std::vector<rid_t> inter;
    std::unordered_set<rid_t> set;
    for (auto &rid: a) {
        set.insert(rid);
    }
    for (auto &rid: b) {
        if (set.count(rid)) {
            inter.push_back(rid);
        }
    }
    return inter;
}

RC ql_insert(const std::string &tbname, const std::vector<value_t> &values_) {
    RC rc;
    auto values = values_;  // TODO: not const input
    int tbidx = sm_find_table(tbname.c_str());
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    auto &tbmeta = db_meta.tables[tbidx];
    if (values.size() != tbmeta.num_fields) { return QL_INVALID_VALUE_COUNT; }
    assert(values.size() == tbmeta.num_fields);
    // Get record file handle
    rm_file_handle_t *rmfh = &sm_fhs[tbidx];
    // Pad values
    for (int i = 0; i < values.size(); i++) {
        auto &field_meta = tbmeta.fields[i];
        rc = make_value(&values[i], field_meta.len);
        if (rc) { return rc; }
    }
    // Check constraints
    bool has_primary = false;
    std::vector<rid_t> inter_rids;
    for (int i = 0; i < tbmeta.num_fields; i++) {
        // Primary key constraints (unique & not null)
        auto &field = tbmeta.fields[i];
        if (field.is_primary) {
            std::vector<rid_t> prim_rids;
            if (values[i].is_null) {
                return QL_EXPECTED_NOT_NULL;
            }
            ix_file_handle_t ih;
            rc = ix_open_index(tbname.c_str(), i, &ih);
            if (rc) { return rc; }
            iid_t lower, upper;
            rc = ix_lower_bound(&ih, values[i].data, &lower);
            if (rc) { return rc; }
            rc = ix_upper_bound(&ih, values[i].data, &upper);
            if (rc) { return rc; }
            while (!ix_scan_equal(&lower, &upper)) {
                rid_t rid;
                rc = ix_get_rid(&ih, &lower, &rid);
                if (rc) { return rc; }
                prim_rids.push_back(rid);
                rc = ix_scan_next(&ih, &lower);
                if (rc) { return rc; }
            }
            rc = ix_close_index(&ih);
            if (rc) { return rc; }
            if (!has_primary) {
                // init
                inter_rids = prim_rids;
            } else {
                inter_rids = intersect(inter_rids, prim_rids);
            }
            has_primary = true;
        }
        // Not null constraint
        if (!field.nullable && values[i].is_null) {
            return QL_EXPECTED_NOT_NULL;
        }
        // Type constraint
        if (!values[i].is_null && values[i].type != field.type) {
            return QL_INCOMPATIBLE_TYPE;
        }
    }
    if (has_primary && !inter_rids.empty()) {
        return QL_DUPLICATE_ENTRY;
    }
    // Check foreign key constraint
    std::vector<condition_t> frgn_conds;
    int frgn_tbidx = -1;
    for (int field_idx = 0; field_idx < tbmeta.num_fields; field_idx++) {
        auto &field_meta = tbmeta.fields[field_idx];
        if (field_meta.has_frgn_key) {
            if (frgn_tbidx == -1) {
                frgn_tbidx = sm_find_table(field_meta.frgn_tab);
                assert(frgn_tbidx < db_meta.num_tables);
            }
            condition_t cond;
            cond.lhs_attr = {.table = field_meta.frgn_tab, .field = field_meta.frgn_col};
            cond.op = OP_EQ;
            cond.is_rhs_attr = false;
            cond.rhs_value = values[field_idx];
            frgn_conds.push_back(cond);
        }
    }
    if (frgn_tbidx != -1) {
        rm_record_t frgn_rec;
        table_scan_t frgn_scan(frgn_tbidx, frgn_conds, &frgn_rec);
        if (frgn_scan.is_end(&frgn_rec)) {
            // Reference key not found, cannot insert.
            return QL_REF_KEY_NOT_FOUND;
        }
    }
    // Make record buffer
    rm_record_t rec;
    rm_record_init(&rec, rmfh);
    for (int i = 0; i < values.size(); i++) {
        auto &field_meta = tbmeta.fields[i];
        auto &val = values[i];
        int offset = field_meta.offset;
        memcpy(rec.data + offset, val.data, field_meta.len + 1);
    }
    // Insert into record file
    rc = rm_insert_record(rmfh, rec.data, &rec.rid);
    if (rc) { return rc; }
    // Insert into ix
    for (int field_idx = 0; field_idx < tbmeta.num_fields; field_idx++) {
        auto &field = tbmeta.fields[field_idx];
        if (field.index) {
            ix_file_handle_t ixfh;
            rc = ix_open_index(tbname.c_str(), field_idx, &ixfh);
            if (rc) { return rc; }
            rc = ix_insert_entry(&ixfh, rec.data + field.offset, &rec.rid);
            if (rc) { return rc; }
            rc = ix_close_index(&ixfh);
            if (rc) { return rc; }
        }
    }
    // Destroy record
    rm_record_destroy(&rec);
    // Free string value
    for (int i = 0; i < values.size(); i++) {
        free_value(&values[i]);
    }
    return 0;
}

RC ql_delete(const std::string &tbname, std::vector<condition_t> &conditions) {
    RC rc;
    int tbidx = sm_find_table(tbname.c_str());
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    // Pad values in where clause
    rc = make_where_clause(tbidx, conditions);
    if (rc) { return rc; }
    // Check where clause
    rc = check_where_clause(tbidx, conditions);
    if (rc) { return rc; }
    auto &tbmeta = db_meta.tables[tbidx];
    std::vector<rid_t> rids;
    // scan table
    rm_record_t rec;
    table_scan_t table_scan(tbidx, conditions, &rec);
    while (!table_scan.is_end(&rec)) {
        // remember rid
        rids.push_back(rec.rid);
        // scan next record
        rc = table_scan.get_next_record(&rec);
        if (rc) { return rc; }
    }
    table_scan.close_scan(&rec);
    // Get record file
    rm_file_handle_t *fh = &sm_fhs[tbidx];
    // Check foreign key constraint
    auto refs = get_refs(tbidx);
    rm_record_init(&rec, fh);
    for (auto &rid: rids) {
        rc = rm_get_record(fh, &rid, rec.data);
        if (rc) { return rc; }
        // Loop over each table that refers to the current one.
        for (auto &ref: refs) {
            std::vector<condition_t> conds;
            int ref_tbidx = ref.first;
            auto &ref_tbmeta = db_meta.tables[ref_tbidx];
            for (int ref_col: ref.second) {
                auto &ref_field = ref_tbmeta.fields[ref_col];
                assert(ref_field.has_frgn_key);
                // Get value in current table
                int field_idx = sm_find_field(tbidx, ref_field.frgn_col);
                assert(field_idx < tbmeta.num_fields);
                auto &field = tbmeta.fields[field_idx];
                buffer_t val_buf = rec.data + field.offset;
                value_t val = {
                        .type = field.type,
                        .data = val_buf,
                        .is_null = !val_buf[0],
                };
                // Find in table with foreign key that refers to this one
                condition_t cond;
                cond.lhs_attr = {.table = ref_tbmeta.name, .field = ref_field.name};
                cond.op = OP_EQ;
                cond.is_rhs_attr = false;
                cond.rhs_value = val;
                conds.push_back(cond);
            }
            table_scan_t ref_scan(ref_tbidx, conds, &rec);
            if (!ref_scan.is_end(&rec)) { return QL_REF_KEY_EXISTS; }
        }
    }
    rm_record_destroy(&rec);
    // Open all necessary index files
    ix_file_handle_t ihs[SM_MAX_FIELDS];
    for (int i = 0; i < tbmeta.num_fields; i++) {
        if (tbmeta.fields[i].index) {
            rc = ix_open_index(tbmeta.name, i, &ihs[i]);
            if (rc) { return rc; }
        }
    }
    // Delete each rid from record file and index file
    rm_record_init(&rec, fh);
    for (auto &rid: rids) {
        rc = rm_get_record(fh, &rid, rec.data);
        if (rc) { return rc; }
        for (int i = 0; i < tbmeta.num_fields; i++) {
            if (tbmeta.fields[i].index) {
                rc = ix_delete_entry(&ihs[i], rec.data + tbmeta.fields[i].offset, &rid);
                if (rc) { return rc; }
            }
        }
        rc = rm_delete_record(fh, &rid);
        if (rc) { return rc; }
    }
    rm_record_destroy(&rec);
    // Close all index files
    for (int i = 0; i < tbmeta.num_fields; i++) {
        if (tbmeta.fields[i].index) {
            rc = ix_close_index(&ihs[i]);
            if (rc) { return rc; }
        }
    }
    // Clean up values in where clause
    free_where_clause(conditions);
    return 0;
}

RC ql_update(const std::string &tbname,
             const std::vector<table_field_t> &set_fields,
             const std::vector<value_t> &rhs_values_,
             std::vector<condition_t> &conditions) {
    RC rc;
    assert(set_fields.size() == rhs_values_.size());
    auto rhs_values = rhs_values_;
    int tbidx = sm_find_table(tbname.c_str());
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    auto &tbmeta = db_meta.tables[tbidx];
    // Ensure all fields are valid
    std::vector<int> field_idxs;
    for (auto &set_field: set_fields) {
        int set_field_idx = sm_find_field(tbidx, set_field.field);
        if (set_field_idx == tbmeta.num_fields) { return SM_FIELD_NOT_FOUND; }
        field_idxs.push_back(set_field_idx);
    }
    assert(field_idxs.size() == set_fields.size());
    // Ensure all values are valid
    for (int i = 0; i < set_fields.size(); i++) {
        value_t val = rhs_values[i];
        int field_idx = field_idxs[i];
        if (!val.is_null && tbmeta.fields[field_idx].type != val.type) { return QL_INCOMPATIBLE_TYPE; }
    }
    // Pad values in set clause
    for (int i = 0; i < set_fields.size(); i++) {
        auto set_field_idx = field_idxs[i];
        auto &rhs_value = rhs_values[i];
        auto &field = tbmeta.fields[set_field_idx];
        rc = make_value(&rhs_value, field.len);
        if (rc) { return rc; }
    }
    // Pad values in where clause
    rc = make_where_clause(tbidx, conditions);
    if (rc) { return rc; }
    // Check where clause
    rc = check_where_clause(tbidx, conditions);
    if (rc) { return rc; }
    std::vector<rid_t> rids;
    // scan table
    rm_record_t rec;
    table_scan_t table_scan(tbidx, conditions, &rec);
    while (!table_scan.is_end(&rec)) {
        // remember rid
        rids.push_back(rec.rid);
        // scan next record
        rc = table_scan.get_next_record(&rec);
        if (rc) { return rc; }
    }
    table_scan.close_scan(&rec);
    // Get record file
    rm_file_handle_t *fh = &sm_fhs[tbidx];
    // Check foreign key constraint
    auto refs = get_refs(tbidx);
    rm_record_init(&rec, fh);
    for (auto &rid: rids) {
        rc = rm_get_record(fh, &rid, rec.data);
        if (rc) { return rc; }
        // Insert constraint
        std::vector<condition_t> frgn_conds;
        int frgn_tbidx = -1;
        for (int field_idx : field_idxs) {
            auto &field_meta = tbmeta.fields[field_idx];
            if (field_meta.has_frgn_key) {
                if (frgn_tbidx == -1) {
                    frgn_tbidx = sm_find_table(field_meta.frgn_tab);
                    assert(frgn_tbidx < db_meta.num_tables);
                }
                condition_t cond;
                cond.lhs_attr = {.table = field_meta.frgn_tab, .field = field_meta.frgn_col};
                cond.op = OP_EQ;
                cond.is_rhs_attr = false;
                cond.rhs_value = rhs_values[field_idx];
                frgn_conds.push_back(cond);
            }
        }
        if (frgn_tbidx != -1) {
            rm_record_t frgn_rec;
            table_scan_t frgn_scan(frgn_tbidx, frgn_conds, &frgn_rec);
            if (frgn_scan.is_end(&frgn_rec)) {
                // Reference key not found, cannot insert.
                return QL_REF_KEY_NOT_FOUND;
            }
        }
        // Delete constraint
        // Loop over each table that refers to the current one.
        for (auto &ref: refs) {
            std::vector<condition_t> conds;
            int ref_tbidx = ref.first;
            auto &ref_tbmeta = db_meta.tables[ref_tbidx];
            for (int ref_col: ref.second) {
                auto &ref_field = ref_tbmeta.fields[ref_col];
                assert(ref_field.has_frgn_key);
                // Get value in current table
                int field_idx = sm_find_field(tbidx, ref_field.frgn_col);
                assert(field_idx < tbmeta.num_fields);
                if (std::find(field_idxs.begin(), field_idxs.end(), field_idx) == field_idxs.end()) {
                    // Reference key not in set field, do nothing
                    continue;
                }
                auto &field = tbmeta.fields[field_idx];
                buffer_t val_buf = rec.data + field.offset;
                value_t val = {
                        .type = field.type,
                        .data = val_buf,
                        .is_null = !val_buf[0],
                };
                // Find in table with foreign key that refers to this one
                condition_t cond;
                cond.lhs_attr = {.table = ref_tbmeta.name, .field = ref_field.name};
                cond.op = OP_EQ;
                cond.is_rhs_attr = false;
                cond.rhs_value = val;
                conds.push_back(cond);
            }
            if (!conds.empty()) {
                table_scan_t ref_scan(ref_tbidx, conds, &rec);
                if (!ref_scan.is_end(&rec)) { return QL_REF_KEY_EXISTS; }
            }
        }
    }
    rm_record_destroy(&rec);
    // Open all necessary index files
    ix_file_handle_t ihs[SM_MAX_FIELDS];
    for (int i = 0; i < tbmeta.num_fields; i++) {
        if (tbmeta.fields[i].index) {
            rc = ix_open_index(tbmeta.name, i, &ihs[i]);
            if (rc) { return rc; }
        }
    }
    // Update each rid from record file and index file
    rm_record_init(&rec, fh);
    for (auto &rid: rids) {
        rc = rm_get_record(fh, &rid, rec.data);
        if (rc) { return rc; }
        // Remove old entry from index
        for (int i = 0; i < tbmeta.num_fields; i++) {
            if (tbmeta.fields[i].index) {
                rc = ix_delete_entry(&ihs[i], rec.data + tbmeta.fields[i].offset, &rid);
                if (rc) { return rc; }
            }
        }
        // Update record in place
        for (int i = 0; i < set_fields.size(); i++) {
            auto &rhs_value = rhs_values[i];
            auto set_field_idx = field_idxs[i];
            auto &field = tbmeta.fields[set_field_idx];
            memcpy(rec.data + field.offset, rhs_value.data, field.len + 1);
        }
        rc = rm_update_record(fh, &rid, rec.data);
        if (rc) { return rc; }
        // Insert new entry into index
        for (int i = 0; i < tbmeta.num_fields; i++) {
            if (tbmeta.fields[i].index) {
                rc = ix_insert_entry(&ihs[i], rec.data + tbmeta.fields[i].offset, &rid);
                if (rc) { return rc; }
            }
        }
    }
    rm_record_destroy(&rec);
    // Close all index files
    for (int i = 0; i < tbmeta.num_fields; i++) {
        if (tbmeta.fields[i].index) {
            rc = ix_close_index(&ihs[i]);
            if (rc) { return rc; }
        }
    }
    // Clean up values in where clause
    free_where_clause(conditions);
    // Free values in set clause
    for (auto &rhs_value: rhs_values) {
        free_value(&rhs_value);
    }
    return 0;
}
