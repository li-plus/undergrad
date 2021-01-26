#pragma once

#include "ql_defs.h"
#include "ix/ix.h"
#include "sm/sm_manager.h"
#include "sm/sm_defs.h"
#include "rm/rm.h"
#include <vector>

typedef struct {
    attr_type_t type;
    buffer_t data;
    bool is_null;
} value_t;

typedef enum {
    OP_EQ,
    OP_LT,
    OP_GT,
    OP_NE,
    OP_LE,
    OP_GE,
} comp_op_t;

typedef struct {
    char *table;
    char *field;
} table_field_t;

typedef struct {
    table_field_t lhs_attr;
    comp_op_t op;
    bool is_rhs_attr;
    union {
        table_field_t rhs_attr;
        value_t rhs_value;
    };
} condition_t;

RC ql_select(std::vector<table_field_t> &sel_fields,    // selectors
             const std::vector<std::string> &tables,    // tables after FROM clause
             const std::vector<condition_t> &conditions);

RC ql_insert(const std::string &tbname, const std::vector<value_t> &values);

RC ql_delete(const std::string &tbname, std::vector<condition_t> &conditions);

RC ql_update(const std::string &tbname,
             const std::vector<table_field_t> &set_fields,
             const std::vector<value_t> &rhs_values_,
             std::vector<condition_t> &conditions);


bool eval_conds(int tbidx, rm_record_t *record, const std::vector<condition_t> &cond);

class table_scan_t {
    std::vector<condition_t> _conds;
    ix_file_handle_t _ih;
    rm_file_handle_t *_fh;
    int _tbidx;
    int _index_no = -1;
    iid_t _lower;
    iid_t _upper;
public:
    table_scan_t(int tbidx, const std::vector<condition_t> &conds, rm_record_t *rec) {
        RC rc;
        assert(tbidx < db_meta.num_tables);
        _tbidx = tbidx;
        _conds = conds;
        table_meta_t &tbmeta = db_meta.tables[tbidx];
        _fh = &sm_fhs[tbidx];
        // conditions are all about this table
        _index_no = -1;
        for (auto &cond: conds) {
            if (!cond.is_rhs_attr && cond.op != OP_NE) {
                // If rhs is value and op is not "<>", find if lhs has index
                int field_idx = sm_find_field(tbidx, cond.lhs_attr.field);
                assert(field_idx < tbmeta.num_fields);
                field_meta_t field_meta = tbmeta.fields[field_idx];
                if (field_meta.index) {
                    // This field has index, use it
                    _index_no = field_idx;
                    break;
                }
            }
        }
        // Init record
        rm_record_init(rec, _fh);
        if (_index_no != -1) {
            // use idx
            // load index handle
            rc = ix_open_index(tbmeta.name, _index_no, &_ih);
            assert(rc == 0);
            ix_leaf_begin(&_ih, &_lower);
            rc = ix_leaf_end(&_ih, &_upper);
            assert(rc == 0);
            field_meta_t index_field = tbmeta.fields[_index_no];
            for (auto &cond: conds) {
                if (!cond.is_rhs_attr && cond.op != OP_NE && strcmp(cond.lhs_attr.field, index_field.name) == 0) {
                    if (cond.op == OP_EQ) {
                        rc = ix_lower_bound(&_ih, (input_buffer_t) cond.rhs_value.data, &_lower);
                        assert(rc == 0);
                        rc = ix_upper_bound(&_ih, (input_buffer_t) cond.rhs_value.data, &_upper);
                        assert(rc == 0);
                    } else if (cond.op == OP_LT) {
                        rc = ix_lower_bound(&_ih, (input_buffer_t) cond.rhs_value.data, &_upper);
                        assert(rc == 0);
                    } else if (cond.op == OP_GT) {
                        rc = ix_upper_bound(&_ih, (input_buffer_t) cond.rhs_value.data, &_lower);
                        assert(rc == 0);
                    } else if (cond.op == OP_LE) {
                        rc = ix_upper_bound(&_ih, (input_buffer_t) cond.rhs_value.data, &_upper);
                        assert(rc == 0);
                    } else if (cond.op == OP_GE) {
                        rc = ix_lower_bound(&_ih, (input_buffer_t) cond.rhs_value.data, &_lower);
                        assert(rc == 0);
                    } else {
                        assert(0);
                    }
                    break; // TODO: remove it
                }
            }
            while (!ix_scan_equal(&_lower, &_upper)) {
                rc = ix_get_rid(&_ih, &_lower, &rec->rid);
                assert(rc == 0);
                rc = rm_get_record(_fh, &rec->rid, rec->data);
                assert(rc == 0);
                if (eval_conds(tbidx, rec, _conds)) { break; }
                rc = ix_scan_next(&_ih, &_lower);
                assert(rc == 0);
            }
        } else {
            // scan rm
            rc = rm_scan_init(_fh, &rec->rid);
            assert(rc == 0);
            // get the first record
            while (!rm_scan_is_end(&rec->rid)) {
                rc = rm_get_record(_fh, &rec->rid, rec->data);
                assert(rc == 0);
                if (eval_conds(tbidx, rec, _conds)) { break; }
                rc = rm_scan_next(_fh, &rec->rid);
                assert(rc == 0);
            }
        }
    }

    void close_scan(rm_record_t *rec) {
        rm_record_destroy(rec);
    }

    bool is_end(rm_record_t *record) {
        if (_index_no != -1) {
            // scan ix
            return ix_scan_equal(&_lower, &_upper);
        } else {
            // scan rm
            return rm_scan_is_end(&record->rid);
        }
    }

    RC get_next_record(rm_record_t *record) {
        RC rc;
        assert(!is_end(record));
        while (!is_end(record)) {
            if (_index_no != -1) {
                // scan ix
                rc = ix_scan_next(&_ih, &_lower);
                if (rc) { return rc; }
                if (ix_scan_equal(&_lower, &_upper)) { break; }
                rc = ix_get_rid(&_ih, &_lower, &record->rid);
                if (rc) { return rc; }
            } else {
                // scan rm
                rc = rm_scan_next(_fh, &record->rid);
                if (rc) { return rc; }
                if (rm_scan_is_end(&record->rid)) { break; }
            }
            rc = rm_get_record(_fh, &record->rid, record->data);
            if (rc) { return rc; }
            // validate all conditions
            if (eval_conds(_tbidx, record, _conds)) {
                break;
            }
        }
        return 0;
    }
};