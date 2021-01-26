#include "sm_manager.h"
#include "sm_error.h"
#include "ql/ql.h"
#include "rm/rm.h"
#include "ix/ix.h"
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <string>
#include <set>

sm_config_t sm_config;

db_meta_t db_meta;

rm_file_handle_t sm_fhs[SM_MAX_TABLES];

int sm_find_table(const char *tbname) {
    int tbidx = 0;
    while (tbidx < db_meta.num_tables) {
        if (strcmp(db_meta.tables[tbidx].name, tbname) == 0) {
            break;
        }
        tbidx++;
    }
    return tbidx;
}

int sm_find_field(int tbidx, const char *field_name) {
    assert(tbidx < db_meta.num_tables);
    table_meta_t &tbmeta = db_meta.tables[tbidx];
    int field_idx = 0;
    while (field_idx < tbmeta.num_fields) {
        if (strcmp(field_name, tbmeta.fields[field_idx].name) == 0) {
            break;
        }
        field_idx++;
    }
    return field_idx;
}

static char *get_buf() {
    static char buf[PAGE_SIZE];
    return buf;
}

static char *get_dbpath(const char *dbname) {
    char *dbpath = get_buf();
    sprintf(dbpath, "%s/%s", sm_config.root_dir, dbname);
    return dbpath;
}

static char *get_meta_path(const char *dbname) {
    char *meta_path = get_buf();
    sprintf(meta_path, "%s/%s/%s", sm_config.root_dir, dbname, "meta");
    return meta_path;
}

static RC load_db_meta(const char *dbname) {
    RC rc;
    int fd;
    rc = pf_open_file(get_meta_path(dbname), &fd);
    if (rc) { return rc; }
    rc = pf_read_page(fd, 0, (output_buffer_t) &db_meta, sizeof(db_meta));
    if (rc) { return rc; }
    rc = pf_close_file(fd);
    if (rc) { return rc; }
    return 0;
}

static RC dump_db_meta(const char *dbname) {
    RC rc;
    int fd;
    rc = pf_open_file(get_meta_path(dbname), &fd);
    if (rc) { return rc; }
    rc = pf_write_page(fd, 0, (input_buffer_t) &db_meta, sizeof(db_meta));
    if (rc) { return rc; }
    rc = pf_close_file(fd);
    if (rc) { return rc; }
    return 0;
}

static RC open_all_rmfh() {
    RC rc;
    for (int i = 0; i < db_meta.num_tables; i++) {
        table_meta_t *tbmeta = &db_meta.tables[i];
        rc = rm_open_file(tbmeta->name, &sm_fhs[i]);
        if (rc) { return rc; }
    }
    return 0;
}

static RC close_all_rmfh() {
    RC rc;
    for (int i = 0; i < db_meta.num_tables; i++) {
        rc = rm_close_file(&sm_fhs[i]);
        if (rc) { return rc; }
    }
    return 0;
}

static bool is_db_open() {
    return strlen(sm_config.curr_db) > 0;
}

static RC sm_close_database() {
    RC rc;
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    // Dump meta info
    rc = dump_db_meta(sm_config.curr_db);
    if (rc) { return rc; }
    rc = close_all_rmfh();
    if (rc) { return rc; }
    db_meta.num_tables = 0;
    return 0;
}

void sm_init(const char *root_dir) {
    strcpy(sm_config.root_dir, root_dir);
}

RC sm_destroy() {
    RC rc;
    if (is_db_open()) {
        rc = sm_close_database();
        if (rc) { return rc; }
    }
    return 0;
}

RC sm_show_databases() {
    DIR *dir = opendir(sm_config.root_dir);
    if (dir == NULL) {
        return SM_BAD_ROOT_DIR;
    }
    struct dirent *path;
    while ((path = readdir(dir)) != NULL) {
        if (path->d_type == DT_DIR && strcmp(path->d_name, ".") != 0 && strcmp(path->d_name, "..") != 0) {
            printf("%s\n", path->d_name);
        }
    }
    if (closedir(dir) != 0) {
        return SM_BAD_ROOT_DIR;
    }
    return 0;
}

bool sm_database_exists(const char *dbname) {
    DIR *dir = opendir(sm_config.root_dir);
    if (dir == NULL) {
        return false;
    }
    struct dirent *path;
    bool found = false;
    while ((path = readdir(dir)) != NULL) {
        if (path->d_type == DT_DIR && strcmp(path->d_name, dbname) == 0) {
            found = true;
            break;
        }
    }
    closedir(dir);
    return found;
}

RC sm_create_database(const char *dbname) {
    RC rc;
    if (sm_database_exists(dbname)) {
        return SM_DATABASE_EXISTS;
    }
    char *dbpath = get_dbpath(dbname);
    if (mkdir(dbpath, S_IRWXU) != 0) { return SM_UNIX; }
    // Create meta data file
    rc = pf_create_file(get_meta_path(dbname));
    if (rc) { return rc; }
    int curr_num_tables = db_meta.num_tables;
    db_meta.num_tables = 0;
    rc = dump_db_meta(dbname);
    db_meta.num_tables = curr_num_tables;
    if (rc) { return rc; }
    return 0;
}

RC sm_drop_database(const char *dbname) {
    if (!sm_database_exists(dbname)) { return SM_DATABASE_NOT_FOUND; }
    if (strcmp(sm_config.curr_db, dbname) == 0) {
        chdir(sm_config.root_dir);
        strcpy(sm_config.curr_db, "");
        sm_close_database();
    }
    char *dbpath = get_dbpath(dbname);
    char cmd[512] = "rm -r ";
    if (system(strcat(cmd, dbpath)) != 0) { return SM_UNIX; }
    return 0;
}

RC sm_use_database(const char *dbname) {
    RC rc;
    if (!sm_database_exists(dbname)) { return SM_DATABASE_NOT_FOUND; }
    if (is_db_open()) {
        rc = sm_close_database();
        if (rc) { return rc; }
    }
    // open new database
    char *dbpath = get_dbpath(dbname);
    if (chdir(dbpath) != 0) { return SM_UNIX; }
    strcpy(sm_config.curr_db, dbname);
    rc = load_db_meta(dbname);
    if (rc) { return rc; }
    rc = open_all_rmfh();
    if (rc) { return rc; }
    return 0;
}

RC sm_show_tables() {
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    printf("Tables_in_%s\n", sm_config.curr_db);
    for (int i = 0; i < db_meta.num_tables; i++) {
        table_meta_t *table = &db_meta.tables[i];
        printf("%-10s PRIMARY KEY: ", table->name);
        for (int j = 0; j < table->num_fields; j++) {
            if (table->fields[j].is_primary) {
                printf("%s ", table->fields[i].name);
            }
        }
        printf("\n");
    }
    return 0;
}

RC sm_create_table(const char *tbname, int num_fields, field_info_t *fields) {
    RC rc;
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    if (pf_exists(tbname)) { return SM_TABLE_EXISTS; }
    // Total record size
    int record_size = 0;
    // Table meta info
    table_meta_t *table_meta = &db_meta.tables[db_meta.num_tables];
    strcpy(table_meta->name, tbname);
    int curr_offset = 0;
    table_meta->num_fields = 0;
    for (int i = 0; i < num_fields; i++) {
        record_size += fields[i].len + 1;
        // Field meta info
        field_meta_t *field_meta = &table_meta->fields[table_meta->num_fields];
        strcpy(field_meta->tbname, table_meta->name);
        strcpy(field_meta->name, fields[i].name);
        field_meta->type = fields[i].type;
        field_meta->len = fields[i].len;
        field_meta->nullable = fields[i].nullable;
        field_meta->index = fields[i].is_primary;   // primary key should be indexed
        field_meta->offset = curr_offset;
        field_meta->is_primary = fields[i].is_primary;
        field_meta->has_frgn_key = fields[i].has_frgn_key;
        if (fields[i].has_frgn_key) {
            strcpy(field_meta->frgn_tab, fields[i].frgn_tab);
            strcpy(field_meta->frgn_col, fields[i].frgn_col);
        }
        curr_offset += fields[i].len + 1;
        table_meta->num_fields++;
    }
    // Create & open record file
    rc = rm_create_file(tbname, record_size);
    if (rc) { return rc; }
    rc = rm_open_file(tbname, &sm_fhs[db_meta.num_tables]);
    if (rc) { return rc; }
    db_meta.num_tables++;
    // Create index for primary key
    for (int i = 0; i < num_fields; i++) {
        if (fields[i].is_primary) {
            rc = ix_create_index(tbname, i, fields[i].type, 1 + fields[i].len);
            if (rc) { return rc; }
        }
    }
    return 0;
}

RC sm_drop_table(const char *tbname) {
    RC rc;
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    // Find table index in db meta
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    table_meta_t *table = &db_meta.tables[tbidx];
    // Close & destroy record file
    rm_file_handle_t *rmfh = &sm_fhs[tbidx];
    rc = rm_close_file(rmfh);
    if (rc) { return rc; }
    rc = rm_destroy_file(tbname);
    if (rc) { return rc; }
    // Close & destroy index file
    for (int field_idx = 0; field_idx < table->num_fields; field_idx++) {
        field_meta_t *field = &table->fields[field_idx];
        if (field->index) {
            rc = ix_destroy_index(tbname, field_idx);
            if (rc) { return rc; }
        }
    }
    // Erase table meta
    memmove(db_meta.tables + tbidx,
            db_meta.tables + tbidx + 1,
            (db_meta.num_tables - tbidx - 1) * sizeof(table_meta_t));
    // Erase rm file handle
    memmove(sm_fhs + tbidx,
            sm_fhs + tbidx + 1,
            (db_meta.num_tables - tbidx - 1) * sizeof(rm_file_handle_t));
    db_meta.num_tables--;
    return 0;
}

RC sm_desc_table(const char *tbname) {
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    // Find the meta info of this table
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) {
        return SM_TABLE_NOT_FOUND;
    }
    table_meta_t *table = &db_meta.tables[tbidx];
    printf("| %15s | %15s | %15s | %15s | %15s | %15s |\n", "Field", "Type", "Len", "Null", "Index", "Offset");
    for (int field_idx = 0; field_idx < table->num_fields; field_idx++) {
        auto &field = table->fields[field_idx];
        printf("| %15s | %15d | %15d | %15d | %15d | %15d |\n", field.name, field.type, field.len, field.nullable,
               field.index, field.offset);
    }
    return 0;
}

RC sm_create_index(const char *tbname, const char *colname) {
    RC rc;
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    // Create index file
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    table_meta_t *table_meta = &db_meta.tables[tbidx];
    int colidx = sm_find_field(tbidx, colname);
    if (colidx == table_meta->num_fields) { return SM_FIELD_NOT_FOUND; }
    field_meta_t *field_meta = &table_meta->fields[colidx];
    rc = ix_create_index(tbname, colidx, field_meta->type, field_meta->len);
    if (rc) { return rc; }
    // Open index file
    ix_file_handle_t ixfh;
    rc = ix_open_index(tbname, colidx, &ixfh);
    if (rc) { return rc; }
    // Open record file
    rm_file_handle_t *rmfh = &sm_fhs[tbidx];
    // Index all records
    rm_record_t rec;
    rm_record_init(&rec, rmfh);
    rc = rm_scan_init(rmfh, &rec.rid);
    if (rc) { return rc; }
    while (!rm_scan_is_end(&rec.rid)) {
        // Get record
        rc = rm_get_record(rmfh, &rec.rid, rec.data);
        if (rc) { return rc; }
        // Insert into ix
        input_buffer_t key = rec.data + field_meta->offset;
        rc = ix_insert_entry(&ixfh, key, &rec.rid);
        if (rc) { return rc; }
        rc = rm_scan_next(rmfh, &rec.rid);
        if (rc) { return rc; }
    }
    rm_record_destroy(&rec);
    return 0;
}

RC sm_drop_index(const char *tbname, const char *colname) {
    RC rc;
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    int colidx = sm_find_field(tbidx, colname);
    if (colidx == db_meta.tables[tbidx].num_fields) { return SM_FIELD_NOT_FOUND; }
    rc = ix_destroy_index(tbname, colidx);
    if (rc) { return rc; }
    return 0;
}

RC sm_add_primary_key(const char *tbname, const std::vector<std::string> &colnames) {
    RC rc;
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    auto &tbmeta = db_meta.tables[tbidx];
    // Check whether primary key exists
    for (int i = 0; i < tbmeta.num_fields; i++) {
        if (tbmeta.fields[i].is_primary) {
            return QL_MULTIPLE_PRIMARY_KEY_DEFINED;
        }
    }
    // check primary key entries
    if (colnames.size() == 1) {
        rm_record_t rec;
        std::string colname = colnames.front();
        int colidx = sm_find_field(tbidx, colname.c_str());
        std::set<int> si;
        std::set<float> sf;
        std::set<std::string> ss;
        table_scan_t scan(tbidx, {}, &rec);
        while (!scan.is_end(&rec)) {
            auto buf = rec.data + tbmeta.fields[colidx].offset;
            if (buf[0] == 0) {
                return QL_EXPECTED_NOT_NULL;
            }
            if (tbmeta.fields[colidx].type == ATTR_INT) {
                int val = *(int *) (buf + 1);
                if (si.count(val)) { return QL_DUPLICATE_ENTRY; }
                si.insert(val);
            } else if (tbmeta.fields[colidx].type == ATTR_FLOAT) {
                float val = *(float *) (buf + 1);
                if (sf.count(val)) { return QL_DUPLICATE_ENTRY; }
                sf.insert(val);
            } else if (tbmeta.fields[colidx].type == ATTR_STRING) {
                auto val = std::string((char *) buf + 1, tbmeta.fields[colidx].len);
                if (ss.count(val)) { return QL_DUPLICATE_ENTRY; }
                ss.insert(val);
            }
            rc = scan.get_next_record(&rec);
            if (rc) { return rc; }
        }
        scan.close_scan(&rec);
    }
    // Add primary key
    for (auto &colname: colnames) {
        int colidx = sm_find_field(tbidx, colname.c_str());
        if (colidx == tbmeta.num_fields) { return SM_FIELD_NOT_FOUND; }
        tbmeta.fields[colidx].is_primary = true;
        tbmeta.fields[colidx].nullable = false;
        if (!tbmeta.fields[colidx].index) {
            tbmeta.fields[colidx].index = true;
            rc = sm_create_index(tbname, colname.c_str());
            if (rc) { return rc; }
        }
    }
    return 0;
}

RC sm_drop_primary_key(const char *tbname) {
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    auto &tbmeta = db_meta.tables[tbidx];
    bool has_prim_key = false;
    for (int i = 0; i < tbmeta.num_fields; i++) {
        if (tbmeta.fields[i].is_primary) {
            has_prim_key = true;
            tbmeta.fields[i].is_primary = false;
        }
    }
    if (!has_prim_key) { return QL_PRIMARY_KEY_NOT_FOUND; }
    return 0;
}

RC sm_rename_table(const char *tbname, const char *new_tbname) {
    RC rc;
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    if (strcmp(tbname, new_tbname) == 0) { return 0; }
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    int new_tbidx = sm_find_table(new_tbname);
    if (new_tbidx != db_meta.num_tables) { return SM_TABLE_EXISTS; }
    auto &tbmeta = db_meta.tables[tbidx];
    // rename rm file
    auto &fh = sm_fhs[tbidx];
    rc = rm_close_file(&fh);
    if (rc) { return rc; }
    if (rename(tbname, new_tbname) != 0) { return PF_UNIX; }
    rc = rm_open_file(new_tbname, &fh);
    if (rc) { return rc; }
    // rename ix file
    for (int i = 0; i < tbmeta.num_fields; i++) {
        if (tbmeta.fields[i].index) {
            char old_ix_name[512], new_ix_name[512];
            strcpy(old_ix_name, ix_get_filename(tbmeta.name, i));
            strcpy(new_ix_name, ix_get_filename(new_tbname, i));
            if (rename(old_ix_name, new_ix_name) != 0) { return PF_UNIX; }
        }
    }
    // rename meta
    strcpy(tbmeta.name, new_tbname);
    for (int i = 0; i < tbmeta.num_fields; i++) {
        strcpy(tbmeta.fields[i].tbname, new_tbname);
    }
    for (int tabidx = 0; tabidx < db_meta.num_tables; tabidx++) {
        auto &tabmeta = db_meta.tables[tabidx];
        for (int colidx = 0; colidx < tabmeta.num_fields; colidx++) {
            if (strcmp(tabmeta.fields[colidx].frgn_tab, tbname) == 0) {
                strcpy(tabmeta.fields[colidx].frgn_tab, new_tbname);
            }
        }
    }
    return 0;
}

RC sm_add_foreign_key(const char *tbname, const char *fkname, const std::vector<std::string> &colnames,
                      const char *ref_tbname, const std::vector<std::string> &ref_colnames) {
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    int ref_tbidx = sm_find_table(ref_tbname);
    if (ref_tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    auto &ref_tbmeta = db_meta.tables[ref_tbidx];
    // check whether ref keys are primary
    for (auto &ref_colname: ref_colnames) {
        int colidx = sm_find_field(ref_tbidx, ref_colname.c_str());
        if (colidx == ref_tbmeta.num_fields) { return SM_FIELD_NOT_FOUND; }
        if (!ref_tbmeta.fields[colidx].is_primary) { return QL_FOREIGN_KEY_NOT_PRIMARY; }
    }
    if (colnames.size() != ref_colnames.size()) { return QL_INVALID_FOREIGN_KEY_COUNT; }
    auto &tbmeta = db_meta.tables[tbidx];
    for (int i = 0; i < colnames.size(); i++) {
        auto &colname = colnames[i];
        auto &ref_colname = ref_colnames[i];
        int colidx = sm_find_field(tbidx, colname.c_str());
        if (colidx == tbmeta.num_fields) { return SM_FIELD_NOT_FOUND; }
        auto &colmeta = tbmeta.fields[colidx];
        colmeta.has_frgn_key = true;
        strcpy(colmeta.frgn_tab, ref_tbname);
        strcpy(colmeta.frgn_col, ref_colname.c_str());
    }
    return 0;
}

RC sm_drop_foreign_key(const char *tbname, const char *fkname) {
    if (!is_db_open()) { return SM_DATABASE_NOT_OPEN; }
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    // TODO: use fkname
    // currently we drop all foreign keys
    auto &tbmeta = db_meta.tables[tbidx];
    for (int i = 0; i < tbmeta.num_fields; i++) {
        tbmeta.fields[i].has_frgn_key = false;
    }
    return 0;
}

RC sm_add_col(const char *tbname, char *colname, bool nullable, attr_type_t type, int len) {
    RC rc;
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    auto &tbmeta = db_meta.tables[tbidx];
    std::vector<field_info_t> new_fields;
    for (int i = 0; i < tbmeta.num_fields; i++) {
        auto &field = tbmeta.fields[i];
        field_info_t info;
        info.name = field.name;
        info.type = field.type;
        info.len = field.len;
        info.nullable = field.nullable;
        info.is_primary = field.is_primary;
        info.has_frgn_key = field.has_frgn_key;
        info.frgn_tab = field.frgn_tab;
        info.frgn_col = field.frgn_col;
        new_fields.push_back(info);
    }
    field_info_t info;
    info.name = colname;
    info.type = type;
    info.len = len;
    info.nullable = nullable;
    info.is_primary = false;
    info.has_frgn_key = false;
    info.frgn_tab = "";
    info.frgn_col = "";
    new_fields.push_back(info);

    std::string tmp_tbname = tbname + std::string("_TMP");
    rc = sm_create_table(tmp_tbname.c_str(), new_fields.size(), &new_fields[0]);
    if (rc) { return rc; }
    int tmp_tbidx = sm_find_table(tmp_tbname.c_str());
    assert(tmp_tbidx < db_meta.num_tables);
    rm_record_t rec;
    table_scan_t tb_scan(tbidx, {}, &rec);
    while (!tb_scan.is_end(&rec)) {
        std::vector<value_t> vals;
        for (int i = 0; i < tbmeta.num_fields; i++) {
            value_t val;
            val.data = rec.data + tbmeta.fields[i].offset + 1;
            val.type = tbmeta.fields[i].type;
            val.is_null = !rec.data[tbmeta.fields[i].offset];
            vals.push_back(val);
        }
        value_t val;
        val.type = type;
        val.is_null = true;
        // NULL?
        vals.push_back(val);
        rc = ql_insert(tmp_tbname, vals);
        if (rc) { return rc; }
        rc = tb_scan.get_next_record(&rec);
        if (rc) { return rc; }
    }
    tb_scan.close_scan(&rec);
    rc = sm_drop_table(tbname);
    if (rc) { return rc; }
    rc = sm_rename_table(tmp_tbname.c_str(), tbname);
    if (rc) { return rc; }
    return 0;
}

RC sm_drop_col(const char *tbname, const char *colname) {
    RC rc;
    int tbidx = sm_find_table(tbname);
    if (tbidx == db_meta.num_tables) { return SM_TABLE_NOT_FOUND; }
    auto &tbmeta = db_meta.tables[tbidx];
    int colidx = sm_find_field(tbidx, colname);
    if (colidx == tbmeta.num_fields) { return SM_FIELD_NOT_FOUND; }
    std::vector<field_info_t> new_fields;
    for (int i = 0; i < tbmeta.num_fields; i++) {
        auto &field = tbmeta.fields[i];
        field_info_t info;
        if (i != colidx) {
            info.name = field.name;
            info.type = field.type;
            info.len = field.len;
            info.nullable = field.nullable;
            info.is_primary = field.is_primary;
            info.has_frgn_key = field.has_frgn_key;
            info.frgn_tab = field.frgn_tab;
            info.frgn_col = field.frgn_col;
            new_fields.push_back(info);
        }
    }
    std::string tmp_tbname = tbname + std::string("_TMPC");
    rc = sm_create_table(tmp_tbname.c_str(), new_fields.size(), &new_fields[0]);
    if (rc) { return rc; }
    rm_record_t rec;
    table_scan_t tb_scan(tbidx, {}, &rec);
    while (!tb_scan.is_end(&rec)) {
        std::vector<value_t> vals;
        for (int i = 0; i < tbmeta.num_fields; i++) {
            if (i != colidx) {
                value_t val;
                val.data = rec.data + tbmeta.fields[i].offset + 1;
                val.type = tbmeta.fields[i].type;
                val.is_null = !rec.data[tbmeta.fields[i].offset];
                vals.push_back(val);
            }
        }
        rc = ql_insert(tmp_tbname, vals);
        if (rc) { return rc; }
        rc = tb_scan.get_next_record(&rec);
        if (rc) { return rc; }
    }
    tb_scan.close_scan(&rec);
    rc = sm_drop_table(tbname);
    if (rc) { return rc; }
    rc = sm_rename_table(tmp_tbname.c_str(), tbname);
    if (rc) { return rc; }
    return 0;
}
