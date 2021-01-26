#include "sm/sm.h"
#include "pf/pf.h"
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cassert>
#include <string>

int main() {
    char cwd[SM_MAX_NAME_LEN];
    assert(getcwd(cwd, sizeof(cwd)) != nullptr);

    std::string root_dir = std::string(cwd) + "/data";
    int ret = mkdir(root_dir.c_str(), S_IRWXU);
    assert(ret == 0 || errno == EEXIST);

    sm_init(root_dir.c_str());
    pf_init();

    char db1[] = "db1";
    char db2[] = "db2";
    if (sm_database_exists(db1)) {
        assert(sm_drop_database(db1) == 0);
    }
    if (sm_database_exists(db2)) {
        assert(sm_drop_database(db2) == 0);
    }
    // Cannot use a database that does not exist
    assert(sm_use_database(db1) == SM_DATABASE_NOT_FOUND);
    // Create database normally
    assert(sm_create_database(db1) == 0);
    // Cannot re-create database
    assert(sm_create_database(db1) == SM_DATABASE_EXISTS);
    // Use database
    assert(sm_use_database(db1) == 0);
    // Create another database
    assert(sm_create_database(db2) == 0);
    // Use db2
    assert(sm_use_database(db2) == 0);
    // Use db1
    assert(sm_use_database(db1) == 0);

    field_info_t fields[3]{
            {.name="a", .type=ATTR_INT, .len=4, .nullable=true, .is_primary=true},
            {.name="b", .type=ATTR_FLOAT, .len=4, .nullable=true, .is_primary=false},
            {.name="c", .type=ATTR_STRING, .len=256, .nullable=true, .is_primary=false}
    };
    // Create table 1
    char tb1[] = "tb1";
    assert(sm_create_table(tb1, 3, fields) == 0);
    assert(sm_desc_table(tb1) == 0);
    // Test index
    assert(sm_create_index(tb1, "a") == PF_FILE_EXISTS);
    assert(sm_create_index(tb1, "b") == 0);
    assert(sm_create_index(tb1, "b") == PF_FILE_EXISTS);
    assert(sm_drop_index(tb1, "b") == 0);
    // Use db2
    assert(sm_use_database(db2) == 0);
    // Create table 2
    char tb2[] = "tb2";
    assert(sm_create_table(tb2, 3, fields) == 0);
    assert(sm_desc_table(tb2) == 0);
    // Drop when using db2
    assert(sm_drop_database(db2) == 0);
    assert(sm_show_tables() == SM_DATABASE_NOT_OPEN);
    // Use db1
    assert(sm_use_database(db1) == 0);
    assert(sm_drop_table(tb1) == 0);
    // clean up
    assert(sm_drop_database(db1) == 0);
    return 0;
}