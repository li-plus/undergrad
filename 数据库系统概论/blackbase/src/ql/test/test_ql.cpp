#include "ql/ql.h"
#include "parser/parser.h"
#include "sm/sm.h"
#include "pf/pf.h"
#include <assert.h>

static void db_perror(RC rc) {
    if (PF_ERROR_START <= rc && rc < PF_ERROR_END) {
        fprintf(stderr, "PF Error: %s\n", pf_str_error(rc));
    } else if (RM_ERROR_START <= rc && rc < RM_ERROR_END) {
        fprintf(stderr, "RM Error: %s\n", rm_str_error(rc));
    } else if (IX_ERROR_START <= rc && rc < IX_ERROR_END) {
        fprintf(stderr, "IX Error: %s\n", ix_str_error(rc));
    } else if (SM_ERROR_START <= rc && rc < SM_ERROR_END) {
        fprintf(stderr, "SM Error: %s\n", sm_str_error(rc));
    } else if (QL_ERROR_START <= rc && rc < QL_ERROR_END) {
        fprintf(stderr, "QL Error: %s\n", ql_str_error(rc));
    } else {
        assert(0);
    }
}

int main() {
    sm_init("/Users/bytedance/blackbase/data");
    pf_init();
    parser_init();
    const char *sqls[] = {
            "drop database db;",
            "create database db;",
            "use db;",
            "create table tb(s int(4), a int(4), b float, c varchar(16), primary key(a));",
            "select * from tb;",
            "insert into tb values (0, 1, 1., 'abc');",
            "select * from tb;",
            "insert into tb values (2, 2, 2., 'def');",
            "insert into tb values (5, 3, 2., 'xyz');",
            "insert into tb values (4, 4, 2., '0123456789abcdef');",
            "insert into tb values (2, 5, NULL, 'oops');",
            "insert into tb values (NULL, 6, 3., NULL);",
            "select * from tb;",
            "select * from tb where a = 3;",
            "select * from tb where b > -100.;",
            "select * from tb where a < 2;",
            "select * from tb where b <> 1.;",
            "select * from tb where c = 'abc';",
            "select * from tb where c <= 'def';",
            "select * from tb where c >= 'def';",
            "select * from tb where c >= 'def' and a < 3;",
            "select * from tb where s < a;",
            "select * from tb where a = s;",
            "select * from tb where s > a;",
            "update tb set a = 996 where a = 3;",
            "select * from tb;",
            "update tb set b = 997., c = 'icu' where c = 'xyz';",
            "select * from tb;",
            "delete from tb where a = 996;",
            "select * from tb;",
            "select s from tb;",
            "select a, s from tb;",
            "select a, s, b, c, b from tb;",
            // join
            "create table tb2(x int(4), y float, z varchar(16), s int(4), primary key(s));",
            "insert into tb2 values (1, 2., '123', 0);",
            "insert into tb2 values (2, 3., '456', 1);",
            "insert into tb2 values (3, 1., '789', 2);",
            "select * from tb, tb2;",
            "create table tb3(m int(4), n int(4));",
            "insert into tb3 values (NULL, 888);",
            "insert into tb3 values (900, NULL);",
            "select * from tb, tb2, tb3;",
            "select * from tb, tb2, tb3 where a > 3 and x < 2;",
            "drop database db;",
    };
    RC rc;
    for (auto &sql : sqls) {
        printf("%s\n", sql);
        YY_BUFFER_STATE buffer = yy_scan_string(sql);
        if (yyparse() == 0 && parse_tree != NULL) {
//            printf("Parse Tree:\n");
//            print_tree(parse_tree, 2);
//            printf("\n");
            rc = exec_sql(parse_tree);
            if (rc) { db_perror(rc); }
            parser_init();
        } else {
            break;
        }
        yy_delete_buffer(buffer);
    }
    printf("exited\n");
    return 0;
}
