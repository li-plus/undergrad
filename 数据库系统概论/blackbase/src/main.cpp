#include "parser/parser.h"
#include "ql/ql.h"
#include "sm/sm.h"
#include "ix/ix.h"
#include "rm/rm.h"
#include "pf/pf.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

static const char *db_str_error(RC rc) {
    if (PF_ERROR_START <= rc && rc < PF_ERROR_END) {
        return pf_str_error(rc);
    } else if (RM_ERROR_START <= rc && rc < RM_ERROR_END) {
        return rm_str_error(rc);
    } else if (IX_ERROR_START <= rc && rc < IX_ERROR_END) {
        return ix_str_error(rc);
    } else if (SM_ERROR_START <= rc && rc < SM_ERROR_END) {
        return sm_str_error(rc);
    } else if (QL_ERROR_START <= rc && rc < QL_ERROR_END) {
        return ql_str_error(rc);
    } else {
        assert(0);
    }
}

int main() {
    RC rc;
    parser_init();
    sm_init("/Users/bytedance/blackbase/data");
    pf_init();
    while (1) {
        if (yyparse() == 0) {
            if (parse_tree == NULL) { break; }
//        printf("Parse Tree:\n");
//        print_tree(parse_tree, 2);
//        printf("\n");
            rc = exec_sql(parse_tree);
            if (rc) {
                fprintf(stderr, "%s\n", db_str_error(rc));
//                print_tree(parse_tree, 0);
//                exec_sql(parse_tree);
            }
        }
        parser_init();
    }
    sm_destroy();
    return 0;
}
