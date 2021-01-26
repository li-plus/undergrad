#include "parser/parser.h"
#include <cstdio>


int main() {
    parser_init();
    while (yyparse() == 0 && parse_tree != nullptr) {
        printf("Parse Tree:\n");
        print_tree(parse_tree, 2);
        printf("\n");
        parser_init();
    }
    printf("exited\n");
    return 0;
}
