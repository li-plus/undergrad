project(sql_c C)

find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

bison_target(yacc yacc.y ${CMAKE_CURRENT_SOURCE_DIR}/yacc.tab.c
        DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/yacc.tab.h)
flex_target(lex lex.l ${CMAKE_CURRENT_SOURCE_DIR}/lex.yy.c)
add_flex_bison_dependency(lex yacc)

set(SOURCES ${BISON_yacc_OUTPUT_SOURCE} ${FLEX_lex_OUTPUTS} ast.c printer.c)
add_library(parser STATIC ${SOURCES})

add_subdirectory(test)
