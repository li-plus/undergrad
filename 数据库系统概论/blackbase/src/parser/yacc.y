%{
#include "ast.h"
#include <stdio.h>
#include <stdlib.h>

int yylex();

void yyerror(const char* s) {
    fprintf(stderr, "Parse error: %s\n", s);
}
%}

// keywords
%token DATABASE DATABASES TABLE TABLES SHOW CREATE
DROP USE PRIMARY KEY NOT SQLNULL INSERT INTO VALUES
DELETE FROM WHERE UPDATE SET SELECT IS INT VARCHAR
DEFAULT CONSTRAINT CHANGE ALTER ADD RENAME DESC
INDEX AND DATE FLOAT FOREIGN REFERENCES ON TO
// non-keywords
%token LEQ NEQ GEQ T_EOF

// available semantic values (sv) types
%union{
    // TODO: move these values into sv_node_t
    char* sv_str;
    int sv_int;
    float sv_float;
    sv_type_len_t sv_type_len;
    sv_comp_op_t sv_comp_op;
    sv_node_t* sv_node;
}

// type-specific tokens
%token <sv_str> IDENTIFIER VALUE_STRING
%token <sv_int> VALUE_INT
%token <sv_float> VALUE_FLOAT

// specify types for non-terminal symbol
%type <sv_node> stmt sysStmt dbStmt tbStmt idxStmt alterStmt
    fieldList field colNameList valueLists valueList value
    whereClause condition colList col expr setClause
    setClauses tableList selector
%type <sv_str> dbName tbName colName idxName pkName fkName SQLNULL
%type <sv_type_len> type
%type <sv_comp_op> op

// %left T_PLUS T_MINUS
// %left T_MULTIPLY T_DIVIDE

%%
start:
        { /* ignore */ }
    |   stmt ';'
    {
        parse_tree = $1;
        YYACCEPT;
    }
    |   T_EOF
    {
        parse_tree = NULL;
        YYACCEPT;
    }
    ;

stmt:
        sysStmt
    |   dbStmt
    |   tbStmt
    |   idxStmt
    |   alterStmt
    ;

sysStmt:
        SHOW DATABASES
    {
        $$ = alloc_show_databases();
    }
    ;

dbStmt:
        CREATE DATABASE dbName
    {
        $$ = alloc_create_database($3);
    }
    |   DROP DATABASE dbName
    {
        $$ = alloc_drop_database($3);
    }
    |   USE dbName
    {
        $$ = alloc_use_database($2);
    }
    |   SHOW TABLES
    {
        $$ = alloc_show_tables();
    }
    ;

tbStmt:
        CREATE TABLE tbName '(' fieldList ')'
    {
        $$ = alloc_create_table($3, $5);
    }
    |   DROP TABLE tbName
    {
        $$ = alloc_drop_table($3);
    }
    |   DESC tbName
    {
        $$ = alloc_desc_table($2);
    }
    |   INSERT INTO tbName VALUES valueLists
    {
        $$ = alloc_insert($3, $5);
    }
    |   DELETE FROM tbName WHERE whereClause
    {
        $$ = alloc_delete_from($3, $5);
    }
    |   UPDATE tbName SET setClauses WHERE whereClause
    {
        $$ = alloc_update_set($2, $4, $6);
    }
    |   SELECT selector FROM tableList WHERE whereClause
    {
        $$ = alloc_select_from($2, $4, $6);
    }
    |   SELECT selector FROM tableList
    {
        $$ = alloc_select_from($2, $4, NULL);
    }
    ;

idxStmt:
        CREATE INDEX idxName ON tbName '(' colNameList ')'
    {
        $$ = alloc_create_index($3, $5, $7);
    }
    |   DROP INDEX idxName
    {
        $$ = alloc_drop_index($3);
    }
    |   ALTER TABLE tbName ADD INDEX idxName '(' colNameList ')'
    {
        $$ = alloc_create_index($6, $3, $8);
    }
    |   ALTER TABLE tbName DROP INDEX idxName
    {
        $$ = alloc_drop_index($6);
    }
    ;

alterStmt:
        ALTER TABLE tbName ADD field
    {
        $$ = alloc_alter_add_field($3, $5);
    }
    |   ALTER TABLE tbName DROP colName
    {
        $$ = alloc_alter_drop_col($3, $5);
    }
    |   ALTER TABLE tbName CHANGE colName field
    {
        $$ = alloc_alter_change_col($3, $5, $6);
    }
    |   ALTER TABLE tbName RENAME TO tbName
    {
        $$ = alloc_alter_rename_table($3, $6);
    }
    |   ALTER TABLE tbName DROP PRIMARY KEY
    {
        $$ = alloc_alter_drop_primary_key($3);
    }
    |   ALTER TABLE tbName ADD CONSTRAINT pkName PRIMARY KEY '(' colNameList ')'
    {
        $$ = alloc_alter_add_primary_key($3, $10);
    }
    |   ALTER TABLE tbName DROP PRIMARY KEY pkName
    {
        $$ = alloc_alter_drop_primary_key($3);
    }
    |   ALTER TABLE tbName ADD CONSTRAINT fkName FOREIGN KEY '(' colNameList ')' REFERENCES tbName '(' colNameList ')'
    {
        $$ = alloc_alter_add_foreign_key($3, $6, $10, $13, $15);
    }
    |   ALTER TABLE tbName DROP FOREIGN KEY fkName
    {
        $$ = alloc_alter_drop_foreign_key($3, $7);
    }
    ;

fieldList:
        field
    {
        $$ = alloc_list($1);
    }
    |   field ',' fieldList
    {
        $$ = sv_list_push_front($3, $1);
    }
    ;

field:
        colName type
    {
        $$ = alloc_column_def($1, $2, true);
    }
    |   colName type NOT SQLNULL
    {
        $$ = alloc_column_def($1, $2, false);
    }
    |   colName type DEFAULT value
    {
        // TODO: defaults
        $$ = alloc_column_def($1, $2, true);
    }
    |   colName type NOT SQLNULL DEFAULT value
    {
        $$ = alloc_column_def($1, $2, false);
    }
    |   PRIMARY KEY '(' colNameList ')'
    {
        $$ = alloc_primary_key($4);
    }
    |   FOREIGN KEY '(' colNameList ')' REFERENCES tbName '(' colNameList ')'
    {
        $$ = alloc_foreign_key($4, $7, $9);
    }
    ;

type:
        INT '(' VALUE_INT ')'
    {
        $$.type = SV_TYPE_INT;
        $$.len = $3;
    }
    |   VARCHAR '(' VALUE_INT ')'
    {
        $$.type = SV_TYPE_STRING;
        $$.len = $3;
    }
    |   DATE
    {
        $$.type = SV_TYPE_STRING;
        $$.len = 10;
    }
    |   FLOAT
    {
        $$.type = SV_TYPE_FLOAT;
        $$.len = 4;
    }
    ;

valueLists:
        '(' valueList ')'
    {
        $$ = alloc_list($2);
    }
    |   '(' valueList ')' ',' valueLists
    {
        $$ = sv_list_push_front($5, $2);
    }
    ;

valueList:
        value
    {
        $$ = alloc_list($1);
    }
    |   value ',' valueList
    {
        $$ = sv_list_push_front($3, $1);
    }
    ;

value:
        VALUE_INT
    {
        $$ = alloc_value_int($1);
    }
    |   VALUE_FLOAT
    {
        $$ = alloc_value_float($1);
    }
    |   VALUE_STRING
    {
        $$ = alloc_value_string($1);
    }
    |   SQLNULL
    {
        $$ = alloc_sqlnull();
    }
    ;

condition:
        col op expr
    {
        $$ = alloc_binary_expr($1, $2, $3);
    }
    |   col IS SQLNULL
    {
        $$ = alloc_binary_expr($1, SV_EQ, alloc_sqlnull());
    }
    |   col IS NOT SQLNULL
    {
        $$ = alloc_binary_expr($1, SV_NE, alloc_sqlnull());
    }
    ;

whereClause:
        condition
    {
        $$ = alloc_list($1);
    }
    |   condition AND whereClause
    {
        $$ = sv_list_push_front($3, $1);
    }
    ;

col:
        tbName '.' colName
    {
        $$ = alloc_col($1, $3);
    }
    |   colName
    {
        $$ = alloc_col("", $1);
    }
    ;

colList:
        col
    {
        $$ = alloc_list($1);
    }
    |   col ',' colList
    {
        $$ = sv_list_push_front($3, $1);
    }
    ;

op:
        '='
    {
        $$ = SV_EQ;
    }
    |   '<'
    {
        $$ = SV_LT;
    }
    |   '>'
    {
        $$ = SV_GT;
    }
    |   NEQ
    {
        $$ = SV_NE;
    }
    |   LEQ
    {
        $$ = SV_LE;
    }
    |   GEQ
    {
        $$ = SV_GE;
    }
    ;

expr:
        value
    |   col
    ;

setClauses:
        setClause
    {
        $$ = alloc_list($1);
    }
    |   setClause ',' setClauses
    {
        $$ = sv_list_push_front($3, $1);
    }
    ;

setClause:
        colName '=' value
    {
        $$ = alloc_set_clause($1, $3);
    }
    ;

selector:
        '*'
    {
        $$ = alloc_list(alloc_col("", "*"));
    }
    |   colList
    {
        $$ = $1;
    }
    ;

tableList:
        tbName
    {
        sv_node_t* curr = alloc_identifier($1);
        $$ = alloc_list(curr);
    }
    |   tbName ',' tableList
    {
        sv_node_t* curr = alloc_identifier($1);
        $$ = sv_list_push_front($3, curr);
    }
    ;

colNameList:
        colName
    {
        sv_node_t* curr = alloc_identifier($1);
        $$ = alloc_list(curr);
    }
    |   colName ',' colNameList 
    {
        sv_node_t* curr = alloc_identifier($1);
        $$ = sv_list_push_front($3, curr);
    }
    ;

dbName: IDENTIFIER;

tbName: IDENTIFIER;

colName: IDENTIFIER;

idxName: IDENTIFIER;

pkName: IDENTIFIER;

fkName: IDENTIFIER;
%%
