grammar SQL;

EqualOrAssign: '=';
Less: '<';
LessEqual: '<=';
Greater: '>';
GreaterEqual: '>=';
NotEqual: '<>';

Count: 'COUNT';
Average: 'AVG';
Max: 'MAX';
Min: 'MIN';
Sum: 'SUM';
Null: 'NULL';

Identifier: [a-zA-Z_] [a-zA-Z_0-9]*;
Integer: [0-9]+;
String:  '\'' (~'\'')* '\'';
Float: ('-')? [0-9]+ '.' [0-9]*;
Whitespace: [ \t\n\r]+ -> skip;
Annotation: '-' '-' (~';')+;

program
    : statement* EOF
    ;

statement
    : db_statement ';'
    | table_statement ';'
    | index_statement ';'
    | Annotation ';'
    | Null ';'
    ;

db_statement
    : 'SHOW' 'TABLES'                   # show_tables
	| 'SHOW' 'INDEXES'					# show_indexes
    ;

table_statement
    : 'CREATE' 'TABLE' Identifier '(' field_list ')'                    # create_table
    | 'DROP' 'TABLE' Identifier                                         # drop_table
    | 'DESC' Identifier                                                 # describe_table
    | 'INSERT' 'INTO' Identifier 'VALUES' value_lists                   # insert_into_table
    | 'DELETE' 'FROM' Identifier 'WHERE' where_and_clause               # delete_from_table
    | 'UPDATE' Identifier 'SET' set_clause 'WHERE' where_and_clause     # update_table
    | select_table                                                      # select_table_
    ;

select_table
    : 'SELECT' selectors 'FROM' identifiers ('WHERE' where_and_clause)? ('GROUP' 'BY' column)? ('LIMIT' Integer ('OFFSET' Integer)?)?
    ;

index_statement
    : 'ALTER' 'TABLE' Identifier 'ADD' 'INDEX' '(' identifiers ')'   			# alter_add_index
    | 'ALTER' 'TABLE' Identifier 'DROP' 'INDEX' '(' identifiers ')'             # alter_drop_index
    ;

field_list
    : field (',' field)*
    ;

field
    : Identifier type_                                                           # normal_field
    ;

type_
    : 'INT'
    | 'VARCHAR' '(' Integer ')'
    | 'FLOAT'
    ;

value_lists
    : value_list (',' value_list)*
    ;

value_list
    : '(' value (',' value)* ')'
    ;

value
    : Integer
    | String
    | Float
    | Null
    ;

where_and_clause
    : where_clause ('AND' where_clause)*
    ;

where_clause
    : column operate expression             # where_operator_expression
    ;

column
    : Identifier '.' Identifier
    ;


expression
    : value
    | column
    ;

set_clause
    : Identifier EqualOrAssign value (',' Identifier EqualOrAssign value)*
    ;

selectors
    : '*'
    | selector (',' selector)*
    ;

selector
    : column
    | aggregator '(' column ')'
    | Count '(' '*' ')'
    ;

identifiers
    : Identifier (',' Identifier)*
    ;

operate
    : EqualOrAssign
    | Less
    | LessEqual
    | Greater
    | GreaterEqual
    | NotEqual
    ;


aggregator
    : Count
    | Average
    | Max
    | Min
    | Sum
    ;
