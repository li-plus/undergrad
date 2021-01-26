    /* case insensitive keywords */
%option caseless
    /* avoid defining yywrap() function */
%option noyywrap
    /* avoid defining yyunput() function */
%option nounput
    /* avoid defining input() function */
%option noinput

%{
#include "ast.h"
#include "yacc.tab.h"
%}

alpha [a-zA-Z]
digit [0-9]
white_space [ \t]+
new_line \r|\n|\r\n
sign "+"|"-"
identifier {alpha}(_|{alpha}|{digit})*
value_int {sign}?{digit}+
value_float {sign}?{digit}+\.({digit}+)?
value_string '[^']*'
single_op ";"|"("|")"|","|"*"|"="|">"|"<"|"."

%x comment

%%
    /* block comment */
"/*" { BEGIN(comment); }
<comment>"*/" { BEGIN(INITIAL); }
<comment>[^*] { /* ignore the text of the comment */ }
<comment>\* { /* ignore *'s that aren't part of */ }
    /* single line comment */
"--".* { /* ignore single line comment */ }
    /* white space and new line */
{white_space} { /* ignore white space */ }
{new_line} { /* ignore new line */ }
    /* keywords */
"DATABASE" { return DATABASE; }
"DATABASES" { return DATABASES; }
"TABLE" { return TABLE; }
"TABLES" { return TABLES; }
"SHOW" { return SHOW; }
"CREATE" { return CREATE; }
"DROP" { return DROP; }
"USE" { return USE; }
"PRIMARY" { return PRIMARY; }
"KEY" { return KEY; }
"NOT" { return NOT; }
"NULL" { return SQLNULL; }
"INSERT" { return INSERT; }
"INTO" { return INTO; }
"VALUES" { return VALUES; }
"DELETE" { return DELETE; }
"FROM" { return FROM; }
"WHERE" { return WHERE; }
"UPDATE" { return UPDATE; }
"SET" { return SET; }
"SELECT" { return SELECT; }
"IS" { return IS; }
"INT" { return INT; }
"VARCHAR" { return VARCHAR; }
"DEFAULT" { return DEFAULT; }
"CONSTRAINT" { return CONSTRAINT; }
"CHANGE" { return CHANGE; }
"ALTER" { return ALTER; }
"ADD" { return ADD; }
"RENAME" { return RENAME; }
"DESC" { return DESC; }
"INDEX" { return INDEX; }
"AND" { return AND; }
"DATE" { return DATE; }
"FLOAT" { return FLOAT; }
"FOREIGN" { return FOREIGN; }
"REFERENCES" { return REFERENCES; }
"ON" { return ON; }
"TO" { return TO; }
    /* operators */
">=" { return GEQ; }
"<=" { return LEQ; }
"<>" { return NEQ; }
{single_op} { return yytext[0]; }
    /* id */
{identifier} {
    yylval.sv_str = alloc_str(strlen(yytext) + 1);
    strcpy(yylval.sv_str, yytext);
    return IDENTIFIER;
}
    /* literals */
{value_int} {
    yylval.sv_int = atoi(yytext);
    return VALUE_INT;
}
{value_float} {
    yylval.sv_float = atof(yytext);
    return VALUE_FLOAT;
}
{value_string} {
    int len = strlen(yytext);
    yylval.sv_str = alloc_str(len - 1);
    sprintf(yylval.sv_str, "%.*s", len - 2, yytext + 1);
    return VALUE_STRING;
}
    /* EOF */
<<EOF>> { return T_EOF; }
    /* unexpected char */
. { fprintf(stderr, "Lexer error: unexpected character %c\n", yytext[0]); }
%%
