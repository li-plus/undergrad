
// Generated from .\SQL.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"
#include "SQLParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by SQLParser.
 */
class  SQLVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by SQLParser.
   */
    virtual antlrcpp::Any visitProgram(SQLParser::ProgramContext *context) = 0;

    virtual antlrcpp::Any visitStatement(SQLParser::StatementContext *context) = 0;

    virtual antlrcpp::Any visitShow_tables(SQLParser::Show_tablesContext *context) = 0;

    virtual antlrcpp::Any visitShow_indexes(SQLParser::Show_indexesContext *context) = 0;

    virtual antlrcpp::Any visitCreate_table(SQLParser::Create_tableContext *context) = 0;

    virtual antlrcpp::Any visitDrop_table(SQLParser::Drop_tableContext *context) = 0;

    virtual antlrcpp::Any visitDescribe_table(SQLParser::Describe_tableContext *context) = 0;

    virtual antlrcpp::Any visitInsert_into_table(SQLParser::Insert_into_tableContext *context) = 0;

    virtual antlrcpp::Any visitDelete_from_table(SQLParser::Delete_from_tableContext *context) = 0;

    virtual antlrcpp::Any visitUpdate_table(SQLParser::Update_tableContext *context) = 0;

    virtual antlrcpp::Any visitSelect_table_(SQLParser::Select_table_Context *context) = 0;

    virtual antlrcpp::Any visitSelect_table(SQLParser::Select_tableContext *context) = 0;

    virtual antlrcpp::Any visitAlter_add_index(SQLParser::Alter_add_indexContext *context) = 0;

    virtual antlrcpp::Any visitAlter_drop_index(SQLParser::Alter_drop_indexContext *context) = 0;

    virtual antlrcpp::Any visitField_list(SQLParser::Field_listContext *context) = 0;

    virtual antlrcpp::Any visitNormal_field(SQLParser::Normal_fieldContext *context) = 0;

    virtual antlrcpp::Any visitType_(SQLParser::Type_Context *context) = 0;

    virtual antlrcpp::Any visitValue_lists(SQLParser::Value_listsContext *context) = 0;

    virtual antlrcpp::Any visitValue_list(SQLParser::Value_listContext *context) = 0;

    virtual antlrcpp::Any visitValue(SQLParser::ValueContext *context) = 0;

    virtual antlrcpp::Any visitWhere_and_clause(SQLParser::Where_and_clauseContext *context) = 0;

    virtual antlrcpp::Any visitWhere_operator_expression(SQLParser::Where_operator_expressionContext *context) = 0;

    virtual antlrcpp::Any visitColumn(SQLParser::ColumnContext *context) = 0;

    virtual antlrcpp::Any visitExpression(SQLParser::ExpressionContext *context) = 0;

    virtual antlrcpp::Any visitSet_clause(SQLParser::Set_clauseContext *context) = 0;

    virtual antlrcpp::Any visitSelectors(SQLParser::SelectorsContext *context) = 0;

    virtual antlrcpp::Any visitSelector(SQLParser::SelectorContext *context) = 0;

    virtual antlrcpp::Any visitIdentifiers(SQLParser::IdentifiersContext *context) = 0;

    virtual antlrcpp::Any visitOperate(SQLParser::OperateContext *context) = 0;

    virtual antlrcpp::Any visitAggregator(SQLParser::AggregatorContext *context) = 0;


};

