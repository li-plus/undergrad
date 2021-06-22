
// Generated from .\SQL.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"
#include "SQLVisitor.h"


/**
 * This class provides an empty implementation of SQLVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  SQLBaseVisitor : public SQLVisitor {
public:

  virtual antlrcpp::Any visitProgram(SQLParser::ProgramContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStatement(SQLParser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitShow_tables(SQLParser::Show_tablesContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitShow_indexes(SQLParser::Show_indexesContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCreate_table(SQLParser::Create_tableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDrop_table(SQLParser::Drop_tableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDescribe_table(SQLParser::Describe_tableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitInsert_into_table(SQLParser::Insert_into_tableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDelete_from_table(SQLParser::Delete_from_tableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUpdate_table(SQLParser::Update_tableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSelect_table_(SQLParser::Select_table_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSelect_table(SQLParser::Select_tableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAlter_add_index(SQLParser::Alter_add_indexContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAlter_drop_index(SQLParser::Alter_drop_indexContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitField_list(SQLParser::Field_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNormal_field(SQLParser::Normal_fieldContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitType_(SQLParser::Type_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitValue_lists(SQLParser::Value_listsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitValue_list(SQLParser::Value_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitValue(SQLParser::ValueContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWhere_and_clause(SQLParser::Where_and_clauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWhere_operator_expression(SQLParser::Where_operator_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitColumn(SQLParser::ColumnContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpression(SQLParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSet_clause(SQLParser::Set_clauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSelectors(SQLParser::SelectorsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSelector(SQLParser::SelectorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdentifiers(SQLParser::IdentifiersContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitOperate(SQLParser::OperateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAggregator(SQLParser::AggregatorContext *ctx) override {
    return visitChildren(ctx);
  }


};

