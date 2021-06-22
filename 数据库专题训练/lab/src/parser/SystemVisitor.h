#pragma once

#include "SQLBaseVisitor.h"
#include "antlr4-runtime.h"
#include "system/instance.h"

namespace thdb {

class SystemVisitor : public SQLBaseVisitor {
 public:
  SystemVisitor(Instance *pDB);
  antlrcpp::Any visitProgram(SQLParser::ProgramContext *ctx) override;
  antlrcpp::Any visitStatement(SQLParser::StatementContext *ctx) override;
  antlrcpp::Any visitShow_tables(SQLParser::Show_tablesContext *ctx) override;
  antlrcpp::Any visitShow_indexes(SQLParser::Show_indexesContext *ctx) override;
  antlrcpp::Any visitCreate_table(SQLParser::Create_tableContext *ctx) override;
  antlrcpp::Any visitDrop_table(SQLParser::Drop_tableContext *ctx) override;
  antlrcpp::Any visitInsert_into_table(
      SQLParser::Insert_into_tableContext *ctx) override;
  antlrcpp::Any visitUpdate_table(SQLParser::Update_tableContext *ctx) override;
  antlrcpp::Any visitDelete_from_table(
      SQLParser::Delete_from_tableContext *ctx) override;
  antlrcpp::Any visitSelect_table(SQLParser::Select_tableContext *ctx) override;
  antlrcpp::Any visitDescribe_table(
      SQLParser::Describe_tableContext *ctx) override;

  antlrcpp::Any visitField_list(SQLParser::Field_listContext *ctx) override;
  antlrcpp::Any visitNormal_field(SQLParser::Normal_fieldContext *ctx) override;

  antlrcpp::Any visitIdentifiers(SQLParser::IdentifiersContext *ctx) override;
  antlrcpp::Any visitWhere_and_clause(
      SQLParser::Where_and_clauseContext *ctx) override;
  antlrcpp::Any visitWhere_operator_expression(
      SQLParser::Where_operator_expressionContext *ctx) override;
  antlrcpp::Any visitColumn(SQLParser::ColumnContext *ctx) override;

  antlrcpp::Any visitValue_lists(SQLParser::Value_listsContext *ctx) override;
  antlrcpp::Any visitValue_list(SQLParser::Value_listContext *ctx) override;

  antlrcpp::Any visitSet_clause(SQLParser::Set_clauseContext *ctx) override;

  antlrcpp::Any visitAlter_add_index(
      SQLParser::Alter_add_indexContext *ctx) override;
  antlrcpp::Any visitAlter_drop_index(
      SQLParser::Alter_drop_indexContext *ctx) override;

 private:
  Instance *_pDB;
};

}  // namespace thdb