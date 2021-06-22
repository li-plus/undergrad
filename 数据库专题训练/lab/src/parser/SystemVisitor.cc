#include "SystemVisitor.h"

#include <float.h>
#include <stdlib.h>

#include "condition/conditions.h"
#include "exception/exceptions.h"
#include "record/fixed_record.h"
#include "record/transform.h"
#include "result/result.h"
#include "table/schema.h"
#include "table/table.h"

#define EPOSILO 1e-5

namespace thdb {

SystemVisitor::SystemVisitor(Instance *pDB) : _pDB{pDB} { assert(_pDB); }

antlrcpp::Any SystemVisitor::visitProgram(SQLParser::ProgramContext *ctx) {
  std::vector<Result *> iResVec;
  for (const auto &it : ctx->statement()) {
    if (it->Null() || it->Annotation()) continue;
    Result *res = it->accept(this);
    iResVec.push_back(res);
  }
  return iResVec;
}

antlrcpp::Any SystemVisitor::visitStatement(SQLParser::StatementContext *ctx) {
  Result *res = nullptr;
  if (ctx->db_statement())
    res = ctx->db_statement()->accept(this);
  else if (ctx->table_statement())
    res = ctx->table_statement()->accept(this);
  else if (ctx->index_statement())
    res = ctx->index_statement()->accept(this);
  else {
    printf("%s\n", ctx->getText().c_str());
    throw SpecialException();
  }
  return res;
}

antlrcpp::Any SystemVisitor::visitShow_tables(
    SQLParser::Show_tablesContext *ctx) {
  Result *res = new MemResult({"Show Tables"});
  // TODO: Add Some info
  for (const auto &sTableName : _pDB->GetTableNames()) {
    FixedRecord *pRes =
        new FixedRecord(1, {FieldType::STRING_TYPE}, {TABLE_NAME_SIZE});
    pRes->SetField(0, new StringField(sTableName));
    res->PushBack(pRes);
  }
  return res;
}

antlrcpp::Any SystemVisitor::visitShow_indexes(
    SQLParser::Show_indexesContext *ctx) {
  Result *res = new MemResult(
      {"Table Name", "Column Name", "Column Type", "Column Size"});
  // TODO: Add Some info
  for (const auto &pRecord : _pDB->GetIndexInfos()) {
    res->PushBack(pRecord);
  }
  return res;
}

antlrcpp::Any SystemVisitor::visitCreate_table(
    SQLParser::Create_tableContext *ctx) {
  Size nSize = 0;
  try {
    Schema iSchema = ctx->field_list()->accept(this);
    String sTableName = ctx->Identifier()->getText();
    _pDB->CreateTable(sTableName, iSchema);
    nSize = 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  Result *res = new MemResult({"Create Table"});
  FixedRecord *pRes = new FixedRecord(1, {FieldType::INT_TYPE}, {4});
  pRes->SetField(0, new IntField(nSize));
  res->PushBack(pRes);
  return res;
}

antlrcpp::Any SystemVisitor::visitDrop_table(
    SQLParser::Drop_tableContext *ctx) {
  Size nSize = 0;
  try {
    String sTableName = ctx->Identifier()->getText();
    _pDB->DropTable(sTableName);
    nSize = 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  Result *res = new MemResult({"Drop Table"});
  FixedRecord *pRes = new FixedRecord(1, {FieldType::INT_TYPE}, {4});
  pRes->SetField(0, new IntField(nSize));
  res->PushBack(pRes);
  return res;
}

antlrcpp::Any SystemVisitor::visitField_list(
    SQLParser::Field_listContext *ctx) {
  std::vector<Column> iColVec;
  for (const auto &it : ctx->field()) {
    iColVec.push_back(it->accept(this));
  }
  return Schema(iColVec);
}

antlrcpp::Any SystemVisitor::visitNormal_field(
    SQLParser::Normal_fieldContext *ctx) {
  String sType = ctx->type_()->getText();
  if (sType == "INT") {
    return Column(ctx->Identifier()->getText(), FieldType::INT_TYPE);
  } else if (sType == "FLOAT") {
    return Column(ctx->Identifier()->getText(), FieldType::FLOAT_TYPE);
  } else {
    int nSize = atoi(ctx->type_()->Integer()->getText().c_str());
    return Column(ctx->Identifier()->getText(), FieldType::STRING_TYPE, nSize);
  }
}

antlrcpp::Any SystemVisitor::visitDescribe_table(
    SQLParser::Describe_tableContext *ctx) {
  Result *res = new MemResult({"Column Name", "Column Type", "Column Size"});
  String sTableName = ctx->Identifier()->getText();
  for (const auto &pRecord : _pDB->GetTableInfos(sTableName))
    res->PushBack(pRecord);
  return res;
}

antlrcpp::Any SystemVisitor::visitDelete_from_table(
    SQLParser::Delete_from_tableContext *ctx) {
  String sTableName = ctx->Identifier()->getText();
  std::map<String, std::vector<Condition *>> iMap =
      ctx->where_and_clause()->accept(this);
  assert(iMap.size() == 1);
  std::vector<Condition *> iIndexCond{};
  std::vector<Condition *> iOtherCond{};
  for (const auto &pCond : iMap[sTableName])
    if (pCond->GetType() == ConditionType::INDEX_TYPE)
      iIndexCond.push_back(pCond);
    else
      iOtherCond.push_back(pCond);
  Condition *pCond = nullptr;
  if (iOtherCond.size() > 0) pCond = new AndCondition(iOtherCond);
  Size nSize = _pDB->Delete(sTableName, pCond, iIndexCond);
  // TODO: Clear Condition
  if (pCond) delete pCond;
  for (const auto &it : iIndexCond)
    if (it) delete it;

  Result *res = new MemResult({"Delete"});
  FixedRecord *pRes = new FixedRecord(1, {FieldType::INT_TYPE}, {4});
  pRes->SetField(0, new IntField(nSize));
  res->PushBack(pRes);
  return res;
}

antlrcpp::Any SystemVisitor::visitSet_clause(
    SQLParser::Set_clauseContext *ctx) {
  std::vector<std::pair<String, String>> iVec;
  for (Size i = 0; i < ctx->Identifier().size(); ++i) {
    iVec.push_back({ctx->Identifier(i)->getText(), ctx->value(i)->getText()});
  }
  return iVec;
}

antlrcpp::Any SystemVisitor::visitUpdate_table(
    SQLParser::Update_tableContext *ctx) {
  std::vector<std::pair<String, String>> iSetVec =
      ctx->set_clause()->accept(this);
  String sTableName = ctx->Identifier()->getText();
  std::map<String, std::vector<Condition *>> iMap =
      ctx->where_and_clause()->accept(this);
  assert(iMap.size() == 1);
  std::vector<Transform> iTrans{};
  for (Size i = 0; i < iSetVec.size(); ++i) {
    FieldID nFieldID = _pDB->GetColID(sTableName, iSetVec[i].first);
    FieldType iType = _pDB->GetColType(sTableName, iSetVec[i].first);
    iTrans.push_back({nFieldID, iType, iSetVec[i].second});
  }
  std::vector<Condition *> iIndexCond{};
  std::vector<Condition *> iOtherCond{};
  for (const auto &pCond : iMap[sTableName])
    if (pCond->GetType() == ConditionType::INDEX_TYPE)
      iIndexCond.push_back(pCond);
    else
      iOtherCond.push_back(pCond);
  Condition *pCond = nullptr;
  if (iOtherCond.size() > 0) pCond = new AndCondition(iOtherCond);
  Size nSize = _pDB->Update(sTableName, pCond, iIndexCond, iTrans);
  // TODO: Clear Condition
  if (pCond) delete pCond;
  for (const auto &it : iIndexCond)
    if (it) delete it;

  Result *res = new MemResult({"Update"});
  FixedRecord *pRes = new FixedRecord(1, {FieldType::INT_TYPE}, {4});
  pRes->SetField(0, new IntField(nSize));
  res->PushBack(pRes);
  return res;
}

antlrcpp::Any SystemVisitor::visitInsert_into_table(
    SQLParser::Insert_into_tableContext *ctx) {
  std::vector<std::vector<String>> iValueListVec =
      ctx->value_lists()->accept(this);
  String sTableName = ctx->Identifier()->getText();
  for (const auto &iValueList : iValueListVec) {
    _pDB->Insert(sTableName, iValueList);
  }
  Result *res = new MemResult({"Insert"});
  FixedRecord *pRes = new FixedRecord(1, {FieldType::INT_TYPE}, {4});
  pRes->SetField(0, new IntField(iValueListVec.size()));
  res->PushBack(pRes);
  return res;
}

antlrcpp::Any SystemVisitor::visitSelect_table(
    SQLParser::Select_tableContext *ctx) {
  std::vector<String> iTableNameVec = ctx->identifiers()->accept(this);
  std::map<String, std::vector<PageSlotID>> iResultMap{};
  std::map<String, std::vector<Condition *>> iCondMap{};
  // TODO: Filter
  if (ctx->where_and_clause()) {
    std::map<String, std::vector<Condition *>> iTempMap =
        ctx->where_and_clause()->accept(this);
    iCondMap = iTempMap;
  }

  for (const auto &sTableName : iTableNameVec) {
    if (iCondMap.find(sTableName) == iCondMap.end())
      iResultMap[sTableName] = _pDB->Search(sTableName, nullptr, {});
    else {
      std::vector<Condition *> iIndexCond{};
      std::vector<Condition *> iOtherCond{};
      for (const auto &pCond : iCondMap[sTableName])
        if (pCond->GetType() == ConditionType::INDEX_TYPE)
          iIndexCond.push_back(pCond);
        else
          iOtherCond.push_back(pCond);
      Condition *pCond = nullptr;
      if (iOtherCond.size() > 0) pCond = new AndCondition(iOtherCond);
      iResultMap[sTableName] = _pDB->Search(sTableName, pCond, iIndexCond);
      if (pCond) delete pCond;
      for (const auto &it : iIndexCond)
        if (it) delete it;
    }
  }

  // TODO: Join
  bool bJoin = (iResultMap.size() > 1);
  std::vector<Condition *> iJoinConds = {};
  if (iCondMap.find("JOIN") != iCondMap.end()) {
    iJoinConds = iCondMap.find("JOIN")->second;
  }
  std::pair<std::vector<String>, std::vector<Record *>> iHeadDataPair{};
  if (bJoin) iHeadDataPair = _pDB->Join(iResultMap, iJoinConds);

  // TODO: Generate Result
  std::vector<PageSlotID> iData;
  if (!bJoin) {
    String sTableName = iTableNameVec[0];
    iData = iResultMap[sTableName];
    Result *pResult = new MemResult(_pDB->GetColumnNames(sTableName));
    for (const auto &it : iData)
      pResult->PushBack(_pDB->GetRecord(iTableNameVec[0], it));
    return pResult;
  } else {
    Result *pResult = new MemResult(iHeadDataPair.first);
    for (const auto &pRecord : iHeadDataPair.second) pResult->PushBack(pRecord);
    return pResult;
  }
}

antlrcpp::Any SystemVisitor::visitWhere_and_clause(
    SQLParser::Where_and_clauseContext *ctx) {
  std::map<String, std::vector<Condition *>> iCondMap;
  for (const auto &it : ctx->where_clause()) {
    std::pair<String, Condition *> iCondPair = it->accept(this);
    // TODO: JOIN CONDITION
    if (iCondPair.second->GetType() == ConditionType::JOIN_TYPE) {
    }
    // Not Join Condition
    if (iCondMap.find(iCondPair.first) == iCondMap.end()) {
      iCondMap[iCondPair.first] = {};
    }
    iCondMap[iCondPair.first].push_back(iCondPair.second);
  }
  return iCondMap;
}

antlrcpp::Any SystemVisitor::visitIdentifiers(
    SQLParser::IdentifiersContext *ctx) {
  std::vector<String> iStringVec;
  for (const auto &it : ctx->Identifier()) {
    iStringVec.push_back(it->getText());
  }
  return iStringVec;
}

antlrcpp::Any SystemVisitor::visitWhere_operator_expression(
    SQLParser::Where_operator_expressionContext *ctx) {
  std::pair<String, String> iPair = ctx->column()->accept(this);
  FieldID nColIndex = _pDB->GetColID(iPair.first, iPair.second);
  if (ctx->expression()->column()) {
    // JOIN CONDITION
    std::pair<String, String> iPairB =
        ctx->expression()->column()->accept(this);
    return std::pair<String, Condition *>(
        "JOIN", new JoinCondition(iPair.first, iPair.second, iPairB.first,
                                  iPairB.second));
  }
  if (_pDB->IsIndex(iPair.first, iPair.second)) {
    double fValue = stod(ctx->expression()->value()->getText());
    FieldType iType = _pDB->GetColType(iPair.first, iPair.second);
    if (ctx->children[1]->getText() == "<") {
      return std::pair<String, Condition *>(
          iPair.first, new IndexCondition(iPair.first, iPair.second, DBL_MIN,
                                          fValue, iType));
    } else if (ctx->children[1]->getText() == ">") {
      return std::pair<String, Condition *>(
          iPair.first, new IndexCondition(iPair.first, iPair.second,
                                          fValue + EPOSILO, DBL_MAX, iType));
    } else if (ctx->children[1]->getText() == "=") {
      return std::pair<String, Condition *>(
          iPair.first, new IndexCondition(iPair.first, iPair.second, fValue,
                                          fValue + EPOSILO, iType));
    } else if (ctx->children[1]->getText() == "<=") {
      return std::pair<String, Condition *>(
          iPair.first, new IndexCondition(iPair.first, iPair.second, DBL_MIN,
                                          fValue + EPOSILO, iType));
    } else if (ctx->children[1]->getText() == ">=") {
      return std::pair<String, Condition *>(
          iPair.first, new IndexCondition(iPair.first, iPair.second, fValue,
                                          DBL_MAX, iType));
    } else if (ctx->children[1]->getText() == "<>") {
      return std::pair<String, Condition *>(
          iPair.first, new NotCondition(new RangeCondition(nColIndex, fValue,
                                                           fValue + EPOSILO)));
    } else {
      throw SpecialException();
    }
  } else {
    double fValue = stod(ctx->expression()->value()->getText());
    if (ctx->children[1]->getText() == "<") {
      return std::pair<String, Condition *>(
          iPair.first, new RangeCondition(nColIndex, DBL_MIN, fValue));
    } else if (ctx->children[1]->getText() == ">") {
      return std::pair<String, Condition *>(
          iPair.first,
          new RangeCondition(nColIndex, fValue + EPOSILO, DBL_MAX));
    } else if (ctx->children[1]->getText() == "=") {
      return std::pair<String, Condition *>(
          iPair.first, new RangeCondition(nColIndex, fValue, fValue + EPOSILO));
    } else if (ctx->children[1]->getText() == "<=") {
      return std::pair<String, Condition *>(
          iPair.first,
          new RangeCondition(nColIndex, DBL_MIN, fValue + EPOSILO));
    } else if (ctx->children[1]->getText() == ">=") {
      return std::pair<String, Condition *>(
          iPair.first, new RangeCondition(nColIndex, fValue, DBL_MAX));
    } else if (ctx->children[1]->getText() == "<>") {
      return std::pair<String, Condition *>(
          iPair.first, new NotCondition(new RangeCondition(nColIndex, fValue,
                                                           fValue + EPOSILO)));
    } else {
      throw SpecialException();
    }
  }
}

antlrcpp::Any SystemVisitor::visitColumn(SQLParser::ColumnContext *ctx) {
  String sTableName = ctx->Identifier(0)->getText();
  String sColumnName = ctx->Identifier(1)->getText();
  return std::pair<String, String>(sTableName, sColumnName);
}

antlrcpp::Any SystemVisitor::visitValue_lists(
    SQLParser::Value_listsContext *ctx) {
  std::vector<std::vector<String>> iValueListVec;
  for (const auto &it : ctx->value_list())
    iValueListVec.push_back(it->accept(this));
  return iValueListVec;
}

antlrcpp::Any SystemVisitor::visitValue_list(
    SQLParser::Value_listContext *ctx) {
  std::vector<String> iValueList;
  for (const auto &it : ctx->value()) iValueList.push_back(it->getText());
  return iValueList;
}

antlrcpp::Any SystemVisitor::visitAlter_add_index(
    SQLParser::Alter_add_indexContext *ctx) {
  String sTableName = ctx->Identifier()->getText();
  std::vector<String> iColNameVec = ctx->identifiers()->accept(this);
  Size nSize = 0;
  for (const auto &sColName : iColNameVec) {
    try {
      FieldType iType = _pDB->GetColType(sTableName, sColName);
      _pDB->CreateIndex(sTableName, sColName, iType);
      ++nSize;
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
  }
  Result *res = new MemResult({"Create Index"});
  FixedRecord *pRes = new FixedRecord(1, {FieldType::INT_TYPE}, {4});
  pRes->SetField(0, new IntField(nSize));
  res->PushBack(pRes);
  return res;
}

antlrcpp::Any SystemVisitor::visitAlter_drop_index(
    SQLParser::Alter_drop_indexContext *ctx) {
  String sTableName = ctx->Identifier()->getText();
  std::vector<String> iColNameVec = ctx->identifiers()->accept(this);
  Size nSize = 0;
  for (const auto &sColName : iColNameVec) {
    try {
      _pDB->DropIndex(sTableName, sColName);
      ++nSize;
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
  }
  Result *res = new MemResult({"Drop Index"});
  FixedRecord *pRes = new FixedRecord(1, {FieldType::INT_TYPE}, {4});
  pRes->SetField(0, new IntField(nSize));
  res->PushBack(pRes);
  return res;
}

}  // namespace thdb
