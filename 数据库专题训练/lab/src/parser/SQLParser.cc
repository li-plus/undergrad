
// Generated from .\SQL.g4 by ANTLR 4.7


#include "SQLVisitor.h"

#include "SQLParser.h"


using namespace antlrcpp;
using namespace antlr4;

SQLParser::SQLParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

SQLParser::~SQLParser() {
  delete _interpreter;
}

std::string SQLParser::getGrammarFileName() const {
  return "SQL.g4";
}

const std::vector<std::string>& SQLParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& SQLParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- ProgramContext ------------------------------------------------------------------

SQLParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SQLParser::ProgramContext::EOF() {
  return getToken(SQLParser::EOF, 0);
}

std::vector<SQLParser::StatementContext *> SQLParser::ProgramContext::statement() {
  return getRuleContexts<SQLParser::StatementContext>();
}

SQLParser::StatementContext* SQLParser::ProgramContext::statement(size_t i) {
  return getRuleContext<SQLParser::StatementContext>(i);
}


size_t SQLParser::ProgramContext::getRuleIndex() const {
  return SQLParser::RuleProgram;
}

antlrcpp::Any SQLParser::ProgramContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitProgram(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::ProgramContext* SQLParser::program() {
  ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, getState());
  enterRule(_localctx, 0, SQLParser::RuleProgram);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(47);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SQLParser::T__1)
      | (1ULL << SQLParser::T__4)
      | (1ULL << SQLParser::T__8)
      | (1ULL << SQLParser::T__9)
      | (1ULL << SQLParser::T__10)
      | (1ULL << SQLParser::T__13)
      | (1ULL << SQLParser::T__16)
      | (1ULL << SQLParser::T__18)
      | (1ULL << SQLParser::T__23)
      | (1ULL << SQLParser::Null)
      | (1ULL << SQLParser::Annotation))) != 0)) {
      setState(44);
      statement();
      setState(49);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(50);
    match(SQLParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

SQLParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SQLParser::Db_statementContext* SQLParser::StatementContext::db_statement() {
  return getRuleContext<SQLParser::Db_statementContext>(0);
}

SQLParser::Table_statementContext* SQLParser::StatementContext::table_statement() {
  return getRuleContext<SQLParser::Table_statementContext>(0);
}

SQLParser::Index_statementContext* SQLParser::StatementContext::index_statement() {
  return getRuleContext<SQLParser::Index_statementContext>(0);
}

tree::TerminalNode* SQLParser::StatementContext::Annotation() {
  return getToken(SQLParser::Annotation, 0);
}

tree::TerminalNode* SQLParser::StatementContext::Null() {
  return getToken(SQLParser::Null, 0);
}


size_t SQLParser::StatementContext::getRuleIndex() const {
  return SQLParser::RuleStatement;
}

antlrcpp::Any SQLParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::StatementContext* SQLParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 2, SQLParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(65);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SQLParser::T__1: {
        enterOuterAlt(_localctx, 1);
        setState(52);
        db_statement();
        setState(53);
        match(SQLParser::T__0);
        break;
      }

      case SQLParser::T__4:
      case SQLParser::T__8:
      case SQLParser::T__9:
      case SQLParser::T__10:
      case SQLParser::T__13:
      case SQLParser::T__16:
      case SQLParser::T__18: {
        enterOuterAlt(_localctx, 2);
        setState(55);
        table_statement();
        setState(56);
        match(SQLParser::T__0);
        break;
      }

      case SQLParser::T__23: {
        enterOuterAlt(_localctx, 3);
        setState(58);
        index_statement();
        setState(59);
        match(SQLParser::T__0);
        break;
      }

      case SQLParser::Annotation: {
        enterOuterAlt(_localctx, 4);
        setState(61);
        match(SQLParser::Annotation);
        setState(62);
        match(SQLParser::T__0);
        break;
      }

      case SQLParser::Null: {
        enterOuterAlt(_localctx, 5);
        setState(63);
        match(SQLParser::Null);
        setState(64);
        match(SQLParser::T__0);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Db_statementContext ------------------------------------------------------------------

SQLParser::Db_statementContext::Db_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SQLParser::Db_statementContext::getRuleIndex() const {
  return SQLParser::RuleDb_statement;
}

void SQLParser::Db_statementContext::copyFrom(Db_statementContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Show_tablesContext ------------------------------------------------------------------

SQLParser::Show_tablesContext::Show_tablesContext(Db_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Show_tablesContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitShow_tables(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Show_indexesContext ------------------------------------------------------------------

SQLParser::Show_indexesContext::Show_indexesContext(Db_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Show_indexesContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitShow_indexes(this);
  else
    return visitor->visitChildren(this);
}
SQLParser::Db_statementContext* SQLParser::db_statement() {
  Db_statementContext *_localctx = _tracker.createInstance<Db_statementContext>(_ctx, getState());
  enterRule(_localctx, 4, SQLParser::RuleDb_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(71);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<Db_statementContext *>(_tracker.createInstance<SQLParser::Show_tablesContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(67);
      match(SQLParser::T__1);
      setState(68);
      match(SQLParser::T__2);
      break;
    }

    case 2: {
      _localctx = dynamic_cast<Db_statementContext *>(_tracker.createInstance<SQLParser::Show_indexesContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(69);
      match(SQLParser::T__1);
      setState(70);
      match(SQLParser::T__3);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Table_statementContext ------------------------------------------------------------------

SQLParser::Table_statementContext::Table_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SQLParser::Table_statementContext::getRuleIndex() const {
  return SQLParser::RuleTable_statement;
}

void SQLParser::Table_statementContext::copyFrom(Table_statementContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Delete_from_tableContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Delete_from_tableContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::Where_and_clauseContext* SQLParser::Delete_from_tableContext::where_and_clause() {
  return getRuleContext<SQLParser::Where_and_clauseContext>(0);
}

SQLParser::Delete_from_tableContext::Delete_from_tableContext(Table_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Delete_from_tableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitDelete_from_table(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Insert_into_tableContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Insert_into_tableContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::Value_listsContext* SQLParser::Insert_into_tableContext::value_lists() {
  return getRuleContext<SQLParser::Value_listsContext>(0);
}

SQLParser::Insert_into_tableContext::Insert_into_tableContext(Table_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Insert_into_tableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitInsert_into_table(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Create_tableContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Create_tableContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::Field_listContext* SQLParser::Create_tableContext::field_list() {
  return getRuleContext<SQLParser::Field_listContext>(0);
}

SQLParser::Create_tableContext::Create_tableContext(Table_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Create_tableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitCreate_table(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Describe_tableContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Describe_tableContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::Describe_tableContext::Describe_tableContext(Table_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Describe_tableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitDescribe_table(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Select_table_Context ------------------------------------------------------------------

SQLParser::Select_tableContext* SQLParser::Select_table_Context::select_table() {
  return getRuleContext<SQLParser::Select_tableContext>(0);
}

SQLParser::Select_table_Context::Select_table_Context(Table_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Select_table_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitSelect_table_(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Drop_tableContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Drop_tableContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::Drop_tableContext::Drop_tableContext(Table_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Drop_tableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitDrop_table(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Update_tableContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Update_tableContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::Set_clauseContext* SQLParser::Update_tableContext::set_clause() {
  return getRuleContext<SQLParser::Set_clauseContext>(0);
}

SQLParser::Where_and_clauseContext* SQLParser::Update_tableContext::where_and_clause() {
  return getRuleContext<SQLParser::Where_and_clauseContext>(0);
}

SQLParser::Update_tableContext::Update_tableContext(Table_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Update_tableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitUpdate_table(this);
  else
    return visitor->visitChildren(this);
}
SQLParser::Table_statementContext* SQLParser::table_statement() {
  Table_statementContext *_localctx = _tracker.createInstance<Table_statementContext>(_ctx, getState());
  enterRule(_localctx, 6, SQLParser::RuleTable_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(103);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SQLParser::T__4: {
        _localctx = dynamic_cast<Table_statementContext *>(_tracker.createInstance<SQLParser::Create_tableContext>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(73);
        match(SQLParser::T__4);
        setState(74);
        match(SQLParser::T__5);
        setState(75);
        match(SQLParser::Identifier);
        setState(76);
        match(SQLParser::T__6);
        setState(77);
        field_list();
        setState(78);
        match(SQLParser::T__7);
        break;
      }

      case SQLParser::T__8: {
        _localctx = dynamic_cast<Table_statementContext *>(_tracker.createInstance<SQLParser::Drop_tableContext>(_localctx));
        enterOuterAlt(_localctx, 2);
        setState(80);
        match(SQLParser::T__8);
        setState(81);
        match(SQLParser::T__5);
        setState(82);
        match(SQLParser::Identifier);
        break;
      }

      case SQLParser::T__9: {
        _localctx = dynamic_cast<Table_statementContext *>(_tracker.createInstance<SQLParser::Describe_tableContext>(_localctx));
        enterOuterAlt(_localctx, 3);
        setState(83);
        match(SQLParser::T__9);
        setState(84);
        match(SQLParser::Identifier);
        break;
      }

      case SQLParser::T__10: {
        _localctx = dynamic_cast<Table_statementContext *>(_tracker.createInstance<SQLParser::Insert_into_tableContext>(_localctx));
        enterOuterAlt(_localctx, 4);
        setState(85);
        match(SQLParser::T__10);
        setState(86);
        match(SQLParser::T__11);
        setState(87);
        match(SQLParser::Identifier);
        setState(88);
        match(SQLParser::T__12);
        setState(89);
        value_lists();
        break;
      }

      case SQLParser::T__13: {
        _localctx = dynamic_cast<Table_statementContext *>(_tracker.createInstance<SQLParser::Delete_from_tableContext>(_localctx));
        enterOuterAlt(_localctx, 5);
        setState(90);
        match(SQLParser::T__13);
        setState(91);
        match(SQLParser::T__14);
        setState(92);
        match(SQLParser::Identifier);
        setState(93);
        match(SQLParser::T__15);
        setState(94);
        where_and_clause();
        break;
      }

      case SQLParser::T__16: {
        _localctx = dynamic_cast<Table_statementContext *>(_tracker.createInstance<SQLParser::Update_tableContext>(_localctx));
        enterOuterAlt(_localctx, 6);
        setState(95);
        match(SQLParser::T__16);
        setState(96);
        match(SQLParser::Identifier);
        setState(97);
        match(SQLParser::T__17);
        setState(98);
        set_clause();
        setState(99);
        match(SQLParser::T__15);
        setState(100);
        where_and_clause();
        break;
      }

      case SQLParser::T__18: {
        _localctx = dynamic_cast<Table_statementContext *>(_tracker.createInstance<SQLParser::Select_table_Context>(_localctx));
        enterOuterAlt(_localctx, 7);
        setState(102);
        select_table();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Select_tableContext ------------------------------------------------------------------

SQLParser::Select_tableContext::Select_tableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SQLParser::SelectorsContext* SQLParser::Select_tableContext::selectors() {
  return getRuleContext<SQLParser::SelectorsContext>(0);
}

SQLParser::IdentifiersContext* SQLParser::Select_tableContext::identifiers() {
  return getRuleContext<SQLParser::IdentifiersContext>(0);
}

SQLParser::Where_and_clauseContext* SQLParser::Select_tableContext::where_and_clause() {
  return getRuleContext<SQLParser::Where_and_clauseContext>(0);
}

SQLParser::ColumnContext* SQLParser::Select_tableContext::column() {
  return getRuleContext<SQLParser::ColumnContext>(0);
}

std::vector<tree::TerminalNode *> SQLParser::Select_tableContext::Integer() {
  return getTokens(SQLParser::Integer);
}

tree::TerminalNode* SQLParser::Select_tableContext::Integer(size_t i) {
  return getToken(SQLParser::Integer, i);
}


size_t SQLParser::Select_tableContext::getRuleIndex() const {
  return SQLParser::RuleSelect_table;
}

antlrcpp::Any SQLParser::Select_tableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitSelect_table(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::Select_tableContext* SQLParser::select_table() {
  Select_tableContext *_localctx = _tracker.createInstance<Select_tableContext>(_ctx, getState());
  enterRule(_localctx, 8, SQLParser::RuleSelect_table);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(105);
    match(SQLParser::T__18);
    setState(106);
    selectors();
    setState(107);
    match(SQLParser::T__14);
    setState(108);
    identifiers();
    setState(111);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SQLParser::T__15) {
      setState(109);
      match(SQLParser::T__15);
      setState(110);
      where_and_clause();
    }
    setState(116);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SQLParser::T__19) {
      setState(113);
      match(SQLParser::T__19);
      setState(114);
      match(SQLParser::T__20);
      setState(115);
      column();
    }
    setState(124);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SQLParser::T__21) {
      setState(118);
      match(SQLParser::T__21);
      setState(119);
      match(SQLParser::Integer);
      setState(122);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SQLParser::T__22) {
        setState(120);
        match(SQLParser::T__22);
        setState(121);
        match(SQLParser::Integer);
      }
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Index_statementContext ------------------------------------------------------------------

SQLParser::Index_statementContext::Index_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SQLParser::Index_statementContext::getRuleIndex() const {
  return SQLParser::RuleIndex_statement;
}

void SQLParser::Index_statementContext::copyFrom(Index_statementContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Alter_drop_indexContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Alter_drop_indexContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::IdentifiersContext* SQLParser::Alter_drop_indexContext::identifiers() {
  return getRuleContext<SQLParser::IdentifiersContext>(0);
}

SQLParser::Alter_drop_indexContext::Alter_drop_indexContext(Index_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Alter_drop_indexContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitAlter_drop_index(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Alter_add_indexContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Alter_add_indexContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::IdentifiersContext* SQLParser::Alter_add_indexContext::identifiers() {
  return getRuleContext<SQLParser::IdentifiersContext>(0);
}

SQLParser::Alter_add_indexContext::Alter_add_indexContext(Index_statementContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Alter_add_indexContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitAlter_add_index(this);
  else
    return visitor->visitChildren(this);
}
SQLParser::Index_statementContext* SQLParser::index_statement() {
  Index_statementContext *_localctx = _tracker.createInstance<Index_statementContext>(_ctx, getState());
  enterRule(_localctx, 10, SQLParser::RuleIndex_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(144);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<Index_statementContext *>(_tracker.createInstance<SQLParser::Alter_add_indexContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(126);
      match(SQLParser::T__23);
      setState(127);
      match(SQLParser::T__5);
      setState(128);
      match(SQLParser::Identifier);
      setState(129);
      match(SQLParser::T__24);
      setState(130);
      match(SQLParser::T__25);
      setState(131);
      match(SQLParser::T__6);
      setState(132);
      identifiers();
      setState(133);
      match(SQLParser::T__7);
      break;
    }

    case 2: {
      _localctx = dynamic_cast<Index_statementContext *>(_tracker.createInstance<SQLParser::Alter_drop_indexContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(135);
      match(SQLParser::T__23);
      setState(136);
      match(SQLParser::T__5);
      setState(137);
      match(SQLParser::Identifier);
      setState(138);
      match(SQLParser::T__8);
      setState(139);
      match(SQLParser::T__25);
      setState(140);
      match(SQLParser::T__6);
      setState(141);
      identifiers();
      setState(142);
      match(SQLParser::T__7);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Field_listContext ------------------------------------------------------------------

SQLParser::Field_listContext::Field_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SQLParser::FieldContext *> SQLParser::Field_listContext::field() {
  return getRuleContexts<SQLParser::FieldContext>();
}

SQLParser::FieldContext* SQLParser::Field_listContext::field(size_t i) {
  return getRuleContext<SQLParser::FieldContext>(i);
}


size_t SQLParser::Field_listContext::getRuleIndex() const {
  return SQLParser::RuleField_list;
}

antlrcpp::Any SQLParser::Field_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitField_list(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::Field_listContext* SQLParser::field_list() {
  Field_listContext *_localctx = _tracker.createInstance<Field_listContext>(_ctx, getState());
  enterRule(_localctx, 12, SQLParser::RuleField_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(146);
    field();
    setState(151);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SQLParser::T__26) {
      setState(147);
      match(SQLParser::T__26);
      setState(148);
      field();
      setState(153);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FieldContext ------------------------------------------------------------------

SQLParser::FieldContext::FieldContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SQLParser::FieldContext::getRuleIndex() const {
  return SQLParser::RuleField;
}

void SQLParser::FieldContext::copyFrom(FieldContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Normal_fieldContext ------------------------------------------------------------------

tree::TerminalNode* SQLParser::Normal_fieldContext::Identifier() {
  return getToken(SQLParser::Identifier, 0);
}

SQLParser::Type_Context* SQLParser::Normal_fieldContext::type_() {
  return getRuleContext<SQLParser::Type_Context>(0);
}

SQLParser::Normal_fieldContext::Normal_fieldContext(FieldContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Normal_fieldContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitNormal_field(this);
  else
    return visitor->visitChildren(this);
}
SQLParser::FieldContext* SQLParser::field() {
  FieldContext *_localctx = _tracker.createInstance<FieldContext>(_ctx, getState());
  enterRule(_localctx, 14, SQLParser::RuleField);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    _localctx = dynamic_cast<FieldContext *>(_tracker.createInstance<SQLParser::Normal_fieldContext>(_localctx));
    enterOuterAlt(_localctx, 1);
    setState(154);
    match(SQLParser::Identifier);
    setState(155);
    type_();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Type_Context ------------------------------------------------------------------

SQLParser::Type_Context::Type_Context(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SQLParser::Type_Context::Integer() {
  return getToken(SQLParser::Integer, 0);
}


size_t SQLParser::Type_Context::getRuleIndex() const {
  return SQLParser::RuleType_;
}

antlrcpp::Any SQLParser::Type_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitType_(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::Type_Context* SQLParser::type_() {
  Type_Context *_localctx = _tracker.createInstance<Type_Context>(_ctx, getState());
  enterRule(_localctx, 16, SQLParser::RuleType_);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(163);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SQLParser::T__27: {
        enterOuterAlt(_localctx, 1);
        setState(157);
        match(SQLParser::T__27);
        break;
      }

      case SQLParser::T__28: {
        enterOuterAlt(_localctx, 2);
        setState(158);
        match(SQLParser::T__28);
        setState(159);
        match(SQLParser::T__6);
        setState(160);
        match(SQLParser::Integer);
        setState(161);
        match(SQLParser::T__7);
        break;
      }

      case SQLParser::T__29: {
        enterOuterAlt(_localctx, 3);
        setState(162);
        match(SQLParser::T__29);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Value_listsContext ------------------------------------------------------------------

SQLParser::Value_listsContext::Value_listsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SQLParser::Value_listContext *> SQLParser::Value_listsContext::value_list() {
  return getRuleContexts<SQLParser::Value_listContext>();
}

SQLParser::Value_listContext* SQLParser::Value_listsContext::value_list(size_t i) {
  return getRuleContext<SQLParser::Value_listContext>(i);
}


size_t SQLParser::Value_listsContext::getRuleIndex() const {
  return SQLParser::RuleValue_lists;
}

antlrcpp::Any SQLParser::Value_listsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitValue_lists(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::Value_listsContext* SQLParser::value_lists() {
  Value_listsContext *_localctx = _tracker.createInstance<Value_listsContext>(_ctx, getState());
  enterRule(_localctx, 18, SQLParser::RuleValue_lists);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(165);
    value_list();
    setState(170);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SQLParser::T__26) {
      setState(166);
      match(SQLParser::T__26);
      setState(167);
      value_list();
      setState(172);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Value_listContext ------------------------------------------------------------------

SQLParser::Value_listContext::Value_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SQLParser::ValueContext *> SQLParser::Value_listContext::value() {
  return getRuleContexts<SQLParser::ValueContext>();
}

SQLParser::ValueContext* SQLParser::Value_listContext::value(size_t i) {
  return getRuleContext<SQLParser::ValueContext>(i);
}


size_t SQLParser::Value_listContext::getRuleIndex() const {
  return SQLParser::RuleValue_list;
}

antlrcpp::Any SQLParser::Value_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitValue_list(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::Value_listContext* SQLParser::value_list() {
  Value_listContext *_localctx = _tracker.createInstance<Value_listContext>(_ctx, getState());
  enterRule(_localctx, 20, SQLParser::RuleValue_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(173);
    match(SQLParser::T__6);
    setState(174);
    value();
    setState(179);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SQLParser::T__26) {
      setState(175);
      match(SQLParser::T__26);
      setState(176);
      value();
      setState(181);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(182);
    match(SQLParser::T__7);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ValueContext ------------------------------------------------------------------

SQLParser::ValueContext::ValueContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SQLParser::ValueContext::Integer() {
  return getToken(SQLParser::Integer, 0);
}

tree::TerminalNode* SQLParser::ValueContext::String() {
  return getToken(SQLParser::String, 0);
}

tree::TerminalNode* SQLParser::ValueContext::Float() {
  return getToken(SQLParser::Float, 0);
}

tree::TerminalNode* SQLParser::ValueContext::Null() {
  return getToken(SQLParser::Null, 0);
}


size_t SQLParser::ValueContext::getRuleIndex() const {
  return SQLParser::RuleValue;
}

antlrcpp::Any SQLParser::ValueContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitValue(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::ValueContext* SQLParser::value() {
  ValueContext *_localctx = _tracker.createInstance<ValueContext>(_ctx, getState());
  enterRule(_localctx, 22, SQLParser::RuleValue);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(184);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SQLParser::Null)
      | (1ULL << SQLParser::Integer)
      | (1ULL << SQLParser::String)
      | (1ULL << SQLParser::Float))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Where_and_clauseContext ------------------------------------------------------------------

SQLParser::Where_and_clauseContext::Where_and_clauseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SQLParser::Where_clauseContext *> SQLParser::Where_and_clauseContext::where_clause() {
  return getRuleContexts<SQLParser::Where_clauseContext>();
}

SQLParser::Where_clauseContext* SQLParser::Where_and_clauseContext::where_clause(size_t i) {
  return getRuleContext<SQLParser::Where_clauseContext>(i);
}


size_t SQLParser::Where_and_clauseContext::getRuleIndex() const {
  return SQLParser::RuleWhere_and_clause;
}

antlrcpp::Any SQLParser::Where_and_clauseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitWhere_and_clause(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::Where_and_clauseContext* SQLParser::where_and_clause() {
  Where_and_clauseContext *_localctx = _tracker.createInstance<Where_and_clauseContext>(_ctx, getState());
  enterRule(_localctx, 24, SQLParser::RuleWhere_and_clause);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(186);
    where_clause();
    setState(191);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SQLParser::T__30) {
      setState(187);
      match(SQLParser::T__30);
      setState(188);
      where_clause();
      setState(193);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Where_clauseContext ------------------------------------------------------------------

SQLParser::Where_clauseContext::Where_clauseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SQLParser::Where_clauseContext::getRuleIndex() const {
  return SQLParser::RuleWhere_clause;
}

void SQLParser::Where_clauseContext::copyFrom(Where_clauseContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Where_operator_expressionContext ------------------------------------------------------------------

SQLParser::ColumnContext* SQLParser::Where_operator_expressionContext::column() {
  return getRuleContext<SQLParser::ColumnContext>(0);
}

SQLParser::OperateContext* SQLParser::Where_operator_expressionContext::operate() {
  return getRuleContext<SQLParser::OperateContext>(0);
}

SQLParser::ExpressionContext* SQLParser::Where_operator_expressionContext::expression() {
  return getRuleContext<SQLParser::ExpressionContext>(0);
}

SQLParser::Where_operator_expressionContext::Where_operator_expressionContext(Where_clauseContext *ctx) { copyFrom(ctx); }

antlrcpp::Any SQLParser::Where_operator_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitWhere_operator_expression(this);
  else
    return visitor->visitChildren(this);
}
SQLParser::Where_clauseContext* SQLParser::where_clause() {
  Where_clauseContext *_localctx = _tracker.createInstance<Where_clauseContext>(_ctx, getState());
  enterRule(_localctx, 26, SQLParser::RuleWhere_clause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    _localctx = dynamic_cast<Where_clauseContext *>(_tracker.createInstance<SQLParser::Where_operator_expressionContext>(_localctx));
    enterOuterAlt(_localctx, 1);
    setState(194);
    column();
    setState(195);
    operate();
    setState(196);
    expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ColumnContext ------------------------------------------------------------------

SQLParser::ColumnContext::ColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> SQLParser::ColumnContext::Identifier() {
  return getTokens(SQLParser::Identifier);
}

tree::TerminalNode* SQLParser::ColumnContext::Identifier(size_t i) {
  return getToken(SQLParser::Identifier, i);
}


size_t SQLParser::ColumnContext::getRuleIndex() const {
  return SQLParser::RuleColumn;
}

antlrcpp::Any SQLParser::ColumnContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitColumn(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::ColumnContext* SQLParser::column() {
  ColumnContext *_localctx = _tracker.createInstance<ColumnContext>(_ctx, getState());
  enterRule(_localctx, 28, SQLParser::RuleColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(198);
    match(SQLParser::Identifier);
    setState(199);
    match(SQLParser::T__31);
    setState(200);
    match(SQLParser::Identifier);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

SQLParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SQLParser::ValueContext* SQLParser::ExpressionContext::value() {
  return getRuleContext<SQLParser::ValueContext>(0);
}

SQLParser::ColumnContext* SQLParser::ExpressionContext::column() {
  return getRuleContext<SQLParser::ColumnContext>(0);
}


size_t SQLParser::ExpressionContext::getRuleIndex() const {
  return SQLParser::RuleExpression;
}

antlrcpp::Any SQLParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::ExpressionContext* SQLParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 30, SQLParser::RuleExpression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(204);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SQLParser::Null:
      case SQLParser::Integer:
      case SQLParser::String:
      case SQLParser::Float: {
        enterOuterAlt(_localctx, 1);
        setState(202);
        value();
        break;
      }

      case SQLParser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(203);
        column();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Set_clauseContext ------------------------------------------------------------------

SQLParser::Set_clauseContext::Set_clauseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> SQLParser::Set_clauseContext::Identifier() {
  return getTokens(SQLParser::Identifier);
}

tree::TerminalNode* SQLParser::Set_clauseContext::Identifier(size_t i) {
  return getToken(SQLParser::Identifier, i);
}

std::vector<tree::TerminalNode *> SQLParser::Set_clauseContext::EqualOrAssign() {
  return getTokens(SQLParser::EqualOrAssign);
}

tree::TerminalNode* SQLParser::Set_clauseContext::EqualOrAssign(size_t i) {
  return getToken(SQLParser::EqualOrAssign, i);
}

std::vector<SQLParser::ValueContext *> SQLParser::Set_clauseContext::value() {
  return getRuleContexts<SQLParser::ValueContext>();
}

SQLParser::ValueContext* SQLParser::Set_clauseContext::value(size_t i) {
  return getRuleContext<SQLParser::ValueContext>(i);
}


size_t SQLParser::Set_clauseContext::getRuleIndex() const {
  return SQLParser::RuleSet_clause;
}

antlrcpp::Any SQLParser::Set_clauseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitSet_clause(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::Set_clauseContext* SQLParser::set_clause() {
  Set_clauseContext *_localctx = _tracker.createInstance<Set_clauseContext>(_ctx, getState());
  enterRule(_localctx, 32, SQLParser::RuleSet_clause);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(206);
    match(SQLParser::Identifier);
    setState(207);
    match(SQLParser::EqualOrAssign);
    setState(208);
    value();
    setState(215);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SQLParser::T__26) {
      setState(209);
      match(SQLParser::T__26);
      setState(210);
      match(SQLParser::Identifier);
      setState(211);
      match(SQLParser::EqualOrAssign);
      setState(212);
      value();
      setState(217);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SelectorsContext ------------------------------------------------------------------

SQLParser::SelectorsContext::SelectorsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SQLParser::SelectorContext *> SQLParser::SelectorsContext::selector() {
  return getRuleContexts<SQLParser::SelectorContext>();
}

SQLParser::SelectorContext* SQLParser::SelectorsContext::selector(size_t i) {
  return getRuleContext<SQLParser::SelectorContext>(i);
}


size_t SQLParser::SelectorsContext::getRuleIndex() const {
  return SQLParser::RuleSelectors;
}

antlrcpp::Any SQLParser::SelectorsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitSelectors(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::SelectorsContext* SQLParser::selectors() {
  SelectorsContext *_localctx = _tracker.createInstance<SelectorsContext>(_ctx, getState());
  enterRule(_localctx, 34, SQLParser::RuleSelectors);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(227);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SQLParser::T__32: {
        enterOuterAlt(_localctx, 1);
        setState(218);
        match(SQLParser::T__32);
        break;
      }

      case SQLParser::Count:
      case SQLParser::Average:
      case SQLParser::Max:
      case SQLParser::Min:
      case SQLParser::Sum:
      case SQLParser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(219);
        selector();
        setState(224);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == SQLParser::T__26) {
          setState(220);
          match(SQLParser::T__26);
          setState(221);
          selector();
          setState(226);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SelectorContext ------------------------------------------------------------------

SQLParser::SelectorContext::SelectorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SQLParser::ColumnContext* SQLParser::SelectorContext::column() {
  return getRuleContext<SQLParser::ColumnContext>(0);
}

SQLParser::AggregatorContext* SQLParser::SelectorContext::aggregator() {
  return getRuleContext<SQLParser::AggregatorContext>(0);
}

tree::TerminalNode* SQLParser::SelectorContext::Count() {
  return getToken(SQLParser::Count, 0);
}


size_t SQLParser::SelectorContext::getRuleIndex() const {
  return SQLParser::RuleSelector;
}

antlrcpp::Any SQLParser::SelectorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitSelector(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::SelectorContext* SQLParser::selector() {
  SelectorContext *_localctx = _tracker.createInstance<SelectorContext>(_ctx, getState());
  enterRule(_localctx, 36, SQLParser::RuleSelector);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(239);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(229);
      column();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(230);
      aggregator();
      setState(231);
      match(SQLParser::T__6);
      setState(232);
      column();
      setState(233);
      match(SQLParser::T__7);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(235);
      match(SQLParser::Count);
      setState(236);
      match(SQLParser::T__6);
      setState(237);
      match(SQLParser::T__32);
      setState(238);
      match(SQLParser::T__7);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdentifiersContext ------------------------------------------------------------------

SQLParser::IdentifiersContext::IdentifiersContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> SQLParser::IdentifiersContext::Identifier() {
  return getTokens(SQLParser::Identifier);
}

tree::TerminalNode* SQLParser::IdentifiersContext::Identifier(size_t i) {
  return getToken(SQLParser::Identifier, i);
}


size_t SQLParser::IdentifiersContext::getRuleIndex() const {
  return SQLParser::RuleIdentifiers;
}

antlrcpp::Any SQLParser::IdentifiersContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitIdentifiers(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::IdentifiersContext* SQLParser::identifiers() {
  IdentifiersContext *_localctx = _tracker.createInstance<IdentifiersContext>(_ctx, getState());
  enterRule(_localctx, 38, SQLParser::RuleIdentifiers);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(241);
    match(SQLParser::Identifier);
    setState(246);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SQLParser::T__26) {
      setState(242);
      match(SQLParser::T__26);
      setState(243);
      match(SQLParser::Identifier);
      setState(248);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OperateContext ------------------------------------------------------------------

SQLParser::OperateContext::OperateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SQLParser::OperateContext::EqualOrAssign() {
  return getToken(SQLParser::EqualOrAssign, 0);
}

tree::TerminalNode* SQLParser::OperateContext::Less() {
  return getToken(SQLParser::Less, 0);
}

tree::TerminalNode* SQLParser::OperateContext::LessEqual() {
  return getToken(SQLParser::LessEqual, 0);
}

tree::TerminalNode* SQLParser::OperateContext::Greater() {
  return getToken(SQLParser::Greater, 0);
}

tree::TerminalNode* SQLParser::OperateContext::GreaterEqual() {
  return getToken(SQLParser::GreaterEqual, 0);
}

tree::TerminalNode* SQLParser::OperateContext::NotEqual() {
  return getToken(SQLParser::NotEqual, 0);
}


size_t SQLParser::OperateContext::getRuleIndex() const {
  return SQLParser::RuleOperate;
}

antlrcpp::Any SQLParser::OperateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitOperate(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::OperateContext* SQLParser::operate() {
  OperateContext *_localctx = _tracker.createInstance<OperateContext>(_ctx, getState());
  enterRule(_localctx, 40, SQLParser::RuleOperate);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(249);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SQLParser::EqualOrAssign)
      | (1ULL << SQLParser::Less)
      | (1ULL << SQLParser::LessEqual)
      | (1ULL << SQLParser::Greater)
      | (1ULL << SQLParser::GreaterEqual)
      | (1ULL << SQLParser::NotEqual))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AggregatorContext ------------------------------------------------------------------

SQLParser::AggregatorContext::AggregatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SQLParser::AggregatorContext::Count() {
  return getToken(SQLParser::Count, 0);
}

tree::TerminalNode* SQLParser::AggregatorContext::Average() {
  return getToken(SQLParser::Average, 0);
}

tree::TerminalNode* SQLParser::AggregatorContext::Max() {
  return getToken(SQLParser::Max, 0);
}

tree::TerminalNode* SQLParser::AggregatorContext::Min() {
  return getToken(SQLParser::Min, 0);
}

tree::TerminalNode* SQLParser::AggregatorContext::Sum() {
  return getToken(SQLParser::Sum, 0);
}


size_t SQLParser::AggregatorContext::getRuleIndex() const {
  return SQLParser::RuleAggregator;
}

antlrcpp::Any SQLParser::AggregatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SQLVisitor*>(visitor))
    return parserVisitor->visitAggregator(this);
  else
    return visitor->visitChildren(this);
}

SQLParser::AggregatorContext* SQLParser::aggregator() {
  AggregatorContext *_localctx = _tracker.createInstance<AggregatorContext>(_ctx, getState());
  enterRule(_localctx, 42, SQLParser::RuleAggregator);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(251);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SQLParser::Count)
      | (1ULL << SQLParser::Average)
      | (1ULL << SQLParser::Max)
      | (1ULL << SQLParser::Min)
      | (1ULL << SQLParser::Sum))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

// Static vars and initialization.
std::vector<dfa::DFA> SQLParser::_decisionToDFA;
atn::PredictionContextCache SQLParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN SQLParser::_atn;
std::vector<uint16_t> SQLParser::_serializedATN;

std::vector<std::string> SQLParser::_ruleNames = {
  "program", "statement", "db_statement", "table_statement", "select_table", 
  "index_statement", "field_list", "field", "type_", "value_lists", "value_list", 
  "value", "where_and_clause", "where_clause", "column", "expression", "set_clause", 
  "selectors", "selector", "identifiers", "operate", "aggregator"
};

std::vector<std::string> SQLParser::_literalNames = {
  "", "';'", "'SHOW'", "'TABLES'", "'INDEXES'", "'CREATE'", "'TABLE'", "'('", 
  "')'", "'DROP'", "'DESC'", "'INSERT'", "'INTO'", "'VALUES'", "'DELETE'", 
  "'FROM'", "'WHERE'", "'UPDATE'", "'SET'", "'SELECT'", "'GROUP'", "'BY'", 
  "'LIMIT'", "'OFFSET'", "'ALTER'", "'ADD'", "'INDEX'", "','", "'INT'", 
  "'VARCHAR'", "'FLOAT'", "'AND'", "'.'", "'*'", "'='", "'<'", "'<='", "'>'", 
  "'>='", "'<>'", "'COUNT'", "'AVG'", "'MAX'", "'MIN'", "'SUM'", "'NULL'"
};

std::vector<std::string> SQLParser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "EqualOrAssign", 
  "Less", "LessEqual", "Greater", "GreaterEqual", "NotEqual", "Count", "Average", 
  "Max", "Min", "Sum", "Null", "Identifier", "Integer", "String", "Float", 
  "Whitespace", "Annotation"
};

dfa::Vocabulary SQLParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> SQLParser::_tokenNames;

SQLParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x35, 0x100, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x3, 0x2, 0x7, 
    0x2, 0x30, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x33, 0xb, 0x2, 0x3, 0x2, 0x3, 
    0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 
    0x3, 0x44, 0xa, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 
    0x4a, 0xa, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x5, 0x5, 0x6a, 0xa, 0x5, 0x3, 0x6, 
    0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x72, 0xa, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x77, 0xa, 0x6, 0x3, 0x6, 
    0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x7d, 0xa, 0x6, 0x5, 0x6, 0x7f, 
    0xa, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 
    0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 
    0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0x93, 0xa, 
    0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x7, 0x8, 0x98, 0xa, 0x8, 0xc, 0x8, 
    0xe, 0x8, 0x9b, 0xb, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xa6, 0xa, 0xa, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x7, 0xb, 0xab, 0xa, 0xb, 0xc, 0xb, 0xe, 
    0xb, 0xae, 0xb, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x7, 0xc, 
    0xb4, 0xa, 0xc, 0xc, 0xc, 0xe, 0xc, 0xb7, 0xb, 0xc, 0x3, 0xc, 0x3, 0xc, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x7, 0xe, 0xc0, 0xa, 
    0xe, 0xc, 0xe, 0xe, 0xe, 0xc3, 0xb, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 
    0x11, 0x5, 0x11, 0xcf, 0xa, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 
    0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x7, 0x12, 0xd8, 0xa, 0x12, 0xc, 
    0x12, 0xe, 0x12, 0xdb, 0xb, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 
    0x13, 0x7, 0x13, 0xe1, 0xa, 0x13, 0xc, 0x13, 0xe, 0x13, 0xe4, 0xb, 0x13, 
    0x5, 0x13, 0xe6, 0xa, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x5, 
    0x14, 0xf2, 0xa, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x7, 0x15, 0xf7, 
    0xa, 0x15, 0xc, 0x15, 0xe, 0x15, 0xfa, 0xb, 0x15, 0x3, 0x16, 0x3, 0x16, 
    0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x2, 0x2, 0x18, 0x2, 0x4, 0x6, 0x8, 
    0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 
    0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2, 0x5, 0x4, 0x2, 0x2f, 0x2f, 
    0x31, 0x33, 0x3, 0x2, 0x24, 0x29, 0x3, 0x2, 0x2a, 0x2e, 0x2, 0x107, 
    0x2, 0x31, 0x3, 0x2, 0x2, 0x2, 0x4, 0x43, 0x3, 0x2, 0x2, 0x2, 0x6, 0x49, 
    0x3, 0x2, 0x2, 0x2, 0x8, 0x69, 0x3, 0x2, 0x2, 0x2, 0xa, 0x6b, 0x3, 0x2, 
    0x2, 0x2, 0xc, 0x92, 0x3, 0x2, 0x2, 0x2, 0xe, 0x94, 0x3, 0x2, 0x2, 0x2, 
    0x10, 0x9c, 0x3, 0x2, 0x2, 0x2, 0x12, 0xa5, 0x3, 0x2, 0x2, 0x2, 0x14, 
    0xa7, 0x3, 0x2, 0x2, 0x2, 0x16, 0xaf, 0x3, 0x2, 0x2, 0x2, 0x18, 0xba, 
    0x3, 0x2, 0x2, 0x2, 0x1a, 0xbc, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xc4, 0x3, 
    0x2, 0x2, 0x2, 0x1e, 0xc8, 0x3, 0x2, 0x2, 0x2, 0x20, 0xce, 0x3, 0x2, 
    0x2, 0x2, 0x22, 0xd0, 0x3, 0x2, 0x2, 0x2, 0x24, 0xe5, 0x3, 0x2, 0x2, 
    0x2, 0x26, 0xf1, 0x3, 0x2, 0x2, 0x2, 0x28, 0xf3, 0x3, 0x2, 0x2, 0x2, 
    0x2a, 0xfb, 0x3, 0x2, 0x2, 0x2, 0x2c, 0xfd, 0x3, 0x2, 0x2, 0x2, 0x2e, 
    0x30, 0x5, 0x4, 0x3, 0x2, 0x2f, 0x2e, 0x3, 0x2, 0x2, 0x2, 0x30, 0x33, 
    0x3, 0x2, 0x2, 0x2, 0x31, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x31, 0x32, 0x3, 
    0x2, 0x2, 0x2, 0x32, 0x34, 0x3, 0x2, 0x2, 0x2, 0x33, 0x31, 0x3, 0x2, 
    0x2, 0x2, 0x34, 0x35, 0x7, 0x2, 0x2, 0x3, 0x35, 0x3, 0x3, 0x2, 0x2, 
    0x2, 0x36, 0x37, 0x5, 0x6, 0x4, 0x2, 0x37, 0x38, 0x7, 0x3, 0x2, 0x2, 
    0x38, 0x44, 0x3, 0x2, 0x2, 0x2, 0x39, 0x3a, 0x5, 0x8, 0x5, 0x2, 0x3a, 
    0x3b, 0x7, 0x3, 0x2, 0x2, 0x3b, 0x44, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x3d, 
    0x5, 0xc, 0x7, 0x2, 0x3d, 0x3e, 0x7, 0x3, 0x2, 0x2, 0x3e, 0x44, 0x3, 
    0x2, 0x2, 0x2, 0x3f, 0x40, 0x7, 0x35, 0x2, 0x2, 0x40, 0x44, 0x7, 0x3, 
    0x2, 0x2, 0x41, 0x42, 0x7, 0x2f, 0x2, 0x2, 0x42, 0x44, 0x7, 0x3, 0x2, 
    0x2, 0x43, 0x36, 0x3, 0x2, 0x2, 0x2, 0x43, 0x39, 0x3, 0x2, 0x2, 0x2, 
    0x43, 0x3c, 0x3, 0x2, 0x2, 0x2, 0x43, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x43, 
    0x41, 0x3, 0x2, 0x2, 0x2, 0x44, 0x5, 0x3, 0x2, 0x2, 0x2, 0x45, 0x46, 
    0x7, 0x4, 0x2, 0x2, 0x46, 0x4a, 0x7, 0x5, 0x2, 0x2, 0x47, 0x48, 0x7, 
    0x4, 0x2, 0x2, 0x48, 0x4a, 0x7, 0x6, 0x2, 0x2, 0x49, 0x45, 0x3, 0x2, 
    0x2, 0x2, 0x49, 0x47, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x7, 0x3, 0x2, 0x2, 
    0x2, 0x4b, 0x4c, 0x7, 0x7, 0x2, 0x2, 0x4c, 0x4d, 0x7, 0x8, 0x2, 0x2, 
    0x4d, 0x4e, 0x7, 0x30, 0x2, 0x2, 0x4e, 0x4f, 0x7, 0x9, 0x2, 0x2, 0x4f, 
    0x50, 0x5, 0xe, 0x8, 0x2, 0x50, 0x51, 0x7, 0xa, 0x2, 0x2, 0x51, 0x6a, 
    0x3, 0x2, 0x2, 0x2, 0x52, 0x53, 0x7, 0xb, 0x2, 0x2, 0x53, 0x54, 0x7, 
    0x8, 0x2, 0x2, 0x54, 0x6a, 0x7, 0x30, 0x2, 0x2, 0x55, 0x56, 0x7, 0xc, 
    0x2, 0x2, 0x56, 0x6a, 0x7, 0x30, 0x2, 0x2, 0x57, 0x58, 0x7, 0xd, 0x2, 
    0x2, 0x58, 0x59, 0x7, 0xe, 0x2, 0x2, 0x59, 0x5a, 0x7, 0x30, 0x2, 0x2, 
    0x5a, 0x5b, 0x7, 0xf, 0x2, 0x2, 0x5b, 0x6a, 0x5, 0x14, 0xb, 0x2, 0x5c, 
    0x5d, 0x7, 0x10, 0x2, 0x2, 0x5d, 0x5e, 0x7, 0x11, 0x2, 0x2, 0x5e, 0x5f, 
    0x7, 0x30, 0x2, 0x2, 0x5f, 0x60, 0x7, 0x12, 0x2, 0x2, 0x60, 0x6a, 0x5, 
    0x1a, 0xe, 0x2, 0x61, 0x62, 0x7, 0x13, 0x2, 0x2, 0x62, 0x63, 0x7, 0x30, 
    0x2, 0x2, 0x63, 0x64, 0x7, 0x14, 0x2, 0x2, 0x64, 0x65, 0x5, 0x22, 0x12, 
    0x2, 0x65, 0x66, 0x7, 0x12, 0x2, 0x2, 0x66, 0x67, 0x5, 0x1a, 0xe, 0x2, 
    0x67, 0x6a, 0x3, 0x2, 0x2, 0x2, 0x68, 0x6a, 0x5, 0xa, 0x6, 0x2, 0x69, 
    0x4b, 0x3, 0x2, 0x2, 0x2, 0x69, 0x52, 0x3, 0x2, 0x2, 0x2, 0x69, 0x55, 
    0x3, 0x2, 0x2, 0x2, 0x69, 0x57, 0x3, 0x2, 0x2, 0x2, 0x69, 0x5c, 0x3, 
    0x2, 0x2, 0x2, 0x69, 0x61, 0x3, 0x2, 0x2, 0x2, 0x69, 0x68, 0x3, 0x2, 
    0x2, 0x2, 0x6a, 0x9, 0x3, 0x2, 0x2, 0x2, 0x6b, 0x6c, 0x7, 0x15, 0x2, 
    0x2, 0x6c, 0x6d, 0x5, 0x24, 0x13, 0x2, 0x6d, 0x6e, 0x7, 0x11, 0x2, 0x2, 
    0x6e, 0x71, 0x5, 0x28, 0x15, 0x2, 0x6f, 0x70, 0x7, 0x12, 0x2, 0x2, 0x70, 
    0x72, 0x5, 0x1a, 0xe, 0x2, 0x71, 0x6f, 0x3, 0x2, 0x2, 0x2, 0x71, 0x72, 
    0x3, 0x2, 0x2, 0x2, 0x72, 0x76, 0x3, 0x2, 0x2, 0x2, 0x73, 0x74, 0x7, 
    0x16, 0x2, 0x2, 0x74, 0x75, 0x7, 0x17, 0x2, 0x2, 0x75, 0x77, 0x5, 0x1e, 
    0x10, 0x2, 0x76, 0x73, 0x3, 0x2, 0x2, 0x2, 0x76, 0x77, 0x3, 0x2, 0x2, 
    0x2, 0x77, 0x7e, 0x3, 0x2, 0x2, 0x2, 0x78, 0x79, 0x7, 0x18, 0x2, 0x2, 
    0x79, 0x7c, 0x7, 0x31, 0x2, 0x2, 0x7a, 0x7b, 0x7, 0x19, 0x2, 0x2, 0x7b, 
    0x7d, 0x7, 0x31, 0x2, 0x2, 0x7c, 0x7a, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x7d, 
    0x3, 0x2, 0x2, 0x2, 0x7d, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x7e, 0x78, 0x3, 
    0x2, 0x2, 0x2, 0x7e, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x7f, 0xb, 0x3, 0x2, 
    0x2, 0x2, 0x80, 0x81, 0x7, 0x1a, 0x2, 0x2, 0x81, 0x82, 0x7, 0x8, 0x2, 
    0x2, 0x82, 0x83, 0x7, 0x30, 0x2, 0x2, 0x83, 0x84, 0x7, 0x1b, 0x2, 0x2, 
    0x84, 0x85, 0x7, 0x1c, 0x2, 0x2, 0x85, 0x86, 0x7, 0x9, 0x2, 0x2, 0x86, 
    0x87, 0x5, 0x28, 0x15, 0x2, 0x87, 0x88, 0x7, 0xa, 0x2, 0x2, 0x88, 0x93, 
    0x3, 0x2, 0x2, 0x2, 0x89, 0x8a, 0x7, 0x1a, 0x2, 0x2, 0x8a, 0x8b, 0x7, 
    0x8, 0x2, 0x2, 0x8b, 0x8c, 0x7, 0x30, 0x2, 0x2, 0x8c, 0x8d, 0x7, 0xb, 
    0x2, 0x2, 0x8d, 0x8e, 0x7, 0x1c, 0x2, 0x2, 0x8e, 0x8f, 0x7, 0x9, 0x2, 
    0x2, 0x8f, 0x90, 0x5, 0x28, 0x15, 0x2, 0x90, 0x91, 0x7, 0xa, 0x2, 0x2, 
    0x91, 0x93, 0x3, 0x2, 0x2, 0x2, 0x92, 0x80, 0x3, 0x2, 0x2, 0x2, 0x92, 
    0x89, 0x3, 0x2, 0x2, 0x2, 0x93, 0xd, 0x3, 0x2, 0x2, 0x2, 0x94, 0x99, 
    0x5, 0x10, 0x9, 0x2, 0x95, 0x96, 0x7, 0x1d, 0x2, 0x2, 0x96, 0x98, 0x5, 
    0x10, 0x9, 0x2, 0x97, 0x95, 0x3, 0x2, 0x2, 0x2, 0x98, 0x9b, 0x3, 0x2, 
    0x2, 0x2, 0x99, 0x97, 0x3, 0x2, 0x2, 0x2, 0x99, 0x9a, 0x3, 0x2, 0x2, 
    0x2, 0x9a, 0xf, 0x3, 0x2, 0x2, 0x2, 0x9b, 0x99, 0x3, 0x2, 0x2, 0x2, 
    0x9c, 0x9d, 0x7, 0x30, 0x2, 0x2, 0x9d, 0x9e, 0x5, 0x12, 0xa, 0x2, 0x9e, 
    0x11, 0x3, 0x2, 0x2, 0x2, 0x9f, 0xa6, 0x7, 0x1e, 0x2, 0x2, 0xa0, 0xa1, 
    0x7, 0x1f, 0x2, 0x2, 0xa1, 0xa2, 0x7, 0x9, 0x2, 0x2, 0xa2, 0xa3, 0x7, 
    0x31, 0x2, 0x2, 0xa3, 0xa6, 0x7, 0xa, 0x2, 0x2, 0xa4, 0xa6, 0x7, 0x20, 
    0x2, 0x2, 0xa5, 0x9f, 0x3, 0x2, 0x2, 0x2, 0xa5, 0xa0, 0x3, 0x2, 0x2, 
    0x2, 0xa5, 0xa4, 0x3, 0x2, 0x2, 0x2, 0xa6, 0x13, 0x3, 0x2, 0x2, 0x2, 
    0xa7, 0xac, 0x5, 0x16, 0xc, 0x2, 0xa8, 0xa9, 0x7, 0x1d, 0x2, 0x2, 0xa9, 
    0xab, 0x5, 0x16, 0xc, 0x2, 0xaa, 0xa8, 0x3, 0x2, 0x2, 0x2, 0xab, 0xae, 
    0x3, 0x2, 0x2, 0x2, 0xac, 0xaa, 0x3, 0x2, 0x2, 0x2, 0xac, 0xad, 0x3, 
    0x2, 0x2, 0x2, 0xad, 0x15, 0x3, 0x2, 0x2, 0x2, 0xae, 0xac, 0x3, 0x2, 
    0x2, 0x2, 0xaf, 0xb0, 0x7, 0x9, 0x2, 0x2, 0xb0, 0xb5, 0x5, 0x18, 0xd, 
    0x2, 0xb1, 0xb2, 0x7, 0x1d, 0x2, 0x2, 0xb2, 0xb4, 0x5, 0x18, 0xd, 0x2, 
    0xb3, 0xb1, 0x3, 0x2, 0x2, 0x2, 0xb4, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xb5, 
    0xb3, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xb6, 0xb8, 
    0x3, 0x2, 0x2, 0x2, 0xb7, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xb9, 0x7, 
    0xa, 0x2, 0x2, 0xb9, 0x17, 0x3, 0x2, 0x2, 0x2, 0xba, 0xbb, 0x9, 0x2, 
    0x2, 0x2, 0xbb, 0x19, 0x3, 0x2, 0x2, 0x2, 0xbc, 0xc1, 0x5, 0x1c, 0xf, 
    0x2, 0xbd, 0xbe, 0x7, 0x21, 0x2, 0x2, 0xbe, 0xc0, 0x5, 0x1c, 0xf, 0x2, 
    0xbf, 0xbd, 0x3, 0x2, 0x2, 0x2, 0xc0, 0xc3, 0x3, 0x2, 0x2, 0x2, 0xc1, 
    0xbf, 0x3, 0x2, 0x2, 0x2, 0xc1, 0xc2, 0x3, 0x2, 0x2, 0x2, 0xc2, 0x1b, 
    0x3, 0x2, 0x2, 0x2, 0xc3, 0xc1, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc5, 0x5, 
    0x1e, 0x10, 0x2, 0xc5, 0xc6, 0x5, 0x2a, 0x16, 0x2, 0xc6, 0xc7, 0x5, 
    0x20, 0x11, 0x2, 0xc7, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xc8, 0xc9, 0x7, 0x30, 
    0x2, 0x2, 0xc9, 0xca, 0x7, 0x22, 0x2, 0x2, 0xca, 0xcb, 0x7, 0x30, 0x2, 
    0x2, 0xcb, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xcc, 0xcf, 0x5, 0x18, 0xd, 0x2, 
    0xcd, 0xcf, 0x5, 0x1e, 0x10, 0x2, 0xce, 0xcc, 0x3, 0x2, 0x2, 0x2, 0xce, 
    0xcd, 0x3, 0x2, 0x2, 0x2, 0xcf, 0x21, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xd1, 
    0x7, 0x30, 0x2, 0x2, 0xd1, 0xd2, 0x7, 0x24, 0x2, 0x2, 0xd2, 0xd9, 0x5, 
    0x18, 0xd, 0x2, 0xd3, 0xd4, 0x7, 0x1d, 0x2, 0x2, 0xd4, 0xd5, 0x7, 0x30, 
    0x2, 0x2, 0xd5, 0xd6, 0x7, 0x24, 0x2, 0x2, 0xd6, 0xd8, 0x5, 0x18, 0xd, 
    0x2, 0xd7, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xd8, 0xdb, 0x3, 0x2, 0x2, 0x2, 
    0xd9, 0xd7, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xda, 0x3, 0x2, 0x2, 0x2, 0xda, 
    0x23, 0x3, 0x2, 0x2, 0x2, 0xdb, 0xd9, 0x3, 0x2, 0x2, 0x2, 0xdc, 0xe6, 
    0x7, 0x23, 0x2, 0x2, 0xdd, 0xe2, 0x5, 0x26, 0x14, 0x2, 0xde, 0xdf, 0x7, 
    0x1d, 0x2, 0x2, 0xdf, 0xe1, 0x5, 0x26, 0x14, 0x2, 0xe0, 0xde, 0x3, 0x2, 
    0x2, 0x2, 0xe1, 0xe4, 0x3, 0x2, 0x2, 0x2, 0xe2, 0xe0, 0x3, 0x2, 0x2, 
    0x2, 0xe2, 0xe3, 0x3, 0x2, 0x2, 0x2, 0xe3, 0xe6, 0x3, 0x2, 0x2, 0x2, 
    0xe4, 0xe2, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xdc, 0x3, 0x2, 0x2, 0x2, 0xe5, 
    0xdd, 0x3, 0x2, 0x2, 0x2, 0xe6, 0x25, 0x3, 0x2, 0x2, 0x2, 0xe7, 0xf2, 
    0x5, 0x1e, 0x10, 0x2, 0xe8, 0xe9, 0x5, 0x2c, 0x17, 0x2, 0xe9, 0xea, 
    0x7, 0x9, 0x2, 0x2, 0xea, 0xeb, 0x5, 0x1e, 0x10, 0x2, 0xeb, 0xec, 0x7, 
    0xa, 0x2, 0x2, 0xec, 0xf2, 0x3, 0x2, 0x2, 0x2, 0xed, 0xee, 0x7, 0x2a, 
    0x2, 0x2, 0xee, 0xef, 0x7, 0x9, 0x2, 0x2, 0xef, 0xf0, 0x7, 0x23, 0x2, 
    0x2, 0xf0, 0xf2, 0x7, 0xa, 0x2, 0x2, 0xf1, 0xe7, 0x3, 0x2, 0x2, 0x2, 
    0xf1, 0xe8, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xed, 0x3, 0x2, 0x2, 0x2, 0xf2, 
    0x27, 0x3, 0x2, 0x2, 0x2, 0xf3, 0xf8, 0x7, 0x30, 0x2, 0x2, 0xf4, 0xf5, 
    0x7, 0x1d, 0x2, 0x2, 0xf5, 0xf7, 0x7, 0x30, 0x2, 0x2, 0xf6, 0xf4, 0x3, 
    0x2, 0x2, 0x2, 0xf7, 0xfa, 0x3, 0x2, 0x2, 0x2, 0xf8, 0xf6, 0x3, 0x2, 
    0x2, 0x2, 0xf8, 0xf9, 0x3, 0x2, 0x2, 0x2, 0xf9, 0x29, 0x3, 0x2, 0x2, 
    0x2, 0xfa, 0xf8, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xfc, 0x9, 0x3, 0x2, 0x2, 
    0xfc, 0x2b, 0x3, 0x2, 0x2, 0x2, 0xfd, 0xfe, 0x9, 0x4, 0x2, 0x2, 0xfe, 
    0x2d, 0x3, 0x2, 0x2, 0x2, 0x16, 0x31, 0x43, 0x49, 0x69, 0x71, 0x76, 
    0x7c, 0x7e, 0x92, 0x99, 0xa5, 0xac, 0xb5, 0xc1, 0xce, 0xd9, 0xe2, 0xe5, 
    0xf1, 0xf8, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

SQLParser::Initializer SQLParser::_init;
