
// Generated from .\SQL.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"




class  SQLParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, T__11 = 12, T__12 = 13, T__13 = 14, 
    T__14 = 15, T__15 = 16, T__16 = 17, T__17 = 18, T__18 = 19, T__19 = 20, 
    T__20 = 21, T__21 = 22, T__22 = 23, T__23 = 24, T__24 = 25, T__25 = 26, 
    T__26 = 27, T__27 = 28, T__28 = 29, T__29 = 30, T__30 = 31, T__31 = 32, 
    T__32 = 33, EqualOrAssign = 34, Less = 35, LessEqual = 36, Greater = 37, 
    GreaterEqual = 38, NotEqual = 39, Count = 40, Average = 41, Max = 42, 
    Min = 43, Sum = 44, Null = 45, Identifier = 46, Integer = 47, String = 48, 
    Float = 49, Whitespace = 50, Annotation = 51
  };

  enum {
    RuleProgram = 0, RuleStatement = 1, RuleDb_statement = 2, RuleTable_statement = 3, 
    RuleSelect_table = 4, RuleIndex_statement = 5, RuleField_list = 6, RuleField = 7, 
    RuleType_ = 8, RuleValue_lists = 9, RuleValue_list = 10, RuleValue = 11, 
    RuleWhere_and_clause = 12, RuleWhere_clause = 13, RuleColumn = 14, RuleExpression = 15, 
    RuleSet_clause = 16, RuleSelectors = 17, RuleSelector = 18, RuleIdentifiers = 19, 
    RuleOperate = 20, RuleAggregator = 21
  };

  SQLParser(antlr4::TokenStream *input);
  ~SQLParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  class ProgramContext;
  class StatementContext;
  class Db_statementContext;
  class Table_statementContext;
  class Select_tableContext;
  class Index_statementContext;
  class Field_listContext;
  class FieldContext;
  class Type_Context;
  class Value_listsContext;
  class Value_listContext;
  class ValueContext;
  class Where_and_clauseContext;
  class Where_clauseContext;
  class ColumnContext;
  class ExpressionContext;
  class Set_clauseContext;
  class SelectorsContext;
  class SelectorContext;
  class IdentifiersContext;
  class OperateContext;
  class AggregatorContext; 

  class  ProgramContext : public antlr4::ParserRuleContext {
  public:
    ProgramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EOF();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProgramContext* program();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Db_statementContext *db_statement();
    Table_statementContext *table_statement();
    Index_statementContext *index_statement();
    antlr4::tree::TerminalNode *Annotation();
    antlr4::tree::TerminalNode *Null();

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StatementContext* statement();

  class  Db_statementContext : public antlr4::ParserRuleContext {
  public:
    Db_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Db_statementContext() : antlr4::ParserRuleContext() { }
    void copyFrom(Db_statementContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Show_tablesContext : public Db_statementContext {
  public:
    Show_tablesContext(Db_statementContext *ctx);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Show_indexesContext : public Db_statementContext {
  public:
    Show_indexesContext(Db_statementContext *ctx);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Db_statementContext* db_statement();

  class  Table_statementContext : public antlr4::ParserRuleContext {
  public:
    Table_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Table_statementContext() : antlr4::ParserRuleContext() { }
    void copyFrom(Table_statementContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Delete_from_tableContext : public Table_statementContext {
  public:
    Delete_from_tableContext(Table_statementContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    Where_and_clauseContext *where_and_clause();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Insert_into_tableContext : public Table_statementContext {
  public:
    Insert_into_tableContext(Table_statementContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    Value_listsContext *value_lists();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Create_tableContext : public Table_statementContext {
  public:
    Create_tableContext(Table_statementContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    Field_listContext *field_list();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Describe_tableContext : public Table_statementContext {
  public:
    Describe_tableContext(Table_statementContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Select_table_Context : public Table_statementContext {
  public:
    Select_table_Context(Table_statementContext *ctx);

    Select_tableContext *select_table();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Drop_tableContext : public Table_statementContext {
  public:
    Drop_tableContext(Table_statementContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Update_tableContext : public Table_statementContext {
  public:
    Update_tableContext(Table_statementContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    Set_clauseContext *set_clause();
    Where_and_clauseContext *where_and_clause();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Table_statementContext* table_statement();

  class  Select_tableContext : public antlr4::ParserRuleContext {
  public:
    Select_tableContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SelectorsContext *selectors();
    IdentifiersContext *identifiers();
    Where_and_clauseContext *where_and_clause();
    ColumnContext *column();
    std::vector<antlr4::tree::TerminalNode *> Integer();
    antlr4::tree::TerminalNode* Integer(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Select_tableContext* select_table();

  class  Index_statementContext : public antlr4::ParserRuleContext {
  public:
    Index_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Index_statementContext() : antlr4::ParserRuleContext() { }
    void copyFrom(Index_statementContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Alter_drop_indexContext : public Index_statementContext {
  public:
    Alter_drop_indexContext(Index_statementContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    IdentifiersContext *identifiers();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Alter_add_indexContext : public Index_statementContext {
  public:
    Alter_add_indexContext(Index_statementContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    IdentifiersContext *identifiers();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Index_statementContext* index_statement();

  class  Field_listContext : public antlr4::ParserRuleContext {
  public:
    Field_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<FieldContext *> field();
    FieldContext* field(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Field_listContext* field_list();

  class  FieldContext : public antlr4::ParserRuleContext {
  public:
    FieldContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    FieldContext() : antlr4::ParserRuleContext() { }
    void copyFrom(FieldContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Normal_fieldContext : public FieldContext {
  public:
    Normal_fieldContext(FieldContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    Type_Context *type_();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FieldContext* field();

  class  Type_Context : public antlr4::ParserRuleContext {
  public:
    Type_Context(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Integer();

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Type_Context* type_();

  class  Value_listsContext : public antlr4::ParserRuleContext {
  public:
    Value_listsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<Value_listContext *> value_list();
    Value_listContext* value_list(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Value_listsContext* value_lists();

  class  Value_listContext : public antlr4::ParserRuleContext {
  public:
    Value_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ValueContext *> value();
    ValueContext* value(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Value_listContext* value_list();

  class  ValueContext : public antlr4::ParserRuleContext {
  public:
    ValueContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Integer();
    antlr4::tree::TerminalNode *String();
    antlr4::tree::TerminalNode *Float();
    antlr4::tree::TerminalNode *Null();

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ValueContext* value();

  class  Where_and_clauseContext : public antlr4::ParserRuleContext {
  public:
    Where_and_clauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<Where_clauseContext *> where_clause();
    Where_clauseContext* where_clause(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Where_and_clauseContext* where_and_clause();

  class  Where_clauseContext : public antlr4::ParserRuleContext {
  public:
    Where_clauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Where_clauseContext() : antlr4::ParserRuleContext() { }
    void copyFrom(Where_clauseContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Where_operator_expressionContext : public Where_clauseContext {
  public:
    Where_operator_expressionContext(Where_clauseContext *ctx);

    ColumnContext *column();
    OperateContext *operate();
    ExpressionContext *expression();
    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Where_clauseContext* where_clause();

  class  ColumnContext : public antlr4::ParserRuleContext {
  public:
    ColumnContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> Identifier();
    antlr4::tree::TerminalNode* Identifier(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ColumnContext* column();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ValueContext *value();
    ColumnContext *column();

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionContext* expression();

  class  Set_clauseContext : public antlr4::ParserRuleContext {
  public:
    Set_clauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> Identifier();
    antlr4::tree::TerminalNode* Identifier(size_t i);
    std::vector<antlr4::tree::TerminalNode *> EqualOrAssign();
    antlr4::tree::TerminalNode* EqualOrAssign(size_t i);
    std::vector<ValueContext *> value();
    ValueContext* value(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Set_clauseContext* set_clause();

  class  SelectorsContext : public antlr4::ParserRuleContext {
  public:
    SelectorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<SelectorContext *> selector();
    SelectorContext* selector(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SelectorsContext* selectors();

  class  SelectorContext : public antlr4::ParserRuleContext {
  public:
    SelectorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ColumnContext *column();
    AggregatorContext *aggregator();
    antlr4::tree::TerminalNode *Count();

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SelectorContext* selector();

  class  IdentifiersContext : public antlr4::ParserRuleContext {
  public:
    IdentifiersContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> Identifier();
    antlr4::tree::TerminalNode* Identifier(size_t i);

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IdentifiersContext* identifiers();

  class  OperateContext : public antlr4::ParserRuleContext {
  public:
    OperateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EqualOrAssign();
    antlr4::tree::TerminalNode *Less();
    antlr4::tree::TerminalNode *LessEqual();
    antlr4::tree::TerminalNode *Greater();
    antlr4::tree::TerminalNode *GreaterEqual();
    antlr4::tree::TerminalNode *NotEqual();

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OperateContext* operate();

  class  AggregatorContext : public antlr4::ParserRuleContext {
  public:
    AggregatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Count();
    antlr4::tree::TerminalNode *Average();
    antlr4::tree::TerminalNode *Max();
    antlr4::tree::TerminalNode *Min();
    antlr4::tree::TerminalNode *Sum();

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AggregatorContext* aggregator();


private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

