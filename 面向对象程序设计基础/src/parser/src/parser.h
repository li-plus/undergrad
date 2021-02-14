#pragma once

#include "lexer.h"
#include "expr.h"
#include "statements.h"
#include "../../backend/src/print.h"
#include "stack.h"

/**
 * @brief A parser parses a sequence of tokens into a statement.
 * @details
 * The parser calls the lexer to scan a series of tokens.
 * By analyzing the logical links among the tokens,
 * the parser generates a statement consisting of all the useful information within the query,
 * and return the statement for backend to execute.
 */
class Parser
{
public:
    /**
     * @brief Constructor with specific sql query
     * @param cmd The sql query to parse
     */
    Parser(const std::string &cmd) : _lexer(cmd)
    {}

    /**
     * @brief Get the current token.
     * @return The current token.
     */
    const Token &token() const
    { return _token; }

    /**
     * @brief Get the inner lexer.
     * @return The inner lexer.
     */
    const Lexer &lexer() const
    { return _lexer; }

    /**
     * @brief parse sql statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseStatement();

public:
    static std::map<Token::Type, int> operatorPriority; ///< The priorities of all different operators.
protected:
    /**
      * @brief Consume a specific type of token.
      * @param type Type of token to be consumed.
      * @return The consumed token.
      */
    Token consume(Token::Type type);

    /**
     * @brief Update the expression in terms of the upcoming operator.
     * @param vals Stack of variants.
     * @param ops Stack of operators.
     * @param pendingOp The pending operator.
     */
    void handlePendingOperator(Stack<Expr> &vals, Stack<Token::Type> &ops, Token::Type pendingOp);

    /**
     * @brief Determine whether a token is an operator
     * @param token The given token
     * @return whether the token is an operator
     */
    bool isOperator(const Token &token) const
    { return operatorPriority.find(token.type()) != operatorPriority.end(); }

    /**
     * @brief parse create statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseCreate();

    /**
     * @brief parse delete statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseDelete();

    /**
     * @brief parse drop statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseDrop();

    /**
     * @brief parse show statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseShow();

    /**
     * @brief parse use statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseUse();

    /**
     * @brief parse insert statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseInsert();

    /**
     * @brief parse update statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseUpdate();

    /**
     * @brief parse select statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseSelect();

    /**
     * @brief parse load statement
     * @return the parsed statement
     */
    std::shared_ptr<Statement> parseLoad();

    /**
     * @brief Parse a list of column names.
     * @return The list of column names.
     */
    std::vector<std::string> parseColumnNameList();

    /**
     * @brief Parse a list of values.
     * @return The list of values.
     */
    std::vector<Variant> parseValueList();

    /**
     * @brief Parse a list of fields
     * @param fields        The output list of fields.
     * @param primaryKey    The output primary key.
     */
    void parseFieldList(std::vector<Field> &fields, std::string &primaryKey);

    /**
     * @brief Parse a field
     * @param fields        The output list of fields.
     * @param primaryKey    The output primary key.
     */
    void parseField(std::vector<Field> &fields, std::string &primaryKey);

    /**
     * @brief Parse an expression
     * @return the expression
     */
    Expr parseExpr();

    /**
     * @brief Parse an expression list
     * @return The expression list.
     */
    std::vector<Expr> parseExprList();

    /**
     * @brief Parse the SET list.
     * @param keys Specific keys.
     * @param values Destination values.
     */
    void parseSetList(std::vector<std::string> &keys, std::vector<Variant> &values);

protected:
    Token _token;   ///< the current token.
    Lexer _lexer;   ///< the lexer.
};
