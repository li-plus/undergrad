#pragma once

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <memory>
#include "token.h"

/**
 * @brief This class organize the expression as a tree structure.
 * @details
 * Each expression could have several sub-expressions and a token which indicates the expression type.
 * Expression may contain undetermined variables. They are represented by their names.
 * When values of all variables are given, the result can be figure out with eval().
 */
class Expr
{
public:
    Expr() = default;

    /**
     * @brief Constructor with specific token. Left & right sub-expressions will be set to nullptr.
     * @param token Token of this expression.
     */
    Expr(const Token &token) : _token(token)
    {}

    /**
     * @brief Construct with token and sub-expressions as smart pointers.
     * @param token Expression token.
     * @param children A vector of smart pointers of sub-expressions.
     */
    Expr(const Token &token, const std::vector<std::shared_ptr<Expr>> &children) : _token(token), _children(children)
    {}

    /**
     * @brief Set token of the expression.
     * @param t Specific token.
     */
    void setToken(const Token &t)
    { _token = t; }

    /**
     * @brief Determine whether an expression is null.
     * @return true if null, false otherwise.
     */
    bool isNull() const
    { return _token.type() == Token::NONE; }

    /**
     * @brief Get the token
     * @return Token of the expression.
     */
    const Token &token() const
    { return _token; }

    /**
     * @brief Calculate the expression result in terms of the specific values
     * @param varMap    the given variant-value map.
     * @return Result of the expression
     */
    Variant eval(const std::map<std::string, Variant> &varMap) const;

    /**
     * @brief Calculate the expression result with count().
     * @param varMap    the given variant-value map.
     * @return  Result of the expression
     */
    Variant eval(const std::map<std::string, std::vector<Variant>> &varMap) const;

    /**
     * @brief Print out the expression
     * @param out   output stream
     * @param expr  expression to print
     * @return output stream
     */
    friend std::ostream &operator<<(std::ostream &out, const Expr &expr);

protected:
    std::vector<std::shared_ptr<Expr>> _children;   ///< sub-expressions
    Token _token;       ///< token of the expression
};
