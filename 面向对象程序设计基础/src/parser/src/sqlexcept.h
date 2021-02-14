#pragma once

#include <exception>
#include <string>

/**
 * @brief Base class for sql command parsing exception.
 */
class SqlError : public std::exception
{
public:
    /**
     * @brief Constructor with specific message
     * @param msg Specific message.
     */
    SqlError(const std::string &msg) : _msg(msg)
    {}

    /**
     * @brief Get the exception message
     * @return The Exception message
     */
    virtual const char *what() const noexcept override
    { return _msg.c_str(); }

protected:
    std::string _msg;     ///< Exception message.
};

/**
 * @brief The exception class for errors during token scanning.
 */
class TokenError : public SqlError
{
public:
    /**
     * @brief Construct with specific exception message, which will be prefixed by "Token Error:".
     * @param msg Specific exception message.
     */
    TokenError(const std::string &msg) : SqlError("Token Error: " + msg)
    {}
};

/**
 * @brief A class for exception during lexing.
 */
class LexerError : public SqlError
{
public:
    /**
     * @brief Construct with specific exception message, which will be prefixed by "Lexer Error:".
     * @param msg Specific exception message.
     */
    LexerError(const std::string &msg) : SqlError("Lexer Error: " + msg)
    {}
};

/**
 * @brief A class for exception during paring.
 */
class ParserError : public SqlError
{
public:
    /**
     * @brief Construct with specific exception message., which will be prefixed by "Parser Error:".
     * @param msg Specific exception message.
     */
    ParserError(const std::string &msg) : SqlError("Parser Error: " + msg)
    {}
};

/**
 * @brief The exception class for Expr operation error.
 */
class ExprError : public SqlError
{
public:
    /**
     * @brief Construct with specific exception message, which will be later prefixed by "Expr Error:".
     * @param msg Specific exception message.
     */
    ExprError(const std::string &msg) noexcept: SqlError("Expr Error: " + msg)
    {}
};
