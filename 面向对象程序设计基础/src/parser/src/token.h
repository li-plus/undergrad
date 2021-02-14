#pragma once

#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#include "../../backend/src/variant.h"

/**
 * @brief A token is the minimum unit in an SQL query.
 * @details
 * A token may be a keyword, operator, identifier or operand.
 */
class Token
{
public:
    enum Type
    {
        NONE,
        ID,
        OPERAND,
        INT,
        DOUBLE,
        CHAR,
        TEXT,
        DATABASE,
        DATABASES,
        TABLE,
        TABLES,
        CREATE,
        INSERT,
        SELECT,
        DELETE,
        SHOW,
        DROP,
        UPDATE,
        USE,
        PRIMARY,
        KEY,
        FROM,
        INTO,
        SET,
        VALUES,
        WHERE,
        COLUMNS,
        LT,
        GT,
        NEQ,
        EQ,
        GEQ,
        LEQ,
        NOT,
        NULL_SQL,
        PLUS,
        MINUS,
        MUL,
        DIV,
        MOD,
        AND,
        OR,
        XOR,
        L_PAREN,
        R_PAREN,
        COMMA,
        SEMICOLON,
        END,
        OUTFILE,
        LOAD,
        DATA,
        INFILE,
        COUNT,
        GROUP,
        ORDER,
        BY,
        UNION,
        ALL,
        LIKE,
        JOIN,
        INNER,
        LEFT,
        RIGHT,
        ON,
        DATE,
        ADDDATE,
        TIME,
        ADDTIME,
        ABS,
        EXP,
        FLOOR,
        CEIL,
        LN,
        LOG10,
        PI,
        SIN,
        COS,
        TAN,
        ASIN,
        ACOS,
        ATAN,
        LINE_COMMENT,       // one-line comment: --
        BLOCK_COMMENT       // block comment: /*
    };

    /**
     * @brief Default trivial constructor.
     */
    Token() = default;

    /**
     * @brief Construct with specific token type
     * @param type Token type
     */
    Token(Type type) : _type(type)
    {}

    /**
     * @brief Constructor with specific token type and data.
     * @param type Token type.
     * @param data Token data.
     */
    Token(Type type, const Variant &data) : _type(type), _data(data)
    {}

    /**
     * @brief Get the type of the token.
     * @return  the type of the token
     */
    Token::Type type() const
    { return _type; }

    /**
     * @brief Determine whether a token represents a function.
     * @return  true if function, false otherwise.
     */
    bool isFunction() const;

    /**
     * @brief Convert the token to an identifier.
     * @return a string of id.
     */
    std::string toId() const;

    /**
     * @brief Convert the token to an operand.
     * @return the operand
     */
    Variant toOperand() const;

    /**
     * Print out the token
     * @param out Output stream
     * @param token The token to print.
     * @return The output stream
     */
    friend std::ostream &operator<<(std::ostream &out, const Token &token);

    /**
     * @brief Get the type name of the token type.
     * @param type
     * @return The type name as an std::string.
     */
    static std::string typeName(Type type);

    /**
     * @brief Standard output method for token type.
     * @param out Standard output stream.
     * @param type Token type.
     * @return Standard output stream.
     */
    friend std::ostream &operator<<(std::ostream &out, const Token::Type &type)
    { return out << typeName(type); }

public:
    static std::map<std::string, Type> keywords;    ///< keyword to token type
    static std::map<std::string, Token::Type> operators; ///< operator to token type
protected:
    Type _type = NONE;  ///< the token type
    Variant _data;      ///< the token data
};

