#pragma once

#include "token.h"
#include <sstream>
#include <cstdio>
#include <cctype>

/**
 * @brief A lexer reads from the string stream and scan the tokens.
 * @details
 * The lexer scans the string from left to right.
 * It converts the plain text into a sequence of tokens.
 * Return one token each time.
 */
class Lexer
{
public:
    /**
     * @brief Construct a lexer with a command.
     * @param cmd A string of command.
     */
    Lexer(const std::string &cmd) : _stream(cmd)
    {}

    /**
     * @brief Determine whether a character is part of an operator
     * @param c     a character
     * @return whether a character is part of an operator
     */
    bool isOperatorChar(char c) const;

    /**
     * @brief Determine whether the stream reaches EOF
     * @return whether the stream reaches EOF
     */
    bool isEnd()
    { return _stream.peek() == EOF; }

    /**
     * @brief Get the next character in the stream but not extracting it.
     * @return The next character to read.
     */
    char peek()
    { return (char) _stream.peek(); }

    /**
     * @brief Scan and return the next token.
     * @return Next token.
     */
    Token next();

    /**
     * @brief Advance one char each time.
     * @return The read char.
     */
    char advance()
    { return (char) _stream.get(); }

    /**
     * @brief Put a char back in the stream.
     * @param c A character.
     */
    void retreat(char c)
    { _stream.putback(c); }

public:
    static const size_t maxOpLen = 2;   ///< maximum operator length
    static const size_t maxBufferSize = 256;   ///< maximum buffer size.
protected:
    char _buffer[maxBufferSize]; ///< Buffer for keywords, identifiers, etc.
    std::istringstream _stream; ///< The input stream
};
