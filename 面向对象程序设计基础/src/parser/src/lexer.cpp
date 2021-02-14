#include "lexer.h"
#include "utils.h"
#include "sqlexcept.h"
#include <iostream>

bool Lexer::isOperatorChar(char c) const
{
    switch (c)
    {
        case '!':
        case '<':
        case '>':
        case '=':
        case '+':
        case '-':
        case '*':
        case '/':
        case '(':
        case ')':
        case ',':
        case ';':
        case '%':
            return true;
        default:
            return false;
    }
}

Token Lexer::next()
{
    while (!isEnd())
    {
        if (isdigit(peek()))      // number
        {
            int bufInt = advance() - '0'; // int
            while (isdigit(peek()))
                bufInt = bufInt * 10 + advance() - '0';
            if (peek() != '.')    // int
                return Token(Token::OPERAND, bufInt);
            double bufDbl = bufInt;
            advance();
            for (double frac = 0.1; isdigit(peek()); frac *= 0.1)
                bufDbl += frac * (advance() - '0');
            return Token(Token::OPERAND, bufDbl);
        }
        else if (isalpha(peek()) || peek() == '_')  // keywords or identifiers
        {
            memset(_buffer, 0, sizeof(_buffer));
            for (size_t cnt = 0; isalnum(peek()) || peek() == '_' || peek() == '.'; cnt++)
            {
                if (cnt >= maxBufferSize)
                    throw LexerError("Exceed maximum identifier length");
                _buffer[cnt] = advance();
            }
            std::string str(_buffer);
            if (Token::keywords.find(toUpper(str)) != Token::keywords.end()) // keyword
                return Token(Token::keywords[toUpper(str)]);
            else // case sensitive for identifier
                return Token(Token::ID, std::string(_buffer));
        }
        else if (peek() == '"' || peek() == '\'')  // string
        {
            char quote = advance();
            int cnt = 0;
            while (peek() != quote)
                _buffer[cnt++] = advance();
            _buffer[cnt] = '\0';
            advance();
            return Token(Token::OPERAND, Variant(std::string(_buffer)));
        }
        else if (isspace(peek()))      // white space
        {
            advance();
        }
        else if (isOperatorChar(peek()))          // operators
        {
            memset(_buffer, 0, sizeof(_buffer));
            for (int cnt = 0; cnt < maxOpLen; cnt++)
            {
                if (!isOperatorChar(peek()))
                    break;
                _buffer[cnt] = advance();
            }
            std::string str(_buffer);
            while (!str.empty() && Token::operators.find(str) == Token::operators.end()) // too long
            {
                retreat(str.back());
                str.pop_back();
            }
            if (str.empty())
                throw LexerError("Invalid operator");
            auto operatorType = Token::operators[str];
            if (operatorType == Token::LINE_COMMENT)    // line comment --
            {
                while (advance() != '\n')
                    /* pass */;
                continue;
            }
            if(operatorType == Token::BLOCK_COMMENT)    // block comment /*
            {
                str.clear();
                str.push_back(advance());
                str.push_back(advance());
                while(str != "*/")
                {
                    str[0] = str[1];
                    str[1] = advance();
                }
                continue;
            }
            return Token(Token::operators[str]);
        }
        else // error
        {
            throw LexerError(std::string("Invalid lexeme ") + peek());
        }
    }
    // peek() == EOF
    advance();
    return Token(Token::END);
}
