#include "token.h"
#include "sqlexcept.h"

std::map<std::string, Token::Type>  Token::keywords = {
        {"CREATE",    Token::CREATE},
        {"TABLE",     Token::TABLE},
        {"TABLES",    Token::TABLES},
        {"DATABASE",  Token::DATABASE},
        {"DATABASES", Token::DATABASES},
        {"INT",       Token::INT},
        {"DOUBLE",    Token::DOUBLE},
        {"CHAR",      Token::CHAR},
        {"TEXT",      Token::TEXT},
        {"PRIMARY",   Token::PRIMARY},
        {"KEY",       Token::KEY},
        {"INSERT",    Token::INSERT},
        {"INTO",      Token::INTO},
        {"VALUES",    Token::VALUES},
        {"DELETE",    Token::DELETE},
        {"FROM",      Token::FROM},
        {"WHERE",     Token::WHERE},
        {"SELECT",    Token::SELECT},
        {"SHOW",      Token::SHOW},
        {"COLUMNS",   Token::COLUMNS},
        {"UPDATE",    Token::UPDATE},
        {"DROP",      Token::DROP},
        {"USE",       Token::USE},
        {"NOT",       Token::NOT},
        {"NULL",      Token::NULL_SQL},
        {"AND",       Token::AND},
        {"XOR",       Token::XOR},
        {"OR",        Token::OR},
        {"SET",       Token::SET},
        {"OUTFILE",   Token::OUTFILE},
        {"LOAD",      Token::LOAD},
        {"DATA",      Token::DATA},
        {"INFILE",    Token::INFILE},
        {"COUNT",     Token::COUNT},
        {"GROUP",     Token::GROUP},
        {"ORDER",     Token::ORDER},
        {"BY",        Token::BY},
        {"UNION",     Token::UNION},
        {"ALL",       Token::ALL},
        {"LIKE",      Token::LIKE},
        {"JOIN",      Token::JOIN},
        {"LEFT",      Token::LEFT},
        {"RIGHT",     Token::RIGHT},
        {"INNER",     Token::INNER},
        {"ON",        Token::ON},
        // extra type
        {"DATE",      Token::DATE},
        {"ADDDATE",   Token::ADDDATE},
        {"TIME",      Token::TIME},
        {"ADDTIME",   Token::ADDTIME},
        // math
        {"ABS",       Token::ABS},
        {"EXP",       Token::EXP},
        {"FLOOR",     Token::FLOOR},
        {"CEIL",      Token::CEIL},
        {"LN",        Token::LN},
        {"LOG10",     Token::LOG10},
        {"PI",        Token::PI},
        {"SIN",       Token::SIN},
        {"COS",       Token::COS},
        {"TAN",       Token::TAN},
        {"ASIN",      Token::ASIN},
        {"ACOS",      Token::ACOS},
        {"ATAN",      Token::ATAN}};
std::map<std::string, Token::Type> Token::operators = {
        {"!",  Token::NOT},
        {"<",  Token::LT},
        {">",  Token::GT},
        {"<>", Token::NEQ},
        {"!=", Token::NEQ},
        {"=",  Token::EQ},
        {">=", Token::GEQ},
        {"<=", Token::LEQ},
        {"+",  Token::PLUS},
        {"-",  Token::MINUS},
        {"*",  Token::MUL},
        {"/",  Token::DIV},
        {"%",  Token::MOD},
        {"(",  Token::L_PAREN},
        {")",  Token::R_PAREN},
        {",",  Token::COMMA},
        {";",  Token::SEMICOLON},
        {"--", Token::LINE_COMMENT},
        {"/*", Token::BLOCK_COMMENT}};

bool Token::isFunction() const
{
    switch (_type)
    {
        case Token::COUNT:
        case Token::ABS:
        case Token::EXP:
        case Token::FLOOR:
        case Token::CEIL:
        case Token::LN:
        case Token::LOG10:
        case Token::SIN:
        case Token::COS:
        case Token::TAN:
        case Token::ASIN:
        case Token::ACOS:
        case Token::ATAN:
        case Token::ADDDATE:
        case Token::ADDTIME:
            return true;
        default:
            return false;
    }
}

std::string Token::toId() const
{
    if (_type != ID)
        throw TokenError("Expected identifier, get otherwise");
    return _data.toStdString();
}

Variant Token::toOperand() const
{
    if (_type != OPERAND)
        throw TokenError("Expected number, get otherwise");
    return _data;
}

std::ostream &operator<<(std::ostream &out, const Token &token)
{
    switch (token.type())
    {
        case Token::ID:
            return out << "(" << token << ", " << token.toId() << ")";
        case Token::OPERAND:
            return out << "(" << token << ", " << token.toOperand() << ")";
        default:
            return out << token.type();
    }
}

std::string Token::typeName(Token::Type type)
{
    for (auto &p: keywords)
    {
        if (p.second == type)
            return p.first;
    }
    for (auto &p: operators)
    {
        if (p.second == type)
            return p.first;
    }
    return std::to_string(type);
}
