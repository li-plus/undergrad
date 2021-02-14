#include "parser.h"
#include "sqlexcept.h"
#include "../../backend/src/field.h"
#include "../../backend/include/stack.h"
#include <algorithm>
#include <cmath>

std::map<Token::Type, int> Parser::operatorPriority = {
        {Token::ADDDATE, 100},
        {Token::ADDTIME, 100},
        {Token::COUNT,   100},
        {Token::ABS,     100},
        {Token::EXP,     100},
        {Token::FLOOR,   100},
        {Token::CEIL,    100},
        {Token::LN,      100},
        {Token::LOG10,   100},
        {Token::PI,      100},
        {Token::SIN,     100},
        {Token::COS,     100},
        {Token::TAN,     100},
        {Token::ASIN,    100},
        {Token::ACOS,    100},
        {Token::ATAN,    100},
        {Token::MUL,     10},
        {Token::DIV,     10},
        {Token::MOD,     10},
        {Token::PLUS,    9},
        {Token::MINUS,   9},
        {Token::LIKE,    5},
        {Token::EQ,      5},
        {Token::NEQ,     5},
        {Token::LEQ,     5},
        {Token::GEQ,     5},
        {Token::GT,      5},
        {Token::LT,      5},
        {Token::NOT,     4},
        {Token::AND,     3},
        {Token::XOR,     2},
        {Token::OR,      1},
        {Token::L_PAREN, 0},
        {Token::R_PAREN, 0},
        {Token::COMMA,   -50},// this priority value is not used.
        {Token::END,     -100}
};

std::shared_ptr<Statement> Parser::parseStatement()
{
    _token = _lexer.next();
    switch (_token.type())
    {
        case Token::CREATE:
            return parseCreate();
        case Token::INSERT:
            return parseInsert();
        case Token::DELETE:
            return parseDelete();
        case Token::UPDATE:
            return parseUpdate();
        case Token::USE:
            return parseUse();
        case Token::SELECT:
            return parseSelect();
        case Token::DROP:
            return parseDrop();
        case Token::SHOW:
            return parseShow();
        case Token::LOAD:
            return parseLoad();
        case Token::END:
        case Token::SEMICOLON:
            return nullptr;
        default:
            throw ParserError("Unexpected first token.");
    }
}

std::shared_ptr<Statement> Parser::parseCreate()
{
    consume(Token::CREATE);
    switch (_token.type())
    {
        case Token::TABLE:
        {
            consume(Token::TABLE);
            std::string table_id = _token.toId();
            consume(Token::ID);
            consume(Token::L_PAREN);
            std::vector<Field> fields;
            std::string priKey;
            parseFieldList(fields, priKey);
            consume(Token::R_PAREN);
            consume(Token::SEMICOLON);
            return std::make_shared<StatementCreateTable>(StatementCreateTable(table_id, fields, priKey));
        }
        case Token::DATABASE:
        {
            consume(Token::DATABASE);
            std::string databaseId = _token.toId();
            consume(Token::ID);
            consume(Token::SEMICOLON);
            return std::make_shared<StatementCreateDatabase>(StatementCreateDatabase(databaseId));
        }
        default:
            throw ParserError("Unexpected token near CREATE.");
    }
}

void Parser::parseFieldList(std::vector<Field> &fields, std::string &primaryKey)
{
    switch (_token.type())
    {
        case Token::ID:
        case Token::PRIMARY:
            parseField(fields, primaryKey);
            while (_token.type() == Token::COMMA)
            {
                consume(Token::COMMA);
                parseField(fields, primaryKey);
            }
            break;
        default:
            throw ParserError("Unexpected token in field list.");
    }
}

void Parser::parseField(std::vector<Field> &fields, std::string &primaryKey)
{
    switch (_token.type())
    {
        case Token::ID:
        {
            std::string name = _token.toId();
            consume(Token::ID);
            switch (_token.type())
            {
                case Token::INT:
                    fields.emplace_back(Field(name, Variant::INT));
                    break;
                case Token::CHAR:
                    fields.emplace_back(Field(name, Variant::CHAR));
                    break;
                case Token::DOUBLE:
                    fields.emplace_back(Field(name, Variant::DOUBLE));
                    break;
                case Token::TEXT:
                    fields.emplace_back(Field(name, Variant::STRING));
                    break;
                case Token::DATE:
                    fields.emplace_back(Field(name, Variant::DATE));
                    break;
                case Token::TIME:
                    fields.emplace_back(Field(name, Variant::TIME));
                    break;
                default:
                    throw ParserError("Invalid column type " + Token::typeName(_token.type()));
            }
            consume(_token.type());
            if (_token.type() == Token::NOT)
            {
                consume(Token::NOT);
                consume(Token::NULL_SQL);
                fields.back().setIsNull(false);
            }
            break;
        }
        case Token::PRIMARY:
        {
            consume(Token::PRIMARY);
            consume(Token::KEY);
            consume(Token::L_PAREN);
            primaryKey = _token.toId();
            consume(Token::ID);
            auto it = std::find_if(fields.begin(), fields.end(), [=](const Field &f) {
                return f.key() == primaryKey;
            });
            if (it == fields.end())
                throw ParserError("Primary key not found");
            it->setPrimary(true);
            consume(Token::R_PAREN);
            break;
        }
        default:
            throw ParserError("Unexpected token in field.");
    }
}

std::vector<std::string> Parser::parseColumnNameList()
{
    std::vector<std::string> colNames = {consume(Token::ID).toId()};
    while (_token.type() == Token::COMMA)
    {
        consume(Token::COMMA);
        colNames.emplace_back(consume(Token::ID).toId());
    }
    return colNames;
}

std::shared_ptr<Statement> Parser::parseInsert()
{
    consume(Token::INSERT);
    consume(Token::INTO);
    std::string table_id = consume(Token::ID).toId();
    consume(Token::L_PAREN);
    std::vector<std::string> columns = parseColumnNameList();
    consume(Token::R_PAREN);
    consume(Token::VALUES);
    consume(Token::L_PAREN);
    std::vector<Variant> values = parseValueList();
    if (values.size() != columns.size())
        throw ParserError("Column size and value size do not match");
    std::map<std::string, Variant> entry;
    for (size_t i = 0; i < values.size(); i++)
        entry[columns[i]] = values[i];
    consume(Token::R_PAREN);
    consume(Token::SEMICOLON);
    return std::make_shared<StatementInsert>(StatementInsert(table_id, entry));
}

std::vector<Variant> Parser::parseValueList()
{
    std::vector<Variant> values = {parseExpr().eval(std::map<std::string, Variant>())};
    while (_token.type() == Token::COMMA)
    {
        consume(Token::COMMA);
        values.emplace_back(parseExpr().eval(std::map<std::string, Variant>()));
    }
    return values;
}

std::shared_ptr<Statement> Parser::parseDelete()
{
    consume(Token::DELETE);
    consume(Token::FROM);
    std::string table_id = consume(Token::ID).toId();
    Expr where;
    switch (_token.type())
    {
        case Token::WHERE:
            consume(Token::WHERE);
            where = parseExpr();
            break;
        case Token::SEMICOLON:
            break;
        default:
            throw ParserError("Unexpected token near where clause.");
    }
    consume(Token::SEMICOLON);
    return std::make_shared<StatementDelete>(StatementDelete(table_id, where));
}

void Parser::parseSetList(std::vector<std::string> &keys, std::vector<Variant> &values)
{
    keys.emplace_back(_token.toId());
    consume(Token::ID);
    consume(Token::EQ);
    values.emplace_back(consume(Token::OPERAND).toOperand());
    while (_token.type() == Token::COMMA)
    {
        consume(Token::COMMA);
        keys.emplace_back(consume(Token::ID).toId());
        consume(Token::EQ);
        values.emplace_back(_token.toOperand());
        consume(Token::OPERAND);
    }
}

void Parser::handlePendingOperator(Stack<Expr> &vals, Stack<Token::Type> &ops, Token::Type pendingOp)
{
    while (!ops.empty())
    {
        if (operatorPriority[ops.back()] >= operatorPriority[pendingOp])
        {
            switch (ops.top())
            {
                case Token::L_PAREN:    // pending op must be R_PAREN
                    if (pendingOp != Token::R_PAREN)
                        throw ParserError("Incomplete parenthesis.");
                    ops.pop();
                    return;
                case Token::PI: // pi
                    ops.pop();
                    vals.push(Expr(Token(Token::OPERAND, M_PI)));
                    break;
                case Token::NOT:
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
                case Token::ATAN:   // unary operator
                {
                    Expr expr(ops.pop(), std::vector<std::shared_ptr<Expr>>{std::make_shared<Expr>(vals.pop())});
                    vals.push(expr);
                    break;
                }
                default:    // binary operator
                {
                    Expr b(vals.pop());
                    Expr a(vals.pop());
                    Expr expr(ops.pop(), std::vector<std::shared_ptr<Expr>>
                            {std::make_shared<Expr>(a), std::make_shared<Expr>(b)});
                    vals.push(expr);
                    break;
                }
            }
        }
        else
            break;
    }
    ops.push(pendingOp);
}

Expr Parser::parseExpr()
{
    if (_token.type() == Token::MUL)
    {
        consume(Token::MUL);
        return Expr(Token(Token::ID, "*"));
    }
    Stack<Expr> vals;       // a stack of operands
    Stack<Token::Type> ops; // a stack of operators

    while (_token.type() != Token::END)
    {
        if (_token.type() == Token::L_PAREN) // sub where clause
        {
            consume(Token::L_PAREN);
            ops.push(Token::L_PAREN);
            continue;
        }
        if ((vals.empty() && ops.empty()) || (!ops.empty() && ops.top() == Token::L_PAREN))
            // initially or right after left parenthesis
        {
            if (_token.type() == Token::MUL)
                _token = Token(Token::ID, "*");
            if (_token.type() == Token::MINUS)
                vals.emplace_back(Expr(Token(Token::OPERAND, 0)));
        }
        if (_token.type() == Token::NULL_SQL)
            _token = Token(Token::OPERAND, Variant());
        if (!(isOperator(_token) || _token.type() == Token::ID || _token.type() == Token::OPERAND) ||
            (_token.type() == Token::R_PAREN && std::find(ops.begin(), ops.end(), Token::L_PAREN) == ops.end()) ||
            (_token.type() == Token::COMMA && std::find(ops.begin(), ops.end(), Token::L_PAREN) == ops.end()))
        {
            handlePendingOperator(vals, ops, Token::END);
            if (vals.size() != 1)
                throw ParserError("Invalid expression");
            return vals[0];
        }
        if (_token.type() == Token::COMMA)
            /* pass */ ;
        else if (isOperator(_token)) // operator
            handlePendingOperator(vals, ops, _token.type());
        else if (_token.type() == Token::ID || _token.type() == Token::OPERAND)
            vals.push(Expr(_token));
        else
            throw ParserError("Unexpected token" + Token::typeName(_token.type()) + " in an expression");
        consume(_token.type());
    }
    throw ParserError("Unexpected end");
}

std::shared_ptr<Statement> Parser::parseSelect()
{
    std::vector<std::shared_ptr<StatementSelect>> ss;
    while (_token.type() == Token::SELECT)
    {
        consume(Token::SELECT);
        std::vector<Expr> expressions = parseExprList();
        std::string fileName;
        std::vector<std::string> groupBy;
        Expr orderAttr, whereClause;
        if (_token.type() == Token::INTO)
        {
            consume(Token::INTO);
            consume(Token::OUTFILE);
            fileName = _token.toOperand().toStdString();
            consume(Token::OPERAND);
        }
        consume(Token::FROM);
        auto tableNames = parseColumnNameList();
        std::vector<Token::Type> joinTypes(tableNames.size(), Token::INNER);
        std::vector<Expr> onClauses(tableNames.size(), Expr(Token(Token::OPERAND, true)));
        // join
        while (_token.type() == Token::INNER || _token.type() == Token::LEFT || _token.type() == Token::RIGHT)
        {
            joinTypes.emplace_back(_token.type());
            consume(_token.type());
            consume(Token::JOIN);
            tableNames.emplace_back(consume(Token::ID).toId());
            if (_token.type() == Token::ON)
            {
                consume(Token::ON);
                onClauses.emplace_back(parseExpr());
            }
            else
            {
                onClauses.emplace_back(Expr(Token(Token::OPERAND, true)));
            }
        }
        if (_token.type() == Token::WHERE)
        {
            consume(Token::WHERE);
            whereClause = parseExpr();
        }
        if (_token.type() == Token::GROUP)
        {
            consume(Token::GROUP);
            consume(Token::BY);
            groupBy = parseColumnNameList();
        }
        if (_token.type() == Token::ORDER)
        {
            consume(Token::ORDER);
            consume(Token::BY);
            orderAttr = parseExpr();
        }
        bool isUnionAll = false;
        if (_token.type() == Token::UNION)
        {
            consume(Token::UNION);
            if (_token.type() == Token::ALL)
            {
                consume(Token::ALL);
                isUnionAll = true;
            }
        }
        ss.emplace_back(std::make_shared<StatementSelect>(tableNames, joinTypes, onClauses, fileName, expressions,
                                                          whereClause, groupBy, orderAttr, nullptr, isUnionAll));
    }
    consume(Token::SEMICOLON);
    for (size_t i = ss.size() - 1; i > 0; i--)
    {
        ss[i - 1]->setNext(ss[i]);
        ss[i]->setUnionAll(ss[i - 1]->isUnionAll());
    }
    return ss.front();
}

std::shared_ptr<Statement> Parser::parseLoad()
{
    consume(Token::LOAD);
    consume(Token::DATA);
    consume(Token::INFILE);
    std::string filename = _token.toOperand().toStdString();
    consume(Token::OPERAND);
    consume(Token::INTO);
    consume(Token::TABLE);
    std::string tableName = _token.toId();
    consume(Token::ID);
    consume(Token::L_PAREN);
    auto columns = parseColumnNameList();
    consume(Token::R_PAREN);
    consume(Token::SEMICOLON);
    return std::make_shared<StatementLoad>(StatementLoad(filename, tableName, columns));
}

std::shared_ptr<Statement> Parser::parseDrop()
{
    consume(Token::DROP);
    switch (_token.type())
    {
        case Token::TABLE:
        {
            consume(Token::TABLE);
            std::vector<std::string> tableNames = {consume(Token::ID).toId()};
            while (_token.type() == Token::COMMA)
            {
                consume(Token::COMMA);
                tableNames.emplace_back(consume(Token::ID).toId());
            }
            consume(Token::SEMICOLON);
            return std::make_shared<StatementDropTable>(StatementDropTable(tableNames));
        }
        case Token::DATABASE:
        {
            consume(Token::DATABASE);
            std::string dbName = _token.toId();
            consume(Token::ID);
            consume(Token::SEMICOLON);
            return std::make_shared<StatementDropDatabase>(StatementDropDatabase(dbName));
        }
        default:
            throw ParserError("Unexpected token near DROP.");
    }
}

std::shared_ptr<Statement> Parser::parseUse()
{
    consume(Token::USE);
    std::string dbName = _token.toId();
    consume(Token::ID);
    consume(Token::SEMICOLON);
    return std::make_shared<StatementUseDatabase>(StatementUseDatabase(dbName));
}

std::shared_ptr<Statement> Parser::parseShow()
{
    consume(Token::SHOW);
    switch (_token.type())
    {
        case Token::TABLES:
        {
            consume(Token::TABLES);
            consume(Token::SEMICOLON);
            return std::make_shared<StatementShowTables>(StatementShowTables());
        }
        case Token::DATABASES:
        {
            consume(Token::DATABASES);
            consume(Token::SEMICOLON);
            return std::make_shared<StatementShowDatabases>(StatementShowDatabases());
        }
        case Token::COLUMNS:
        {
            consume(Token::COLUMNS);
            consume(Token::FROM);
            std::string tableName = _token.toId();
            consume(Token::ID);
            consume(Token::SEMICOLON);
            return std::make_shared<StatementShowColumns>(StatementShowColumns(tableName));
        }
        default:
            throw ParserError("Unexpected token near SHOW");
    }
}

std::shared_ptr<Statement> Parser::parseUpdate()
{
    consume(Token::UPDATE);
    std::string tableName = _token.toId();
    consume(Token::ID);
    consume(Token::SET);
    std::vector<std::string> keys;
    std::vector<Variant> values;
    parseSetList(keys, values);
    Expr expr;
    switch (_token.type())
    {
        case Token::WHERE:
            consume(Token::WHERE);
            expr = parseExpr();
            break;
        case Token::SEMICOLON:
            break;
        default:
            throw ParserError("Unexpected token near where clause");
    }
    return std::make_shared<StatementUpdate>(StatementUpdate(tableName, keys, values, expr));
}

std::vector<Expr> Parser::parseExprList()
{
    std::vector<Expr> columns = {parseExpr()};
    while (_token.type() == Token::COMMA)
    {
        consume(Token::COMMA);
        columns.emplace_back(parseExpr());
    }
    return columns;
}

Token Parser::consume(Token::Type t)
{
    if (_token.type() != t)
        throw ParserError("Expected type " + Token::typeName(t) + ". Got " + Token::typeName(_token.type()));
    auto backup = _token;
    _token = _lexer.next();
    return backup;
}
