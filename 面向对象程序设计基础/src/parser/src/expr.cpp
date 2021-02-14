#include "expr.h"
#include <algorithm>
#include <regex>
#include "../../backend/src/variantmath.h"
#include "sqlexcept.h"

std::ostream &operator<<(std::ostream &out, const Expr &expr)
{
    if (expr.token().type() == Token::ID)
        return out << expr.token().toId();
    if (expr.token().type() == Token::OPERAND)
        return out << expr.token().toOperand();
    if (expr.token().isFunction())  // function
    {
        out << expr.token() << '('; // function name goes first.
        if (!expr._children.empty())
        {
            out << *expr._children[0];
            for (size_t c = 1; c < expr._children.size(); c++)
                out << ',' << *expr._children[c];
        }
        return out << ')';
    }
    out << "(";
    if (expr._children.size() == 1)
        out << expr.token() << *expr._children[0];
    else if (expr._children.size() == 2)
        out << *expr._children[0] << expr.token() << *expr._children[1];
    return out << ")";
}

Variant Expr::eval(const std::map<std::string, Variant> &varMap) const
{
    std::shared_ptr<Expr> left, right;
    if (!_children.empty())
    {
        left = _children.front();
        right = _children.back();
    }
    switch (_token.type())
    {
        case Token::NONE:
            return true;
        case Token::NULL_SQL:
            return Variant();
        case Token::OPERAND:
            return _token.toOperand();
        case Token::ID:
            return varMap.at(_token.toId());
        case Token::AND:
            return left->eval(varMap).toBool() && right->eval(varMap).toBool();
        case Token::OR:
            return left->eval(varMap).toBool() || right->eval(varMap).toBool();
        case Token::NOT:
            return !right->eval(varMap).toBool();
        case Token::XOR:
            return left->eval(varMap) ^ right->eval(varMap);
        case Token::GT:
            return left->eval(varMap) > right->eval(varMap);
        case Token::LT:
            return left->eval(varMap) < right->eval(varMap);
        case Token::EQ:
            return left->eval(varMap) == right->eval(varMap);
        case Token::NEQ:
            return left->eval(varMap) != right->eval(varMap);
        case Token::GEQ:
            return left->eval(varMap) >= right->eval(varMap);
        case Token::LEQ:
            return left->eval(varMap) <= right->eval(varMap);
        case Token::PLUS:
            return left->eval(varMap) + right->eval(varMap);
        case Token::MINUS:
            return left->eval(varMap) - right->eval(varMap);
        case Token::MUL:
            return left->eval(varMap) * right->eval(varMap);
        case Token::DIV:
            return left->eval(varMap) / right->eval(varMap);
        case Token::MOD:
            return left->eval(varMap) % right->eval(varMap);
        case Token::LIKE:
        {
            std::string pattern = right->eval(varMap).toStdString();
            pattern = std::regex_replace(pattern, std::regex(R"(([\*\.\?\+\$\^\[\]\(\)\{\}\|\\]))"), "\\$1");
            pattern = std::regex_replace(pattern, std::regex("%"), ".*");
            return std::regex_match(left->eval(varMap).toStdString(), std::regex(pattern));
        }
            // extra type
        case Token::ADDTIME:
            return left->eval(varMap).toTime() + right->eval(varMap).toInt();
        case Token::ADDDATE:
            return left->eval(varMap).toDate() + right->eval(varMap).toInt();
            // math
        case Token::ABS:
            return abs(right->eval(varMap));
        case Token::EXP:
            return exp(right->eval(varMap));
        case Token::FLOOR:
            return floor(right->eval(varMap));
        case Token::CEIL:
            return ceil(right->eval(varMap));
        case Token::LN:
            return ln(right->eval(varMap));
        case Token::LOG10:
            return log10(right->eval(varMap));
        case Token::SIN:
            return sin(right->eval(varMap));
        case Token::COS:
            return cos(right->eval(varMap));
        case Token::TAN:
            return tan(right->eval(varMap));
        case Token::ASIN:
            return asin(right->eval(varMap));
        case Token::ACOS:
            return acos(right->eval(varMap));
        case Token::ATAN:
            return atan(right->eval(varMap));
        default:
            throw ExprError("Expr::eval failed. Unexpected token " + Token::typeName(_token.type()));
    }
}

Variant Expr::eval(const std::map<std::string, std::vector<Variant>> &varMap) const
{
    switch (_token.type())
    {
        case Token::COUNT:
            if (_children.back()->token().toId() == "*")
            {
                return (int) varMap.begin()->second.size();
            }
            else
            {
                std::vector<Variant> col = varMap.at(_children.back()->token().toId());
                return (int) std::count_if(col.begin(), col.end(), [=](const Variant &v) {
                    return !v.isNull();
                });
            }
        default:
            throw ExprError("Expr::eval. Expected COUNT, got " + Token::typeName(_token.type()));
    }
}
