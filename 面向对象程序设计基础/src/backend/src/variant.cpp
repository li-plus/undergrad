#include "variant.h"
#include "databaseexcept.h"

int Variant::compare(const Variant &v) const
{
    auto comType = commonType(_type, v.type());
    auto a = this->convertTo(comType);
    auto b = v.convertTo(comType);
    switch (comType)
    {
        case INT:
            return (a.toInt() < b.toInt() ? -1 : a.toInt() > b.toInt());
        case BOOL:
            return (a.toBool() < b.toBool() ? -1 : a.toBool() > b.toBool());
        case CHAR:
            return (a.toChar() < b.toChar() ? -1 : a.toChar() > b.toChar());
        case DOUBLE:
            return (a.toDouble() < b.toDouble() ? -1 : a.toDouble() > b.toDouble());
        case NONE:
            return (_type == NONE && v.type() == NONE) ? 0 : -1;
        case STRING:
            return (a.toStdString() < b.toStdString() ? -1 : a.toStdString() > b.toStdString());
        case DATE:
            return (a.toDate() < b.toDate() ? -1 : a.toDate() > b.toDate());
        case TIME:
            return (a.toTime() < b.toTime() ? -1 : a.toTime() > b.toTime());
        default:
            throw VariantError("Cannot compare types");
    }
}

std::ostream &operator<<(std::ostream &out, const Variant::Type &type)
{
    switch (type)
    {
        case Variant::BOOL:
            return out << "bool";
        case Variant::INT:
            return out << "int(11)";
        case Variant::DOUBLE:
            return out << "double";
        case Variant::CHAR:
            return out << "char(1)";
        case Variant::STRING:
            return out << "text";
        case Variant::NONE:
            return out << "NONE";
        case Variant::DATE:
            return out << "date";
        case Variant::TIME:
            return out << "time";
        default:
            return out;
    }
}

std::ostream &operator<<(std::ostream &os, const Variant &var)
{
    switch (var.type())
    {
        case Variant::INT:
            return os << var.toInt();
        case Variant::BOOL:
            return os << (var.toBool() ? "True" : "False");
        case Variant::CHAR:
            return os << var.toChar();
        case Variant::DOUBLE:
            os.setf(std::ios::fixed);
            return os << std::setprecision(4) << var.toDouble();
        case Variant::STRING:
            return os << var.toStdString();
        case Variant::DATE:
            return os << var.toDate();
        case Variant::TIME:
            return os << var.toTime();
        case Variant::NONE:
            return os << "NULL";
        default:
            return os;
    }
}

DataStream &operator<<(DataStream &out, const Variant &var)
{
    out << (char) var._type;
    switch (var._type)
    {
        case Variant::INT:
            return out << var.toInt();
        case Variant::BOOL:
            return out << (var.toBool() ? "True" : "False");
        case Variant::CHAR:
            return out << var.toChar();
        case Variant::DOUBLE:
            return out << var.toDouble();
        case Variant::STRING:
            return out << var.toStdString();
        case Variant::TIME:
            return out << var.toTime();
        case Variant::DATE:
            return out << var.toDate();
        default:
            return out;
    }
}

DataStream &operator>>(DataStream &in, Variant &var)
{
    char type;
    in >> type;
    var._type = (Variant::Type) type;
    switch (var._type)
    {
        case Variant::INT:
            return in >> var._basic.i;
        case Variant::BOOL:
            return in >> var._basic.b;
        case Variant::CHAR:
            return in >> var._basic.c;
        case Variant::DOUBLE:
            return in >> var._basic.d;
        case Variant::STRING:
            var._shared = std::make_shared<std::string>();
            return in >> *std::static_pointer_cast<std::string>(var._shared);
        case Variant::DATE:
            var._shared = std::make_shared<Date>();
            return in >> *std::static_pointer_cast<Date>(var._shared);
        case Variant::TIME:
            var._shared = std::make_shared<Time>();
            return in >> *std::static_pointer_cast<Time>(var._shared);
        default:
            return in;
    }
}

Variant Variant::convertTo(Variant::Type type) const
{
    switch (_type)
    {
        case INT:
            return convertTo(type, toInt());
        case DOUBLE:
            return convertTo(type, toDouble());
        case CHAR:
            return convertTo(type, toChar());
        case BOOL:
            return convertTo(type, toBool());
        case STRING:
            switch (type)
            {
                case DATE:
                    return Date(toStdString());
                case TIME:
                    return Time(toStdString());
                case CHAR:
                    return toStdString().front();
                case STRING:
                    return *this;
                default:
                    return Variant();
            }
        case DATE:
            if (type != DATE)
                return Variant();
            return *this;
        case TIME:
            if (type != TIME)
                return Variant();
            return *this;
        default:
            return Variant();
    }
}

Variant::Type Variant::commonType(Variant::Type a, Variant::Type b) const
{
    if (a == Variant::NONE || b == Variant::NONE)
        return Variant::NONE;
    if (a == Variant::DATE || b == Variant::DATE)
        return Variant::DATE;
    if (a == Variant::TIME || b == Variant::TIME)
        return Variant::TIME;
    if (a == Variant::STRING && b == Variant::STRING)
        return Variant::STRING;
    if (a == Variant::DOUBLE || b == Variant::DOUBLE)
        return Variant::DOUBLE;
    if (a == Variant::INT || b == Variant::INT)
        return Variant::INT;
    if (a == Variant::CHAR || b == Variant::CHAR)
        return Variant::CHAR;
    if (a == Variant::BOOL && b == Variant::BOOL)
        return Variant::BOOL;
    throw VariantError("Conversion to common type failed.");
}

Variant Variant::operator+(const Variant &v) const
{
    if (isNull() || v.isNull())
        return Variant();
    auto retType = commonType(_type, v._type);
    auto a = convertTo(retType);
    auto b = v.convertTo(retType);
    switch (a.type())
    {
        case INT:
            return a.toInt() + b.toInt();
        case DOUBLE:
            return a.toDouble() + b.toDouble();
        case CHAR:
            return a.toChar() + b.toChar();
        case BOOL:
            return a.toBool() + b.toBool();
        default:
            throw VariantError("Operator + failed");
    }
}

Variant Variant::operator-(const Variant &v) const
{
    if (isNull() || v.isNull())
        return Variant();
    auto retType = commonType(_type, v._type);
    auto a = convertTo(retType);
    auto b = v.convertTo(retType);
    switch (a.type())
    {
        case INT:
            return a.toInt() - b.toInt();
        case DOUBLE:
            return a.toDouble() - b.toDouble();
        case CHAR:
            return a.toChar() - b.toChar();
        case BOOL:
            return a.toBool() - b.toBool();
        default:
            throw VariantError("Operator - failed");
    }
}

Variant Variant::operator*(const Variant &v) const
{
    if (isNull() || v.isNull())
        return Variant();
    auto retType = commonType(_type, v._type);
    auto a = convertTo(retType);
    auto b = v.convertTo(retType);
    switch (a.type())
    {
        case INT:
            return a.toInt() * b.toInt();
        case DOUBLE:
            return a.toDouble() * b.toDouble();
        case CHAR:
            return a.toChar() * b.toChar();
        case BOOL:
            return a.toBool() * b.toBool();
        default:
            throw VariantError("Operator * failed");
    }
}

Variant Variant::operator/(const Variant &v) const
{
    if (isNull() || v.isNull())
        return Variant();
    return convertTo(DOUBLE).toDouble() / v.convertTo(DOUBLE).toDouble();
}

Variant Variant::operator%(const Variant &v) const
{
    if (isNull() || v.isNull())
        return Variant();
    auto comType = commonType(_type, v._type);
    auto a = convertTo(comType);
    auto b = v.convertTo(comType);
    switch (comType)
    {
        case INT:
            return (b.toInt() ? a.toInt() % b.toInt() : Variant());
        default:
            throw VariantError("Operator % failed");
    }
}

Variant Variant::operator^(const Variant &v) const
{
    if (isNull() || v.isNull())
        return Variant();
    return convertTo(BOOL).toBool() ^ v.convertTo(BOOL).toBool();
}

Variant &Variant::operator=(const Variant &v)
{
    _type = v._type;
    switch (v._type)
    {
        case STRING:
            _shared = std::make_shared<std::string>(*std::static_pointer_cast<std::string>(v._shared));
            break;
        case DATE:
            _shared = std::make_shared<Date>(*std::static_pointer_cast<Date>(v._shared));
            break;
        case TIME:
            _shared = std::make_shared<Time>(*std::static_pointer_cast<Time>(v._shared));
            break;
        default:    // basic type
            _basic = v._basic;
            break;
    }
    return *this;
}

Variant &Variant::operator=(Variant &&v) noexcept
{
    _type = v._type;
    _basic = v._basic;
    _shared.swap(v._shared);
    return *this;
}
