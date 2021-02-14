#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include "datastream.h"
#include "variantextra.h"

/**
 * @brief Variant class unifies variants of different types into one class.
 * @details
 * The Variant class support types of bool, char, int, double, std::string, date and time.
 * Construct Variant directly with data of a specific type,
 * and retrieve the data via toBool(), toChar(), etc.
 * Basic c++ type will be stored within a union.
 * Custom type will be recorded into a shared pointer.
 * Always conduct deep copy during copy construction and copy assignment by default.
 * Move constructor and move assignment operator is also implemented.
 * Input & output methods are implemented. Support std::cout and DataStream I/O.
 */
class Variant
{
public:
    enum Type
    {
        NONE,
        INT,
        BOOL,
        CHAR,
        DOUBLE,
        STRING,
        DATE,
        TIME,
    };

    /**
     * @brief Default trivial constructor.
     */
    Variant() = default;

    /**
     * @brief Copy constructor
     * @param v Another variant.
     */
    Variant(const Variant &v)
    { *this = v; }

    /**
     * @brief Move constructor.
     * @param v Another variant.
     */
    Variant(Variant &&v)
    { *this = std::move(v); }

    /**
     * @brief A trivial virtual destructor.
     */
    virtual ~Variant() = default;

    /**
     * @brief Copy assignment operator.
     * @param v Another variant.
     * @return The reference of this variant.
     */
    Variant &operator=(const Variant &v);

    /**
     * @brief Move assignment operator
     * @param v Another variant.
     * @return The reference of this variant.
     */
    Variant &operator=(Variant &&v) noexcept;

    /**
     * @brief Get the variant type
     * @return The variant type.
     */
    Type type() const
    { return _type; }

    /**
     * @brief Determine whether the variant is NULL.
     * @return Whether the variant is NULL.
     */
    bool isNull() const
    { return _type == NONE; }

    /**
     * @brief Construct variant with an integer.
     * @param i An integer.
     */
    Variant(int i) : _type(INT)
    { _basic.i = i; }

    /**
     * @brief Construct variant with a bool.
     * @param b A bool.
     */
    Variant(bool b) : _type(BOOL)
    { _basic.b = b; }

    /**
     * @brief Construct variant with a char.
     * @param c A char.
     */
    Variant(char c) : _type(CHAR)
    { _basic.c = c; }

    /**
     * @brief Construct variant with a double. If d is nan or inf, variant will be set to NULL.
     * @param d A double.
     */
    Variant(double d) : _type(DOUBLE)
    { std::isfinite(d) ? _basic.d = d : _type = NONE; }

    /**
     * @brief Construct variant with a std::string.
     * @param s A string.
     */
    Variant(const std::string &s) : _type(STRING), _shared(std::make_shared<std::string>(s))
    {}

    /**
     * @brief Construct variant with a string
     * @param s A string
     */
    Variant(const char *s) : _type(STRING), _shared(std::make_shared<std::string>(s))
    {}

    /**
     * @brief Construct variant with a date.
     * @param date A date.
     */
    Variant(const Date &date) : _type(DATE), _shared(std::make_shared<Date>(date))
    {}

    /**
     * @brief Construct variant with a time.
     * @param time A time.
     */
    Variant(const Time &time) : _type(TIME), _shared(std::make_shared<Time>(time))
    {}

    /**
     * @brief Convert to bool value. Variant type must be BOOL to return the correct value.
     * @return The bool value.
     */
    bool toBool() const
    { return _basic.b; }

    /**
     * @brief Convert to int value. Variant type must be INT to return the correct value.
     * @return The int value.
     */
    int toInt() const
    { return _basic.i; }

    /**
     * @brief Convert to char value. Variant type must be CHAR to return the correct value.
     * @return The char value.
     */
    char toChar() const
    { return _basic.c; }

    /**
     * @brief Convert to double value. Variant type must be DOUBLE to return the correct value.
     * @return The double value.
     */
    double toDouble() const
    { return _basic.d; }

    /**
     * @brief Convert to string value. Variant type must be STRING to return the correct value.
     * @return The string value.
     */
    std::string toStdString() const
    { return *std::static_pointer_cast<std::string>(_shared); }

    /**
     * @brief Convert to date value. Variant type must be DATE to return the correct value.
     * @return The date value.
     */
    Date toDate() const
    { return *std::static_pointer_cast<Date>(_shared); }

    /**
     * @brief Convert to time value. Variant type must be TIME to return the correct value.
     * @return The time value.
     */
    Time toTime() const
    { return *std::static_pointer_cast<Time>(_shared); }

    /**
     * @brief Determine whether this variant is less than another.
     * @param v Another variant.
     * @return Whether this variant is less than another.
     */
    bool operator<(const Variant &v) const
    { return compare(v) < 0; }

    /**
     * @brief Determine whether this variant is equal to another.
     * @param v Another variant.
     * @return Whether this variant is equal to another.
     */
    bool operator==(const Variant &v) const
    { return compare(v) == 0; }

    /**
     * @brief Determine whether this variant is greater than another.
     * @param v Another variant.
     * @return Whether this variant is greater than another.
     */
    bool operator>(const Variant &v) const
    { return compare(v) > 0; }

    /**
     * @brief Determine whether this variant is unequal to another.
     * @param v Another variant.
     * @return Whether this variant is unequal to another.
     */
    bool operator!=(const Variant &v) const
    { return compare(v) != 0; }

    /**
     * @brief Determine whether this variant is greater or equal to another.
     * @param v Another variant.
     * @return Whether this variant is greater or equal to another.
     */
    bool operator>=(const Variant &v) const
    { return compare(v) >= 0; }

    /**
     * @brief Determine whether this variant is less than or equal to another.
     * @param v Another variant.
     * @return Whether this variant is less than or equal to another.
     */
    bool operator<=(const Variant &v) const
    { return compare(v) <= 0; }

    /**
     * @brief Addition operator.
     * @param v Another variant.
     * @return *this + v.
     */
    Variant operator+(const Variant &v) const;

    /**
     * @brief Subtraction operator.
     * @param v Another variant.
     * @return *this - v.
     */
    Variant operator-(const Variant &v) const;

    /**
     * @brief Multiplication operator.
     * @param v Another variant.
     * @return *this * v.
     */
    Variant operator*(const Variant &v) const;

    /**
     * @brief Division operator.
     * @param v Another variant.
     * @return *this / v.
     */
    Variant operator/(const Variant &v) const;

    /**
     * @brief Mod operator.
     * @param v Another variant.
     * @return *this % v.
     */
    Variant operator%(const Variant &v) const;

    /**
     * @brief Xor operator.
     * @param v Another variant.
     * @return *this xor v.
     */
    Variant operator^(const Variant &v) const;

    /**
     * @brief convert data to variant of specific type
     * @tparam T data type
     * @param type destination Variant type
     * @param data input data
     * @return data of specific type. return null variant if conversion failed.
     */
    template<typename T>
    Variant convertTo(Type type, T data) const;

    /**
     * @brief Convert to specific type
     * @param type Destination type
     * @return Variant of specific type. return null variant if conversion failed.
     */
    Variant convertTo(Type type) const;

    /**
     * @brief Compare *this and variant v
     * @param v another instance of variant
     * @return 1 if *this > v, 0 if *this == v, -1 if *this < v
     */
    int compare(const Variant &v) const;

    /**
     * @brief Standard output method for variant type.
     * @param out Standard output stream.
     * @param type Variant type.
     * @return Standard output stream.
     */
    friend std::ostream &operator<<(std::ostream &out, const Type &type);

    /**
     * @brief Standard output method.
     * @param out Standard output stream.
     * @param var A variant.
     * @return Standard output stream.
     */
    friend std::ostream &operator<<(std::ostream &out, const Variant &var);

    /**
     * @brief Binary output method.
     * @param out Binary output stream.
     * @param var A variant.
     * @return Binary output stream.
     */
    friend DataStream &operator<<(DataStream &out, const Variant &var);

    /**
     * @brief Binary input method.
     * @param in Binary input stream.
     * @param var A variant.
     * @return Binary input stream.
     */
    friend DataStream &operator>>(DataStream &in, Variant &var);

protected:
    /**
     * @brief Get the type of operation result between a and b.
     * @param a Type of an operand.
     * @param b Type of another operand.
     * @return The type of operation result between a and b.
     */
    Type commonType(Type a, Type b) const;

protected:
    Type _type = NONE;  ///< Variant type. NONE by default.
    union
    {
        int i;
        bool b;
        char c;
        double d;
    } _basic;   ///< A union of c++ basic type.
    std::shared_ptr<void> _shared;  ///< A shared_ptr for custom type.
};

template<typename T>
Variant Variant::convertTo(Variant::Type type, T data) const
{
    switch (type)
    {
        case Variant::INT:
            return Variant((int) data);
        case Variant::DOUBLE:
            return Variant((double) data);
        case Variant::CHAR:
            return Variant((char) data);
        case Variant::BOOL:
            return Variant((bool) data);
        default:
            return Variant();
    }
}
