#pragma once

#include "variant.h"
#include <iostream>

/**
 * @brief This class provide methods for field management.
 * @details
 * A field contains a key name, a key type, an indicator of null, and an indicator of primary key.
 * The information can be read and modified if necessary.
 * Input & output methods are implemented. Support std::cout and DataStream I/O.
 */
class Field
{
public:
    Field() = default;

    Field(const std::string &key, Variant::Type type, bool isNull = true, bool isPrimary = false)
            : _key(key), _type(type), _isNull(isNull), _isPrimary(isPrimary)
    {}

    /**
     * @brief Get the key name of the field.
     * @return The key name.
     */
    const std::string &key() const
    { return _key; }

    /**
     * @brief Get the data type of the field.
     * @return The data type.
     */
    Variant::Type type() const
    { return _type; }

    /**
     * @brief Get whether the key is allowed to be null.
     * @return Whether the key is allowed to be null.
     */
    bool isNull() const
    { return _isNull; }

    /**
     * @brief Get whether the key is marked primary.
     * @return Whether the key is marked primary.
     */
    bool isPrimary() const
    { return _isPrimary; }

    /**
     * @brief Set the key name of the field.
     * @param key The key name.
     */
    void setKey(const std::string &key)
    { _key = key; }

    /**
     * @brief Set whether the key is allowed to be null.
     * @param isNull Indicator of whether the key is allowed to be null.
     */
    void setIsNull(bool isNull)
    { _isNull = isNull; }

    /**
     * @brief Set whether the key is primary.
     * @param isPrimary Indicator of whether the key is primary.
     */
    void setPrimary(bool isPrimary)
    { _isPrimary = isPrimary; }

    /**
     * @brief Standard output method.
     * @param out Output stream
     * @param f The field to print out
     * @return The output stream
     */
    friend std::ostream &operator<<(std::ostream &out, const Field &f)
    {
        return out << f._key << '\t' << f._type << '\t' << (f._isNull ? "YES" : "NO") << '\t'
                   << (f._isPrimary ? "PRI" : "") << "\tNULL";
    }

    /**
     * @brief Binary output method.
     * @param out Binary output stream.
     * @param f The field to print out.
     * @return The binary output stream.
     */
    friend DataStream &operator<<(DataStream &out, const Field &f)
    { return out << f._key << f._type << f._isNull << f._isPrimary; }

    /**
     * @brief Binary input method.
     * @param in Binary input stream.
     * @param f The destination field.
     * @return The binary input stream.
     */
    friend DataStream &operator>>(DataStream &in, Field &f)
    { return in >> f._key >> f._type >> f._isNull >> f._isPrimary; }

protected:
    std::string _key;   ///< key name
    Variant::Type _type = Variant::NONE;    ///< Field type.
    bool _isNull = true;    ///< Whether the key is allowed to be null.
    bool _isPrimary = false;    ///< Whether the key is primary.
};
