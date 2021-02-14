#pragma once

#include <fstream>
#include <vector>
#include <list>
#include <cstring>
#include <map>
#include <iostream>

/**
 * @brief This class provides methods to serialize data into binary files
 * @details
 * It supports all c++ basic types, std::string, and all enum class.
 * It inherits std::fstream, and always open files in binary mode.
 * Operator >> and << are overloaded for input and output.
 *
 * Example
 * ```
 * int a;
 * DataStream ds(a.txt);
 * ds << a; // write binary data
 * ds >> a; // read binary data
 * ```
 */
class DataStream : public std::fstream
{
public:
    DataStream() : std::fstream()
    {}

    DataStream(const char *fileName, std::ios::openmode mode = std::ios::in | std::ios::out)
            : std::fstream(fileName, mode | std::ios::binary)
    {}

    DataStream(const std::string &file, std::ios::openmode mode = std::ios::in | std::ios::out)
            : std::fstream(file, mode | std::ios::binary)
    {}

    DataStream(const DataStream &) = delete;

    DataStream(DataStream &&rhs) noexcept : std::fstream(std::move(rhs))
    {}

    DataStream &operator=(const DataStream &) = delete;

    DataStream &operator=(DataStream &&rhs) noexcept
    {
        std::fstream::operator=(std::move(rhs));
        return *this;
    }

    void open(const std::string &filename, std::ios::openmode mode = std::ios::in | std::ios::out)
    { std::fstream::open(filename, mode | std::ios::binary); }

    // output basic type
    DataStream &operator<<(bool b)
    { return writeBasicType(b); }

    DataStream &operator<<(char c)
    { return writeBasicType(c); }

    DataStream &operator<<(unsigned char uc)
    { return writeBasicType(uc); }

    DataStream &operator<<(short s)
    { return writeBasicType(s); }

    DataStream &operator<<(unsigned short us)
    { return writeBasicType(us); }

    DataStream &operator<<(int i)
    { return writeBasicType(i); }

    DataStream &operator<<(unsigned int ui)
    { return writeBasicType(ui); }

    DataStream &operator<<(long l)
    { return writeBasicType(l); }

    DataStream &operator<<(unsigned long ul)
    { return writeBasicType(ul); }

    DataStream &operator<<(long long ll)
    { return writeBasicType(ll); }

    DataStream &operator<<(unsigned long long ull)
    { return writeBasicType(ull); }

    DataStream &operator<<(float f)
    { return writeBasicType(f); }

    DataStream &operator<<(double d)
    { return writeBasicType(d); }

    DataStream &operator<<(long double ld)
    { return writeBasicType(ld); }

    // input basic type
    DataStream &operator>>(bool &b)
    { return readBasicType(b); }

    DataStream &operator>>(char &c)
    { return readBasicType(c); }

    DataStream &operator>>(unsigned char &uc)
    { return readBasicType(uc); }

    DataStream &operator>>(short &s)
    { return readBasicType(s); }

    DataStream &operator>>(unsigned short &us)
    { return readBasicType(us); }

    DataStream &operator>>(int &i)
    { return readBasicType(i); }

    DataStream &operator>>(unsigned int &ui)
    { return readBasicType(ui); }

    DataStream &operator>>(long &l)
    { return readBasicType(l); }

    DataStream &operator>>(unsigned long &ul)
    { return readBasicType(ul); }

    DataStream &operator>>(long long &ll)
    { return readBasicType(ll); }

    DataStream &operator>>(unsigned long long &ull)
    { return readBasicType(ull); }

    DataStream &operator>>(float &f)
    { return readBasicType(f); }

    DataStream &operator>>(double &d)
    { return readBasicType(d); }

    DataStream &operator>>(long double &ld)
    { return readBasicType(ld); }

    // vector I/O
    template<typename T>
    DataStream &operator<<(const std::vector<T> &vec)
    {
        (*this) << vec.size();
        for (auto &v : vec)
            (*this) << v;
        return *this;
    }

    template<typename T>
    DataStream &operator>>(std::vector<T> &vec)
    {
        size_t size = 0;
        (*this) >> size;
        vec.resize(size);
        for (auto &it : vec)
            (*this) >> it;
        return *this;
    }

    // list I/O
    template<typename T>
    DataStream &operator<<(const std::list<T> &ls)
    {
        (*this) << ls.size();
        for (auto &it : ls)
            (*this) << it;
        return *this;
    }

    template<typename T>
    DataStream &operator>>(std::list<T> &ls)
    {
        size_t size = 0;
        (*this) >> size;
        ls.resize(size);
        for (auto &it : ls)
            (*this) >> it;
        return *this;
    }

    // string I/O
    DataStream &operator<<(const char *s)
    {
        size_t size = strlen(s) + 1;
        (*this) << size;
        std::fstream::write(s, size);
        return *this;
    }

    DataStream &operator>>(char *s)
    {
        size_t size = 0;
        (*this) >> size;
        std::fstream::read(s, size);
        return *this;
    }

    DataStream &operator<<(const std::string &s)
    {
        (*this) << s.size();
        std::fstream::write(s.c_str(), s.size());
        return *this;
    }

    DataStream &operator>>(std::string &s)
    {
        size_t size = 0;
        (*this) >> size;
        s.resize(size);
        std::fstream::read(&s[0], size);
        return *this;
    }

    // pair
    template<typename K, typename V>
    DataStream &operator<<(const std::pair<K, V> &p)
    {
        return *this << p.first << p.second;
    }

    template<typename K, typename V>
    DataStream &operator>>(std::pair<K, V> &p)
    {
        return *this >> p.first >> p.second;
    }

    // map I/O
    template<typename K, typename V>
    DataStream &operator<<(const std::map<K, V> &m)
    {
        *this << m.size();
        for (auto &p : m)
            *this << p;
        return *this;
    }

    template<typename K, typename V>
    DataStream &operator>>(std::map<K, V> &m)
    {
        size_t size = 0;
        *this >> size;
        std::pair<K, V> p;
        for (size_t i = 0; i < size; i++)
        {
            *this >> p;
            m.insert(p);
        }
        return *this;
    }

protected:
    template<typename T>
    DataStream &writeBasicType(const T &val)
    {
        std::fstream::write((char *) &val, sizeof(val));
        return *this;
    }

    template<typename T>
    DataStream &readBasicType(T &val)
    {
        std::fstream::read((char *) &val, sizeof(val));
        return *this;
    }
};

template<typename T>
static DataStream &operator<<(typename std::enable_if<std::is_enum<T>::value, DataStream>::type &out, const T &e)
{
    return out << static_cast<typename std::underlying_type<T>::type>(e);
}

template<typename T>
static DataStream &operator>>(typename std::enable_if<std::is_enum<T>::value, DataStream>::type &in, T &e)
{
    typename std::underlying_type<T>::type buf;
    in >> buf;
    e = static_cast<T>(buf);
    return in;
}
