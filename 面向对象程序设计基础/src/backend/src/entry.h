#pragma once

#include <vector>
#include "variant.h"

/**
 * @brief Entry class organizes database entry into vector structure.
 * @details
 * This class publicly inherits std::vector<Variant>. Constructors are overwritten.
 * Input & output methods are implemented. Support std::cout and DataStream I/O.
 */
class Entry : public std::vector<Variant>
{
public:
    Entry() = default;

    Entry(const Entry &entry) : std::vector<Variant>(entry)
    {}

    Entry(Entry &&entry) : std::vector<Variant>(entry)
    {}

    Entry(size_t n, const Variant &v) : std::vector<Variant>(n, v)
    {}

    Entry(std::initializer_list<Variant> initList) : vector<Variant>(initList)
    {}

    Entry(const vector <Variant> &v) : vector<Variant>(v)
    {}

    Entry(vector <Variant> &&v) : vector<Variant>(std::move(v))
    {}

    Entry &operator=(const Entry &entry)
    {
        vector<Variant>::operator=(entry);
        return (*this);
    }

    Entry &operator=(Entry &&entry) noexcept
    {
        vector<Variant>::operator=(std::move(entry));
        return (*this);
    }

    friend std::ostream &operator<<(std::ostream &out, const Entry &entry)
    {
        for (auto &val : entry)
            out << val << '\t';
        return out;
    }

    friend DataStream &operator<<(DataStream &out, const Entry &entry)
    {
        out << entry.size();
        for (auto &val : entry)
            out << val;
        return out;
    }

    friend DataStream &operator>>(DataStream &in, Entry &entry)
    {
        size_t size = 0;
        in >> size;
        entry.resize(size);
        for (auto &e : entry)
            in >> e;
        return in;
    }
};
