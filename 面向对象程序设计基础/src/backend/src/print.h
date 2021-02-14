#pragma once

#include <iostream>
#include <vector>
#include <list>
#include <map>

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v)
{
    out << "vector( ";
    for (auto &i : v)
        out << i << ' ';
    return out << ')';
}

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::list<T> &l)
{
    out << "list( ";
    for (auto &i : l)
        out << i << ' ';
    return out << ')';
}

template<typename K, typename V>
std::ostream &operator<<(std::ostream &out, const std::pair<K, V> &p)
{
    return out << "pair(" << p.first << ", " << p.second << ')';
}

template<typename K, typename V>
std::ostream &operator<<(std::ostream &out, const std::map<K, V> &m)
{
    out << "map( ";
    for (auto &p : m)
        out << p << ' ';
    return out << ')';
}
