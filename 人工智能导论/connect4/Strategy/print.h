#pragma once

#include <iostream>
#include <vector>
#include <list>
#include <map>
using namespace std;


template <typename T>
static ostream &operator<<(ostream &out, const vector<T> &v)
{
    out << "vector( ";

    for (auto &i : v)
        out << i << ' ';

    out << ')';
    return out;
}

template <typename T>
static ostream &operator<<(ostream &out, const list<T>  &l)
{
    out << "list( ";

    for (auto &i : l)
        out << i << ' ';

    out << ')';
    return out;
}
template <typename K, typename V>
static ostream &operator<<(ostream & out, const pair<K, V> & p)
{
    return out << "pair(" << p.first << ", " << p.second << ')';
}
template <typename K, typename V>
static ostream & operator<<(ostream & out, const map<K, V> & m)
{
    out << "map( ";

    for (auto & p : m)
        out << p << ' ';

    return out << ')';
}
