#pragma once

#include <algorithm>
#include <string>
#include <list>
#include <cstring>
#include <vector>
#include <dirent.h>
#include <iostream>
#include <fstream>
#ifdef WIN32
#include <io.h>
#include <direct.h>
#elif linux
#include <unistd.h>
#include <dirent.h>
#endif

/**
 * trim spaces at front
 * @param s input & output string
 */
static inline std::string ltrim(const std::string &str)
{
    std::string s(str);
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
                return !std::isspace(ch);
            }));
    return s;
}

/**
 * trim spaces at end
 * @param s input & output string
 */
static inline std::string rtrim(const std::string &str)
{
    std::string s(str);
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
                return !std::isspace(ch);
            }).base(), s.end());
    return s;
}

/**
  * trim spaces from both ends
  * @param s input & output string
  */
static inline std::string trim(const std::string &s)
{
    return ltrim(rtrim(s));
}

/**
 * convert the entire string to lower case
 * @param s input string
 * @return converted string in upper case
 */
static inline std::string toLower(const std::string &s)
{
    std::string ret(s);
    std::transform(s.begin(), s.end(), ret.begin(), ::tolower);
    return ret;
}

/**
 * conert the entire string to upper case
 * @param s input string
 * @return converted string in upper case
 */
static inline std::string toUpper(const std::string &s)
{
    std::string ret(s);
    std::transform(s.begin(), s.end(), ret.begin(), ::toupper);
    return ret;
}

/**
 * get random int in [0, n)
 * @param n upper bound
 * @return random int
 */
static inline int dice(int n)
{
    return rand() % n;
}

/**
 * split string with separator sep
 * @param s input string
 * @param sep separator
 * @return split string
 */
static std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> res;
    std::string tmp(s);
    while (true)
    {
        size_t firstSep = tmp.find_first_of(sep);
        if (firstSep == std::string::npos)
            break;
        res.emplace_back(tmp.substr(0, firstSep));
        tmp = tmp.substr(firstSep + 1);
    }
    res.emplace_back(tmp);
    return res;
}

/**
 * slice a vector
 * @tparam T data type
 * @param vec input vector
 * @param indices slice indices
 * @return sliced vector
 */
template <typename T>
static std::vector<T> slice(const std::vector<T> &vec, const std::vector<size_t> &indices)
{
    std::vector<T> res;
    for (auto &i : indices)
        res.emplace_back(vec[i]);
    return res;
}

 /**
  * @brief Merge sort IN PLACE
  * @tparam Iter Iter Iterator type
  * @tparam Pr Comparer type
  * @param begin Begin iterator
  * @param end End iterator
  * @param buffer Buffer for temporary merge result
  * @param pred Comparer
  */
template<typename Iter, typename Pr>
static void mergeSort(Iter begin, Iter end, Iter buffer, Pr pred)
{
    if (end - begin < 2)
        return;

    Iter mid = begin + (end - begin) / 2;
    mergeSort(begin, mid, buffer, pred);
    mergeSort(mid, end, buffer, pred);
    std::merge(begin, mid, mid, end, buffer, pred);
    std::copy(buffer, buffer + (end - begin), begin);
}
