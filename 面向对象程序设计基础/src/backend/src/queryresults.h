#pragma once

#include <iostream>
#include <vector>
#include "entry.h"
#include "../../parser/src/expr.h"

/**
 * @brief Base class of query result
 */
class QueryResult
{
public:
    enum Type
    {
        NONE,
        SELECT,
    };

    /**
     * @brief Construct the query result with specific type.
     * @param type
     */
    QueryResult(Type type) : _type(type)
    {}

    /**
     * @brief Virtual trivial destructor.
     */
    virtual ~QueryResult() = default;

    /**
     * @brief Virtual print function.
     */
    virtual void print() const = 0;

protected:
    Type _type = NONE;  ///< Type of query result
};

/**
 * @brief Query result of select statement
 */
class QueryResultSelect : public QueryResult
{
public:
    QueryResultSelect() : QueryResult(SELECT)
    {}

    QueryResultSelect(const std::vector<Expr> &expressions, const std::vector<Entry> &entries,
                      const std::string &fileName)
            : QueryResult(SELECT), _expressions(expressions), _entries(entries), _fileName(fileName)
    {}

    virtual void print() const override
    {
        if (_entries.empty())
            return;
        for (auto &e : _expressions)
            std::cout << e << '\t';
        std::cout << std::endl;
        for (auto &r : _entries)
            std::cout << r << std::endl;
    }

    /**
     * @brief Print out the result on screen if file is empty, or dump the result to file if file is given.
     */
    void output()
    {
        if (_expressions.empty())
            return;
        if (_fileName.empty())
        {
            for (auto &c : _expressions)
                std::cout << c << '\t';
            std::cout << std::endl;
            printSelectResult(std::cout, _entries);
        }
        else
        {
            std::ofstream out(_fileName);
            printSelectResult(out, _entries);
            out.close();
        }
    }

    const std::vector<Expr> &expressions() const
    { return _expressions; }

    std::vector<Expr> &expressions()
    { return _expressions; }

    const std::vector<Entry> &entries() const
    { return _entries; }

    std::vector<Entry> &entries()
    { return _entries; }

    const std::string &fileName() const
    { return _fileName; }

protected:
    void printSelectResult(std::ostream &out, const std::vector<Entry> &result)
    {
        for (auto &entry: result)
        {
            for (auto &d: entry)
                out << d << '\t';
            out << std::endl;
        }
    }

protected:
    std::vector<Expr> _expressions; ///< Selected expressions.
    std::vector<Entry> _entries;    ///< Result entries.
    std::string _fileName;          ///< File name.
};
