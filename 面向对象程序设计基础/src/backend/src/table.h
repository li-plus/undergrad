#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "entry.h"
#include "field.h"
#include "print.h"
#include "datastream.h"
#include "utils.h"
#include "databaseexcept.h"
#include "queryresults.h"
#include "../../parser/src/expr.h"
#include "../../parser/src/statements.h"

/**
 * @brief Table class provides methods for table management and operation
 * @details
 * Support creating / showing / dropping / inserting / updating / deleting / selecting from tables.
 * Return QueryResult when entries is selected.
 * Input and output methods are implemented. Support std::cout and DataStream I/O.
 */
class Table
{
public:
    Table() = default;

    /**
     * @brief Construct a table with table name, fields and primary key.
     * @param tableName Table name.
     * @param fields All fields.
     * @param primaryKey Primary key name
     */
    Table(const std::string &tableName, const std::vector<Field> &fields, const std::string &primaryKey)
            : _name(tableName), _fields(fields), _primaryKey(primaryKey)
    {
        if (primaryKey.empty())
            _primaryKey = _fields.front().key();
    }

    /**
     * @brief Convert a key name to the index among the fields.
     * @param keyName Input key name.
     * @return The corresponding rank.
     */
    size_t keyToRank(const std::string &keyName) const
    {
        auto it = std::find_if(_fields.begin(), _fields.end(), [=](const Field &f) { return f.key() == keyName; });
        if (it == _fields.end())
            throw DatabaseError("invalid key " + keyName);
        return (size_t) (it - _fields.begin());
    }

    /**
     * @brief Get all entries.
     * @return All entries.
     */
    const std::vector<Entry> &entries() const
    { return _entries; }

    /**
     * @brief Get the reference to all entries.
     * @return The reference to all entries.
     */
    std::vector<Entry> &entries()
    { return _entries; }

    /**
     * @brief Get the table name.
     * @return The table name.
     */
    const std::string &name() const
    { return _name; }

    /**
     * @brief Get all fields of the table.
     * @return All fields of the table.
     */
    const std::vector<Field> &fields() const
    { return _fields; }

    /**
     * @brief Get all fields of the table.
     * @return The reference of all fields of the table.
     */
    std::vector<Field> &fields()
    { return _fields; }

    /**
     * @brief Get the primary key of the table.
     * @return Primary key of string type.
     */
    const std::string &primaryKey() const
    { return _primaryKey; }

    /**
     * @brief Convert an entry to std::map type.
     * @param entry The input entry.
     * @return The converted map.
     */
    std::map<std::string, Variant> entryToMap(const Entry &entry);

    /**
     * @brief Convert multiple entries to std::map type.
     * @param entries The input entries.
     * @return The converted map.
     */
    std::map<std::string, std::vector<Variant>> entryToMap(const std::vector<Entry> &entries);

    /**
     * @brief Insert an entry into a table
     * @param entry An entry of std::map format to be inserted.
     */
    void insertInto(const std::map<std::string, Variant> &entry);

    /**
     * @brief Select expressions from a table based on where clause, group-by clause, order-by clause.
     * @param expressions Selected expressions.
     * @param whereClause The condition to locate the entries.
     * @param fileName The possible output file. Output to screen if file name is empty.
     * @param groupBy The group-by expression.
     * @param orderBy The order-by expression.
     * @return The select result for printing out.
     */
    QueryResultSelect selectFrom(std::vector<Expr> expressions, const Expr &whereClause, const std::string &fileName,
                                 const std::vector<std::string> &groupBy, const Expr &orderBy);

    /**
     * @brief Separate the entries into several groups with different values on group-by columns.
     * @param result The input entries.
     * @param groupBy The group-by columns.
     * @return The grouped result.
     */
    std::vector<std::vector<Entry>> makeGroups(std::vector<Entry> &result, const std::vector<std::string> &groupBy);

    /**
     * @brief Show columns from this table.
     */
    void showColumns();

    /**
     * @brief Update specific keys with given values based on where clause.
     * @param keys Keys to update
     * @param values Destination values.
     * @param whereClause The condition to locate target entries.
     */
    void update(const std::vector<std::string> &keys, const std::vector<Variant> &values, const Expr &whereClause);

    /**
     * @brief Delete entries based on where clause.
     * @param whereClause The condition to locate target entries.
     */
    void deleteWhere(const Expr &whereClause);

    /**
     * @brief Binary output method.
     * @param out Output data stream
     * @param table Table to output.
     * @return Output data stream.
     */
    friend DataStream &operator<<(DataStream &out, const Table &table)
    { return out << table._name << table._fields << table._primaryKey << table._entries; }

    /**
     * @brief Binary input method.
     * @param in Input data stream.
     * @param table Destination table;
     * @return Input data stream.
     */
    friend DataStream &operator>>(DataStream &in, Table &table)
    { return in >> table._name >> table._fields >> table._primaryKey >> table._entries; }

    /**
     * @brief Standard output method.
     * @param out Output stream
     * @param table Table to output.
     * @return Output stream.
     */
    friend std::ostream &operator<<(std::ostream &out, const Table &table)
    { return out << "Table(" << table._name << ',' << table._fields << ',' << table._primaryKey << ')'; }

protected:
    std::string _name;              ///< Table name.
    std::vector<Field> _fields;     ///< All fields in a table.
    std::string _primaryKey;        ///< Primary key name
    std::vector<Entry> _entries;    ///< All entries.
};
