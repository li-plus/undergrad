#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <exception>
#include "utils.h"
#include "simsql.h"
#include "table.h"
#include "print.h"
#include "field.h"
#include "datastream.h"
#include "fileio.h"
#include "../../parser/src/expr.h"
#include "queryresults.h"
#include "databaseexcept.h"
#include "../../parser/src/statements.h"

/**
 * @brief Database class provides methods for table operations.
 * @details
 * Support creating / showing / dropping / inserting / updating / deleting / selecting from tables.
 * In terms of table operations, it calls Table class to execute.
 */
class Database
{
public:
    Database() = default;

    /**
     * @brief Construct a database with specific name
     * @param dbName Database name.
     */
    Database(const std::string &dbName) : _dbName(dbName)
    {}

    /**
     * @brief Get names of all tables.
     * @return List of table names.
     */
    std::vector<std::string> tableNames() const;

    /**
     * @brief Create a table with specific information.
     * @param tableName Name of the table.
     * @param fields Fields of the table.
     * @param primaryKey Primary key of the table.
     */
    void createTable(const std::string &tableName, const std::vector<Field> &fields, const std::string &primaryKey);

    /**
     * @brief Show columns from specific table
     * @param tableName table name.
     */
    void showColumnsFrom(const std::string &tableName);

    /**
     * @brief Show all tables.
     */
    void showTables() const;

    /**
     * @brief Drop specific table
     * @param tableName Table name.
     */
    void dropTable(const std::string &tableName);

    /**
     * @brief Insert an entry into a table
     * @param tableName Table name.
     * @param entry An entry.
     */
    void insertInto(const std::string &tableName, const std::map<std::string, Variant> &entry);

    /**
     * @brief Execute select statement with union clause.
     * @param head  Head of the select statement list.
     * @return The select result.
     */
    QueryResultSelect selectFromUnion(const std::shared_ptr<StatementSelect> &head);

    /**
     * @brief Update the table.
     * @param tableName Table name.
     * @param keys The keys to update.
     * @param values The specific values.
     * @param whereClause A where clause
     */
    void update(const std::string &tableName, const std::vector<std::string> &keys, const std::vector<Variant> &values,
                const Expr &whereClause);

    /**
     * @brief Delete entries from a table
     * @param tableName Table name.
     * @param whereClause A where clause
     */
    void deleteFrom(const std::string &tableName, const Expr &whereClause);

    /**
     * @brief Load data from file
     * @param fileName File name
     * @param tableName Table name
     * @param columns Destination columns
     */
    void loadData(const std::string &fileName, const std::string &tableName, const std::vector<std::string> &columns);

    /**
     * @brief Binary output method
     * @param out output datastream
     * @param db Database
     * @return output datastream
     */
    friend DataStream &operator<<(DataStream &out, const Database &db)
    { return out << db._dbName << db._tables; }

    /**
     * @brief Binary input method
     * @param in input datastream
     * @param db Database
     * @return output datastream
     */
    friend DataStream &operator>>(DataStream &in, Database &db)
    { return in >> db._dbName >> db._tables; }

protected:
    /**
     * @brief Determine whether a specific table exists.
     * @param tableName Table name.
     * @return whether a specific table exists.
     */
    bool isTable(const std::string &tableName) const
    { return _tables.find(tableName) != _tables.end(); }

    /**
     * @brief Assert a specific table exists.
     * @param tableName table name.
     */
    void assertTableExist(const std::string &tableName) const;

    /**
     * @brief Assert a specific table does not exist.
     * @param tableName table name.
     */
    void assertTableNotExist(const std::string &tableName) const;

    /**
     * @brief Select an expression from one or multiple tables based on various conditions.
     * @param tableNames All the selected tables.
     * @param joinTypes Join types of every two tables.
     * @param onClauses On clauses of join expressions.
     * @param expressions Selected expressions.
     * @param whereClause Where clause
     * @param fileName Output file name.
     * @param groupBy The column names to group by.
     * @param orderBy The expression to sort by.
     * @return The result of select query.
     */
    QueryResultSelect
    selectFrom(const std::vector<std::string> &tableNames, const std::vector<Token::Type> &joinTypes,
               const std::vector<Expr> &onClauses, const std::vector<Expr> &expressions, const Expr &whereClause,
               const std::string &fileName, const std::vector<std::string> &groupBy, const Expr &orderBy);

    /**
     * @brief Join two table
     * @param left Left table
     * @param right Right table
     * @param joinType Join policy. Should be INNER, LEFT or RIGHT.
     * @param onClause On clause of the join expression.
     * @return The joined table.
     */
    Table crossJoin(const Table &left, const Table &right, Token::Type joinType, const Expr &onClause);

protected:
    std::string _dbName;                    ///< Name of this database.
    std::map<std::string, Table> _tables;   ///< All belonging tables.
};
