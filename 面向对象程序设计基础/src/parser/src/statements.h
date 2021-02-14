#pragma once

#include "expr.h"
#include <set>
#include <iostream>
#include <string>
#include <map>
#include "../../backend/src/print.h"
#include "../../backend/src/field.h"

/**
 * @brief The base class of all statements.
 * @details
 * Consists of identifier and statement type.
 * Create the class for a specific kind of statement by inheriting from this basic class.
 */
class Statement
{
public:
    enum StatementType
    {
        NONE = 0,
        CREATE_TABLE,
        CREATE_DATABASE,
        USE_DATABASE,
        SHOW_TABLES,
        SHOW_DATABASES,
        SHOW_COLUMNS,
        DROP_TABLE,
        DROP_DATABASE,
        SELECT,
        INSERT,
        UPDATE,
        DELETE,
        LOAD
    };

    /**
     * @brief Constructor with identifier and statement type
     * @param id    Identifier.
     * @param type  Statement type.
     */
    Statement(const std::string &id, StatementType type) : _id(id), _type(type)
    {}

    /**
     * @brief A trivial virtual destructor.
     */
    virtual ~Statement() = default;

    /**
     * @brief Get the identifier
     * @return The identifier.
     */
    const std::string &id() const
    { return _id; }

    /**
     * @brief Get the statement type.
     * @return Statement type.
     */
    virtual StatementType type() const
    { return _type; }

    /**
     * @brief Get the type name as a string.
     * @return the type name as a string.
     */
    virtual std::string typeString() const
    { return typeStringMap[_type]; }

    /**
     * @brief Print out the statement.
     */
    virtual void print() const
    { std::cout << "StatementBase " << typeString() << " id:" << _id << std::endl; }

public:
    static std::map<StatementType, std::string> typeStringMap;  ///< A type-string map.
protected:
    std::string _id;            ///< Identifier of the corresponding table or database.
    StatementType _type = NONE; ///< Type of the statement.
};

/**
 * @brief Statement create table.
 */
class StatementCreateTable : public Statement
{
public:
    StatementCreateTable(const std::string &tableName, const std::vector<Field> &fields, const std::string &priKey)
            : Statement(tableName, CREATE_TABLE), _fields(fields), _priKey(priKey)
    {}

    const std::vector<Field> &fields() const
    { return _fields; }

    const std::string &primaryKey() const
    { return _priKey; }

    virtual void print() const override
    {
        Statement::print();
        std::cout << "Primary Key " << _priKey << "\nFields " << _fields << std::endl;
    }

protected:
    const std::vector<Field> _fields;   ///< fields of the table
    std::string _priKey;    ///< the primary key of the table
};

/**
 * @brief Statement load.
 */
class StatementLoad : public Statement
{
public:
    StatementLoad(const std::string &fileName, const std::string &tableName, const std::vector<std::string> &columns)
            : Statement(tableName, LOAD), _fileName(fileName), _columns(columns)
    {}

    const std::string &tableName() const
    { return _id; }

    const std::string &fileName() const
    { return _fileName; }

    const std::vector<std::string> &columns() const
    { return _columns; }

    virtual void print() const override
    {
        Statement::print();
        std::cout << "Load data. File name:" << fileName() << std::endl << "table:" << tableName() << std::endl
                  << "columns:" << _columns << std::endl;
    }

protected:
    std::string _fileName;  ///< name of file to load.
    std::vector<std::string> _columns;///< columns of destination table.
};

/**
 * @brief Statement create database.
 */
class StatementCreateDatabase : public Statement
{
public:
    StatementCreateDatabase(const std::string &id) : Statement(id, CREATE_DATABASE)
    {}
};

/**
 * @brief Statement insert.
 */
class StatementInsert : public Statement
{
public:
    StatementInsert(const std::string &tableName, const std::map<std::string, Variant> &entry)
            : Statement(tableName, INSERT), _entry(entry)
    {}

    const std::map<std::string, Variant> &entry() const
    { return _entry; }

    virtual void print() const override
    {
        Statement::print();
        std::cout << "Entry:" << _entry << std::endl;
    }

protected:
    std::map<std::string, Variant> _entry;  ///< The entry to be inserted.
};

/**
 * @brief Statement delete.
 */
class StatementDelete : public Statement
{
public:
    StatementDelete(const std::string &tableName, const Expr &whereClause) : Statement(tableName, DELETE),
                                                                             _whereClause(whereClause)
    {}

    const Expr &whereClause() const
    { return _whereClause; }

    virtual void print() const override
    {
        Statement::print();
        std::cout << "where clause:" << _whereClause << std::endl;
    }

protected:
    Expr _whereClause;  ///< A where clause
};

/**
 * @brief Statement select.
 */
class StatementSelect : public Statement, public std::enable_shared_from_this<StatementSelect>
{
public:
    StatementSelect(const std::vector<std::string> &tableNames, const std::vector<Token::Type> &joinTypes,
                    const std::vector<Expr> &onClauses, const std::string &filename,
                    const std::vector<Expr> &expressions, const Expr &whereClause,
                    const std::vector<std::string> &groupBy, const Expr &orderBy,
                    const std::shared_ptr<StatementSelect> &next, bool isUnionAll)
            : Statement("", SELECT), _tableNames(tableNames), _joinTypes(joinTypes), _onClauses(onClauses),
              _filename(filename), _expressions(expressions), _whereClause(whereClause), _groupBy(groupBy),
              _orderBy(orderBy), _next(next), _isUnionAll(isUnionAll)
    {}

    const std::vector<std::string> &tableNames() const
    { return _tableNames; }

    const std::vector<Token::Type> &joinTypes() const
    { return _joinTypes; }

    const std::vector<std::string> &groupBy() const
    { return _groupBy; }

    const Expr &orderBy() const
    { return _orderBy; }

    const std::vector<Expr> &onClauses() const
    { return _onClauses; }

    void setUnionAll(bool b)
    { _isUnionAll = b; }

    bool isUnionAll() const
    { return _isUnionAll; }

    const std::vector<Expr> &expressions() const
    { return _expressions; }

    const Expr &whereClause() const
    { return _whereClause; }

    const std::string &fileName() const
    { return _filename; }

    const std::shared_ptr<StatementSelect> &next() const
    { return _next; }

    void setNext(const std::shared_ptr<StatementSelect> &next)
    { _next = next; }

    virtual void print() const override
    {
        Statement::print();
        for (auto ptr = this->shared_from_this(); ptr; ptr = ptr->_next)
        {
            std::cout << "next:" << std::endl
                      << "filename:" << ptr->_filename << std::endl
                      << "join types:" << ptr->joinTypes() << std::endl
                      << "on clauses:" << ptr->onClauses() << std::endl
                      << "expressions:" << ptr->_expressions << std::endl
                      << "tables:" << ptr->_tableNames << std::endl
                      << "join types:" << ptr->_joinTypes << std::endl
                      << "where clause:" << ptr->_whereClause << std::endl
                      << "group by:" << ptr->_groupBy << std::endl
                      << "order by:" << ptr->_orderBy << std::endl;
        }
    }

protected:
    std::vector<std::string> _tableNames;   ///< all the selected tables.
    std::vector<Token::Type> _joinTypes;    ///< join policies. Should be INNER, LEFT or RIGHT.
    std::vector<Expr> _onClauses;           ///< on clauses of join.
    std::vector<Expr> _expressions;         ///< selected expressions.
    std::string _filename;                  ///< dump file name.
    Expr _whereClause;                      ///< where clause
    std::vector<std::string> _groupBy;      ///< the column name to be grouped by
    Expr _orderBy;                          ///< the expression to sort by
    std::shared_ptr<StatementSelect> _next; ///< the next select statement to union.
    bool _isUnionAll = false;               ///< whether is union all or union.
};

/**
 * @brief Statement update.
 */
class StatementUpdate : public Statement
{
public:
    StatementUpdate(const std::string &tableName, const std::vector<std::string> &keys,
                    const std::vector<Variant> &values,
                    const Expr &whereClause)
            : Statement(tableName, UPDATE), _keys(keys), _values(values), _whereClause(whereClause)
    {}

    const std::vector<std::string> &keys() const
    { return _keys; }

    const std::vector<Variant> &values() const
    { return _values; }

    const Expr &whereClause() const
    { return _whereClause; }

    virtual void print() const override
    {
        Statement::print();
        std::cout << "set keys:" << _keys << " values:" << _values << "\nwhere clause" << _whereClause << std::endl;
    }

protected:
    std::vector<std::string> _keys; ///< The keys to update.
    std::vector<Variant> _values;   ///< The destination values.
    Expr _whereClause;              ///< The where clause.
};

/**
 * @brief Statement drop.
 */
class StatementDropTable : public Statement
{
public:
    StatementDropTable(const std::vector<std::string> &tableNames) : Statement("", DROP_TABLE), _tableNames(tableNames)
    {}

    const std::vector<std::string> &tableNames() const
    { return _tableNames; }

protected:
    std::vector<std::string> _tableNames;
};

/**
 * @brief Statement drop database.
 */
class StatementDropDatabase : public Statement
{
public:
    StatementDropDatabase(const std::string &dbName) : Statement(dbName, DROP_DATABASE)
    {}
};

/**
 * @brief Statement show databases.
 */
class StatementShowDatabases : public Statement
{
public:
    StatementShowDatabases() : Statement(std::string(), SHOW_DATABASES)
    {}
};

/**
 * @brief Statement show tables.
 */
class StatementShowTables : public Statement
{
public:
    StatementShowTables() : Statement(std::string(), SHOW_TABLES)
    {}
};

/**
 * @brief Statement show columns from table.
 */
class StatementShowColumns : public Statement
{
public:
    StatementShowColumns(const std::string &tableName) : Statement(tableName, SHOW_COLUMNS)
    {}
};

/**
 * @brief Statement use database.
 */
class StatementUseDatabase : public Statement
{
public:
    StatementUseDatabase(const std::string &dbName) : Statement(dbName, USE_DATABASE)
    {}
};
