#pragma once

#include <map>
#include "database.h"
#include "databaseexcept.h"
#include "datastream.h"
#include "../../parser/src/parser.h"

/**
 * @brief The interface of the database.
 * @details
 * With this API, the user could execute an sql statement and get the result.
 *
 * Example usage:
 * ```
 * DatabaseAPI api;
 * api.execute("CREATE DATABASE oop;");
 * ```
 */
class DatabaseAPI
{
public:
    /**
     * @brief Construct and load database from file.
     */
    DatabaseAPI()
    { load(); }

    /**
     * @brief Execute the sql command.
     * @param command The sql command with semicolon at end.
     */
    void execute(const std::string &command);

protected:
    /**
     * @brief Load the database from binary data file.
     */
    void load();

    /**
     * @brief Save the database into binary data file.
     */
    void save();

    /**
     * @brief Create a database of specific name.
     * @param dbName Database name.
     */
    void createDatabase(const std::string &dbName);

    /**
     * @brief Drop specific database.
     * @param dbName Database name.
     */
    void dropDatabase(const std::string &dbName);

    /**
     * @brief Show all databases.
     */
    void showDatabases() const;

    /**
     * @brief Use specific database.
     * @param dbName Database name.
     */
    void useDatabase(const std::string &dbName);

    /**
     * @brief Assert the specific database exists.
     * @param dbName Database name.
     */
    void assertDatabaseExist(const std::string &dbName) const;

    /**
     * @brief Assert the specific database does not exist.
     * @param dbName Database name.
     */
    void assertDatabaseNotExist(const std::string &dbName) const;

    /**
     * @brief Assert current working database is not none.
     */
    void assertDatabaseSelected() const;

    /**
     * @brief Determine whether the specific database exists
     * @param dbName Database name
     * @return whether the specific database exists
     */
    bool isDatabase(const std::string &dbName) const
    { return _databases.find(dbName) != _databases.end(); }

protected:
    std::map<std::string, Database> _databases; ///< all databases.
    std::string _curBaseName;   ///< current working database name.
};
