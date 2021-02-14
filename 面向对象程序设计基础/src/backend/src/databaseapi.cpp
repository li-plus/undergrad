#include "databaseapi.h"
#include "../../parser/src/sqlexcept.h"
#include <cmath>

void DatabaseAPI::assertDatabaseExist(const std::string &dbName) const
{
    if (!isDatabase(dbName))
        throw DatabaseError("Unknown database " + dbName);
}

void DatabaseAPI::assertDatabaseNotExist(const std::string &dbName) const
{
    if (isDatabase(dbName))
        throw DatabaseError("Database " + dbName + " already exists");
}

void DatabaseAPI::assertDatabaseSelected() const
{
    if (_curBaseName.empty())
        throw DatabaseError("Database not selected");
}

void DatabaseAPI::execute(const std::string &command)
{
    Parser parser(command);
    try
    {
        auto statement = parser.parseStatement();
        if (!statement)
            return;
        switch (statement->type())
        {
            case Statement::SHOW_COLUMNS:
                assertDatabaseSelected();
                _databases[_curBaseName].showColumnsFrom(statement->id());
                break;
            case Statement::SHOW_DATABASES:
                showDatabases();
                break;
            case Statement::SHOW_TABLES:
                assertDatabaseSelected();
                _databases[_curBaseName].showTables();
                break;
            case Statement::DROP_DATABASE:
                dropDatabase(statement->id());
                break;
            case Statement::DROP_TABLE:
            {
                assertDatabaseSelected();
                auto s = std::dynamic_pointer_cast<StatementDropTable>(statement);
                for (auto &tableName: s->tableNames())
                    _databases[_curBaseName].dropTable(tableName);
                break;
            }
            case Statement::CREATE_DATABASE:
                createDatabase(statement->id());
                break;
            case Statement::CREATE_TABLE:
            {
                assertDatabaseSelected();
                auto s = std::dynamic_pointer_cast<StatementCreateTable>(statement);
                _databases[_curBaseName].createTable(s->id(), s->fields(), s->primaryKey());
                break;
            }
            case Statement::USE_DATABASE:
                useDatabase(statement->id());
                break;
            case Statement::DELETE:
            {
                assertDatabaseSelected();
                auto s = std::dynamic_pointer_cast<StatementDelete>(statement);
                _databases[_curBaseName].deleteFrom(s->id(), s->whereClause());
                break;
            }
            case Statement::UPDATE:
            {
                assertDatabaseSelected();
                auto s = std::dynamic_pointer_cast<StatementUpdate>(statement);
                _databases[_curBaseName].update(s->id(), s->keys(), s->values(), s->whereClause());
                break;
            }
            case Statement::INSERT:
            {
                assertDatabaseSelected();
                auto s = std::dynamic_pointer_cast<StatementInsert>(statement);
                s->entry();
                _databases[_curBaseName].insertInto(s->id(), s->entry());
                break;
            }
            case Statement::SELECT:
            {
                assertDatabaseSelected();
                auto s = std::dynamic_pointer_cast<StatementSelect>(statement);
                _databases[_curBaseName].selectFromUnion(s).output();
                break;
            }
            case Statement::LOAD:
            {
                assertDatabaseSelected();
                auto s = std::dynamic_pointer_cast<StatementLoad>(statement);
                _databases[_curBaseName].loadData(s->fileName(), s->tableName(), s->columns());
            }
            default:
                break;
        }
        save();
    }
    catch (const SqlError &e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (const BackendError &e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}

void DatabaseAPI::createDatabase(const std::string &dbName)
{
    assertDatabaseNotExist(dbName);
    _databases[dbName] = Database(dbName);
}

void DatabaseAPI::dropDatabase(const std::string &dbName)
{
    assertDatabaseExist(dbName);
    _databases.erase(dbName);
}

void DatabaseAPI::showDatabases() const
{
    std::cout << "Database" << std::endl;
    for (auto &p: _databases)
        std::cout << p.first << std::endl;
}

void DatabaseAPI::useDatabase(const std::string &dbName)
{
    assertDatabaseExist(dbName);
    _curBaseName = dbName;
}

void DatabaseAPI::load()
{
    if (!isDir(SimSql::dataDir))
        return;
    DataStream in(SimSql::dataFilePath, std::ios::in);
    if (!in.is_open())
        return;
    in >> _databases;
    in.close();
}

void DatabaseAPI::save()
{
    if (!isDir(SimSql::dataDir))
        createDir(SimSql::dataDir);
    DataStream out(SimSql::dataFilePath, std::ios::out);
    if (!out.is_open())
        throw DatabaseError("Cannot open file " + SimSql::dataFilePath);
    out << _databases;
    out.close();
}
