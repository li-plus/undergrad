#include "database.h"
#include "../../parser/src/statements.h"
#include "../../parser/src/lexer.h"

void Database::createTable(const std::string &tableName, const std::vector<Field> &fields,
                           const std::string &primaryKey)
{
    assertTableNotExist(tableName);
    _tables[tableName] = Table(tableName, fields, primaryKey);
}

void Database::showColumnsFrom(const std::string &tableName)
{
    assertTableExist(tableName);
    _tables[tableName].showColumns();
}

void Database::insertInto(const std::string &tableName, const std::map<std::string, Variant> &entry)
{
    assertTableExist(tableName);
    _tables[tableName].insertInto(entry);
}

void Database::dropTable(const std::string &tableName)
{
    assertTableExist(tableName);
    _tables.erase(tableName);
}

std::vector<std::string> Database::tableNames() const
{
    std::vector<std::string> tn;
    for (auto &t: _tables)
        tn.emplace_back(t.first);
    return tn;
}

void Database::showTables() const
{
    std::cout << "Tables_in_" << _dbName << std::endl;
    auto tNames = tableNames();
    for (auto &t : tableNames())
        std::cout << t << std::endl;
}

void Database::assertTableExist(const std::string &tableName) const
{
    if (!isTable(tableName))
        throw DatabaseError("Unknown table " + tableName);
}

void Database::assertTableNotExist(const std::string &tableName) const
{
    if (isTable(tableName))
        throw DatabaseError("Table " + tableName + " already exists");
}

void Database::deleteFrom(const std::string &tableName, const Expr &whereClause)
{
    assertTableExist(tableName);
    _tables[tableName].deleteWhere(whereClause);
}

QueryResultSelect
Database::selectFrom(const std::vector<std::string> &tableNames, const std::vector<Token::Type> &joinTypes,
                     const std::vector<Expr> &onClauses, const std::vector<Expr> &expressions, const Expr &whereClause,
                     const std::string &fileName, const std::vector<std::string> &groupBy, const Expr &orderBy)
{
    // single table
    auto firstTableName = tableNames.front();
    assertTableExist(firstTableName);
    if (tableNames.size() == 1)
        return _tables[firstTableName].selectFrom(expressions, whereClause, fileName, groupBy, orderBy);

    // multi table
    auto mergeTable = _tables[firstTableName];
    for (auto &f: mergeTable.fields())
        f.setKey(firstTableName + "." + f.key());
    for (int i = 1; i < tableNames.size(); i++)
    {
        auto tableName = tableNames[i];
        assertTableExist(tableName);
        auto nextTable = _tables[tableName];
        for (auto &f: nextTable.fields())
            f.setKey(tableName + "." + f.key());
        mergeTable = crossJoin(mergeTable, nextTable, joinTypes[i], onClauses[i]);
    }
    return mergeTable.selectFrom(expressions, whereClause, fileName, groupBy, orderBy);
}

void Database::update(const std::string &tableName, const std::vector<std::string> &keys,
                      const std::vector<Variant> &values, const Expr &whereClause)
{
    assertTableExist(tableName);
    _tables[tableName].update(keys, values, whereClause);
}

void
Database::loadData(const std::string &fileName, const std::string &tableName, const std::vector<std::string> &columns)
{
    if (!isFile(fileName))
        throw DatabaseError("No such file or dictionary: " + fileName);
    std::ifstream in(fileName);
    std::string line;
    while (getline(in, line))
    {
        std::map<std::string, Variant> entry;
        std::vector<Variant> values;
        auto strList = split(trim(line), '\t');
        for (auto &str: strList)
        {
            Lexer lexer(str);
            Token token = lexer.next();
            switch (token.type())
            {
                case Token::ID:
                    values.emplace_back(token.toId());
                    break;
                case Token::OPERAND:
                    values.emplace_back(token.toOperand());
                    break;
                default:
                    throw DatabaseError("Invalid value from file");
            }
        }
        if (values.size() != columns.size())
            throw DatabaseError("Value list size and column list size did not match");
        for (int i = 0; i < values.size(); i++)
            entry[columns[i]] = values[i];
        insertInto(tableName, entry);
    }
}

QueryResultSelect Database::selectFromUnion( const std::shared_ptr<StatementSelect> &head)
{
    auto sel = head;
    // tail
    auto tail = head;
    while (tail->next())
        tail = tail->next();
    // global order attribute
    auto orderAttr = tail->orderBy();
    // result of first select clause
    auto result = selectFrom(sel->tableNames(), sel->joinTypes(), sel->onClauses(), sel->expressions(),
                             sel->whereClause(), sel->fileName(), sel->groupBy(), orderAttr);
    for (sel = sel->next(); sel; sel = sel->next())
    {
        auto subResult = selectFrom(sel->tableNames(), sel->joinTypes(), sel->onClauses(), sel->expressions(),
                                    sel->whereClause(), sel->fileName(), sel->groupBy(), orderAttr);
        if (subResult.expressions().size() != result.expressions().size()) // size incompatible
            throw DatabaseError("Union result size incompatible");
        // union all by default
        result.entries().insert(result.entries().end(), subResult.entries().begin(), subResult.entries().end());
        if (!sel->isUnionAll())  // union
        {
            std::set<Entry> s(result.entries().begin(), result.entries().end());
            result.entries().assign(s.begin(), s.end());
        }
    }
    if (result.entries().empty())
        return result;

    // sort
    std::sort(result.entries().begin(), result.entries().end(),
              [=](const Entry &a, const Entry &b) { return a.back() < b.back(); });
    result.expressions().pop_back();
    for (auto &entry: result.entries())
        entry.pop_back();
    return result;
}

Table Database::crossJoin(const Table &left, const Table &right, Token::Type joinType, const Expr &onClause)
{
    Table joined("merge", left.fields(), "");
    joined.fields().insert(joined.fields().end(), right.fields().begin(), right.fields().end());
    for (auto &ia: left.entries())
    {
        for (auto &ib: right.entries())
        {
            Entry appended = ia;
            appended.insert(appended.end(), ib.begin(), ib.end());
            joined.entries().emplace_back(appended);
        }
    }
    for (auto it = joined.entries().begin(); it != joined.entries().end(); it++)
    {
        auto entry = *it;
        if (!onClause.eval(joined.entryToMap(entry)).toBool())
            joined.entries().erase(it--);
    }
    if (joinType == Token::INNER)
        return joined;
    if (joinType == Token::LEFT)    // left join
    {
        auto orgJoinedSize = joined.entries().size();
        for (auto &el: left.entries())
        {
            joined.entries().emplace_back(el);
            joined.entries().back().insert(joined.entries().back().end(), right.fields().size(), Variant());
            for (size_t j = 0; j < orgJoinedSize; j++)
            {
                if (std::mismatch(el.begin(), el.end(), joined.entries()[j].begin()).first == el.end()) // found
                {
                    joined.entries().pop_back();
                    break;
                }
            }
        }
    }
    else // right join
    {
        auto orgJoinedSize = joined.entries().size();
        for (auto &er: right.entries())
        {
            joined.entries().emplace_back(er);
            joined.entries().back().insert(joined.entries().back().begin(), left.fields().size(), Variant());
            for (size_t j = 0; j < orgJoinedSize; j++)
            {
                if (std::mismatch(er.rbegin(), er.rend(), joined.entries()[j].rbegin()).first == er.rend()) // found
                {
                    joined.entries().pop_back();
                    break;
                }
            }
        }
    }
    return joined;
}
