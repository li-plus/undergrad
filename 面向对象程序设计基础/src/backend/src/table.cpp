
#include "table.h"
#include "../../parser/src/statements.h"

std::map<std::string, Variant> Table::entryToMap(const Entry &entry)
{
    std::map<std::string, Variant> m;
    for (size_t i = 0; i < _fields.size(); i++)
        m[_fields[i].key()] = entry[i];
    return m;
}

std::map<std::string, std::vector<Variant>> Table::entryToMap(const std::vector<Entry> &entries)
{
    std::map<std::string, std::vector<Variant> > entriesMap;
    for (size_t i = 0; i < _fields.size(); i++)
    {
        std::vector<Variant> entryVec;
        for (auto &entry: entries)
            entryVec.emplace_back(entry[i]);
        entriesMap[_fields[i].key()] = entryVec;
    }
    return entriesMap;
}


void Table::insertInto(const std::map<std::string, Variant> &entryMap)
{
    Entry entry;
    for (auto &f : _fields)
    {
        if (!entryMap.count(f.key())) // key not found
            entry.emplace_back(Variant());
        else
            entry.emplace_back(entryMap.at(f.key()).convertTo(f.type()));
    }
    _entries.emplace_back(entry);
}

QueryResultSelect Table::selectFrom(std::vector<Expr> expressions, const Expr &whereClause, const std::string &fileName,
                                    const std::vector<std::string> &groupBy, const Expr &orderBy)
{
    // construct result
    std::vector<Entry> result;
    for (auto &entry: _entries)
    {
        if (whereClause.eval(entryToMap(entry)).toBool())
            result.emplace_back(entry);
    }
    if (result.empty())
        return QueryResultSelect();

    // if select all
    if (expressions.front().token().type() == Token::ID && expressions.front().token().toId() == "*")
    {
        expressions.clear();
        for (auto &f:_fields)
            expressions.emplace_back(Expr(Token(Token::ID, f.key())));
    }

    // order by
    if (orderBy.isNull())
        expressions.emplace_back(Expr(Token(Token::ID, primaryKey())));
    else
        expressions.emplace_back(orderBy);
    std::vector<Entry> finalResult;

    // record whether collapse
    bool isCollapse = false;
    // make groups
    if (!groupBy.empty())
        isCollapse = true;
    auto groupResult = makeGroups(result, groupBy);
    for (auto &group : groupResult)
    {
        std::vector<Entry> finalGroup;
        for (auto &entry: group)
        {
            Entry finalEntry;
            for (auto &col: expressions)
            {
                if (col.token().type() == Token::COUNT)
                {
                    finalEntry.emplace_back(col.eval(entryToMap(group)));
                    isCollapse = true;
                }
                else
                {
                    finalEntry.emplace_back(col.eval(entryToMap(entry)));
                }
            }
            finalGroup.emplace_back(finalEntry);
        }
        if (isCollapse)
            finalGroup.erase(finalGroup.begin() + 1, finalGroup.end());
        finalResult.insert(finalResult.end(), finalGroup.begin(), finalGroup.end());
    }

    // unsorted
    return QueryResultSelect(expressions, finalResult, fileName);
}

void Table::showColumns()
{
    std::cout << "Field\tType\tNull\tKey\tDefault\tExtra" << std::endl;
    for (auto &f : _fields)
        std::cout << f.key() << '\t' << f.type() << '\t' << (f.isNull() ? "YES" : "NO") << '\t'
                  << (f.isPrimary() ? "PRI" : "") << "\tNULL" << std::endl;
}

void Table::update(const std::vector<std::string> &keys, const std::vector<Variant> &values, const Expr &whereClause)
{
    for (auto &entry: _entries)
    {
        if (whereClause.eval(entryToMap(entry)).toBool())
        {
            for (size_t i = 0; i < keys.size(); i++)
                entry[keyToRank(keys[i])] = values[i].convertTo(_fields[keyToRank(keys[i])].type());
        }
    }
}

void Table::deleteWhere(const Expr &whereClause)
{
    for (auto it = _entries.begin(); it != _entries.end(); it++)
    {
        if (whereClause.eval(entryToMap(*it)).toBool())
            _entries.erase(it--);
    }
}

std::vector<std::vector<Entry>> Table::makeGroups(std::vector<Entry> &result, const std::vector<std::string> &groupBy)
{
    if (groupBy.empty())
        return std::vector<std::vector<Entry>>{result};
    std::vector<size_t> colRanks;
    for (auto &g: groupBy)
        colRanks.emplace_back(keyToRank(g));
    for (auto &r: colRanks)  // merge sort
    {
        std::vector<Entry> buffer(result.size());
        mergeSort(result.begin(), result.end(), buffer.begin(), [=](const Entry &a, const Entry &b) {
            return a[r] < b[r];
        });
    }
    std::vector<std::vector<Entry>> groups;
    std::vector<Entry> curGroup{result.front()};
    for (size_t i = 1; i < result.size(); i++)
    {
        if (slice(result[i], colRanks) != slice(result[i - 1], colRanks))
        {
            groups.emplace_back(curGroup);
            curGroup.clear();
        }
        curGroup.emplace_back(result[i]);
    }
    groups.emplace_back(curGroup);
    return groups;
}
