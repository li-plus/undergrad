#pragma once

#include "errorstream.h"
#include "../../backend/src/databaseapi.h"

/**
 * @brief The database manager calls the DatabaseAPI to execute the SQL command.
 * @tparam T Just an adaptor
 */
template<typename T = int>
class DataBaseManager
{
public:
    /**
     * @brief It calls the DatabaseAPI to execute the query
     * @param command SQL command
     */
    void Query(const std::string &command)
    { _dbAPI.execute(command + ";"); }

protected:
    DatabaseAPI _dbAPI; ///< database api
};
