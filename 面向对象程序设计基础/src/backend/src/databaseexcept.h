#pragma once

#include <exception>

/**
 * @brief Base class for database execution exception.
 */
class BackendError : public std::exception
{
public:
    /**
     * @brief Construct with specific exception message.
     * @param msg Specific exception message.
     */
    BackendError(const std::string &msg) noexcept : _msg(msg)
    {}

    /**
     * @brief Get the exception message
     * @return The exception message
     */
    virtual const char *what() const noexcept override
    { return _msg.c_str(); }

protected:
    std::string _msg;   ///< Exception message.
};

/**
 * @brief This class provide exception message during database operation.
 */
class DatabaseError : public BackendError
{
public:
    /**
     * @brief Construct with exception message, which will be later prefixed by "Database Error:".
     * @param msg Specific exception message.
     */
    DatabaseError(const std::string &msg) noexcept : BackendError("Database Error: " + msg)
    {}
};

/**
 * @brief The exception class for variant operation error.
 */
class VariantError : public BackendError
{
public:
    /**
     * @brief Construct with exception message, which will be later prefixed by "Variant Error:".
     * @param msg Specific exception message.
     */
    VariantError(const std::string &msg) noexcept: BackendError("Variant Error: " + msg)
    {}
};
