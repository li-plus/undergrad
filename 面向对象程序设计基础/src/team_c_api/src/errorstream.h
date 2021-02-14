#pragma once

#include <exception>
#include <iostream>
#include <string>

enum DATA_BASE_ERROR_EVENT
{
    ERROR_NONE,
    ERROR_UNKNOWN,
    ERROR_COMMAND_FORM,
    ERROR_TYPE_NOT_SUPPORT,
    ERROR_BASE_EXIST,
    ERROR_BASE_NOT_EXIST,
    ERROR_TABLE_EXIST,
    ERROR_TABLE_NOT_EXIST,
    ERROR_ATTRIBUTE_NOT_EXIST,
    ERROR_INSERT_DATA_ATTRIBUTE_TYPE_MISMATCH,
    ERROR_PRIMARY_KEY_REPEATED,
    ERROR_NOT_NULL_KEY_NULL,
};

/**
 * @brief The legacy database error exception class
 */
class DataBaseErrorEvent : public std::exception
{
    int type_;
public:
    DataBaseErrorEvent() : type_(ERROR_NONE)
    {}

    DataBaseErrorEvent(int type) : type_(type)
    {}

    int getType() const
    { return type_; }

    virtual const char *getErrorInfo() const
    {
        static std::string res = "";
        res = "";
        switch (type_)
        {
            case ERROR_UNKNOWN:
                res = "Error: Some unknown errors existed.";
                break;
            case ERROR_COMMAND_FORM:
                res = "Error: The form of your command is wrong. Please check your input.";
                break;
            case ERROR_TYPE_NOT_SUPPORT:
                res = "Error: This database does not support some value types.";
                break;
            case ERROR_BASE_EXIST:
                res = "Error: The name of this database has already been used.";
                break;
            case ERROR_BASE_NOT_EXIST:
                res = "Error: This database has never existed.";
                break;
            case ERROR_TABLE_EXIST:
                res = "Error: The name of this table has already been used.";
                break;
            case ERROR_TABLE_NOT_EXIST:
                res = "Error: This table has never existed.";
                break;
            case ERROR_ATTRIBUTE_NOT_EXIST:
                res = "Error: Some attributes have never existed.";
                break;
            case ERROR_INSERT_DATA_ATTRIBUTE_TYPE_MISMATCH:
                res = "Error: The type of your input value mismatches its attribute.";
                break;
            case ERROR_PRIMARY_KEY_REPEATED:
                res = "Error: The primary key repeated in one table.";
                break;
            case ERROR_NOT_NULL_KEY_NULL:
                res = "Error: A not-null-key becomes null.";
                break;
            default:
            {
                break;
            }
        }
        return res.c_str();
    }

    virtual const char *what() const throw()
    {
        return getErrorInfo();
    }
};

const DataBaseErrorEvent kERROR_NONE = ERROR_NONE;
const DataBaseErrorEvent kERROR_UNKNOWN = ERROR_UNKNOWN;
const DataBaseErrorEvent kERROR_COMMAND_FORM = ERROR_COMMAND_FORM;
const DataBaseErrorEvent kERROR_TYPE_NOT_SUPPORT = ERROR_TYPE_NOT_SUPPORT;
const DataBaseErrorEvent kERROR_BASE_EXIST = ERROR_BASE_EXIST;
const DataBaseErrorEvent kERROR_BASE_NOT_EXIST = ERROR_BASE_NOT_EXIST;
const DataBaseErrorEvent kERROR_TABLE_EXIST = ERROR_TABLE_EXIST;
const DataBaseErrorEvent kERROR_TABLE_NOT_EXIST = ERROR_TABLE_NOT_EXIST;
const DataBaseErrorEvent kERROR_ATTRIBUTE_NOT_EXIST = ERROR_ATTRIBUTE_NOT_EXIST;
const DataBaseErrorEvent kERROR_INSERT_DATA_ATTRIBUTE_TYPE_MISMATCH = ERROR_INSERT_DATA_ATTRIBUTE_TYPE_MISMATCH;
const DataBaseErrorEvent kERROR_PRIMARY_KEY_REPEATED = ERROR_PRIMARY_KEY_REPEATED;
const DataBaseErrorEvent kERROR_NOT_NULL_KEY_NULL = ERROR_NOT_NULL_KEY_NULL;
