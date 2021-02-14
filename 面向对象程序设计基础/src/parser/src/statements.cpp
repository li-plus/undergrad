#include "statements.h"

std::map<Statement::StatementType, std::string> Statement::typeStringMap{
    {Statement::NONE, "none"},

    {Statement::CREATE_TABLE, "create table"},
    {Statement::CREATE_DATABASE, "create database"},

    {Statement::USE_DATABASE, "use database"},

    {Statement::SHOW_TABLES, "show tables"},
    {Statement::SHOW_DATABASES, "show databases"},
    {Statement::SHOW_COLUMNS, "show columns"},

    {Statement::DROP_TABLE, "drop table"},
    {Statement::DROP_DATABASE, "drop database"},

    {Statement::SELECT, "select"},
    {Statement::INSERT, "insert"},
    {Statement::UPDATE, "update"},
    {Statement::DELETE, "delete"},
    {Statement::LOAD, "load"}};
