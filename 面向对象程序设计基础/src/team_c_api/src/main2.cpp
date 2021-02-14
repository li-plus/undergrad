#include <iostream>
#include <string>
#include "DataBaseManager.h"
#include "utils.h"

int main(int argc, char **argv)
{
    DataBaseManager<> master;
    std::string command;
    while (getline(std::cin, command, ';'))
    {
        if (std::cin.eof())
            break;
        try
        {
            master.Query(command);
        }
        catch (DataBaseErrorEvent &error)
        {
            error = ERROR_NONE;
        }
    }
    return 0;
}