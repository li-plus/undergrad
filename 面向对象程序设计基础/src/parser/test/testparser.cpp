#include <iostream>
#include <fstream>
#include <string>
#include "../src/parser.h"
#include "../src/sqlexcept.h"

int main()
{
    std::string cmd;
    while (getline(std::cin, cmd, ';'))
    {
        cmd += ';';
        Parser parser(cmd);
        try
        {
            auto statement = parser.parseStatement();
            statement->print();
        }
        catch (TokenError &e)
        {
            std::cout << e.what() << std::endl;
        }
        catch (LexerError &e)
        {
            std::cout << e.what() << std::endl;
        }
        catch (ParserError &e)
        {
            std::cout << e.what() << std::endl;
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}