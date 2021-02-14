#include <iostream>
#include <string>

#include "DataBaseManager.h"

bool readCommand(char *cmd, int maxLen, std::istream &inf)
{
	int n = 0;
	while (!inf.eof() && n < maxLen)
	{
		char c = inf.get();
		if (c == '\n')
			break;
		if (c == ';')
			continue;
		cmd[n ++] = c;
	}
	if (n < maxLen)
		cmd[n] = '\0';
	return !inf.eof();
}

int main(int argc, char **argv) {

	//freopen("input.sql", "r", stdin);

	const int maxCmdLen = (1 << 16) + 3;
	static char cmd[maxCmdLen];

	DataBaseManager<> *master = new DataBaseManager<>();

	while (readCommand(cmd, maxCmdLen, std::cin))
	{
		try {
			master->Query(cmd);
		}
		catch (DataBaseErrorEvent &error) {
			if (error.getType() != ERROR_NONE)
				//std::cerr << error.what() << std::endl;
				;
			error = ERROR_NONE;
		}
	}

	return 0;
}