#pragma once
#ifdef WIN32

#include <io.h>
#include <direct.h>

#elif linux
#include <unistd.h>
#include <dirent.h>
#endif

/**
 * @brief List all files in the specific directory
 * @param dirName specific directory
 * @return All files in the specific directory
 */
static std::vector<std::string> listDir(const std::string &dirName)
{
    std::vector<std::string> files; // file names

#ifdef WIN32
    _finddata_t file;
    long long lf;
    if ((lf = _findfirst((dirName + "*").c_str(), &file)) == -1) // find fir
    {
        std::cerr << dirName << " not found!!!" << std::endl;
    }
    else
    {
        while (_findnext(lf, &file) == 0)
        {
            if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
                continue;
            files.emplace_back(file.name);
        }
    }
    _findclose(lf);
#elif unix
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir = opendir(dirName.c_str())) == nullptr)
    {
        perror("Open dir error...");
        exit(1);
    }
    while ((ptr = readdir(dir)) != nullptr)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) // current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8) // file
            files.emplace_back(ptr->d_name);
        else if (ptr->d_type == 10) // link file
            continue;
        else if (ptr->d_type == 4) // dir
            files.emplace_back(ptr->d_name);
    }
    closedir(dir);
#endif
    sort(files.begin(), files.end());
    return files;
}

/**
 * @brief Determine whether a specific file exists
 * @param fileName Input file name
 * @return Whether a specific file exists
 */
static inline bool isFile(const std::string &fileName)
{
    std::fstream in(fileName, std::ios::in);
    return in.is_open();
}

/**
 * @brief Determine whether a specific directory exists
 * @param dir Input directory name
 * @return Whether a specific directory exists
 */
static inline bool isDir(const std::string &dir)
{
#ifdef WIN32
    return access(dir.c_str(), 0) == 0;
#elif unix
    return opendir(dir.c_str()) != nullptr;
#endif
}

/**
 * @brief Create a new directory
 * @param dir new directory name
 * @return Whether operation succeeded.
 */
static inline bool createDir(const std::string &dir)
{
#ifdef WIN32
    return !isDir(dir.c_str()) && mkdir(dir.c_str()) == 0;
#elif unix
    return !isDir(dir.c_str()) && system(("mkdir " + dir).c_str()) == 0;
#endif
}

/**
 * @brief Create a new file
 * @param fileName new file name
 * @return Whether operation succeeded.
 */
static inline bool createFile(const std::string &fileName)
{
    return std::fstream(fileName, std::ios::out).is_open();
}

/**
 * @brief Remove an existing directory
 * @param dir directory name
 * @return Whether operation succeeded.
 */
static inline bool removeDir(const std::string &dir)
{
    return rmdir(dir.c_str()) == 0;
}

/**
 * @brief remove an existing file
 * @param fileName File name
 * @return Whether operation succeeded.
 */
static inline bool removeFile(const std::string &fileName)
{
    return remove(fileName.c_str()) == 0;
}

/**
  * @brief get current working directory
  * @return current working directory
  */
static inline std::string getCwd()
{
    char buf[256];
    return std::string(getcwd(buf, 256));
}

/**
 * @brief change current working directory
 * @param dir destination directory
 * @return Whether operation succeeded.
 */
static inline bool changeCwd(const std::string &dir)
{
    return chdir(dir.c_str()) == 0;
}