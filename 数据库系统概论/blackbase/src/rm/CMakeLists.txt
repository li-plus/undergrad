project(rm_c C)

set(SOURCES rm_file.c rm_record.c rm_error.c)
add_library(rm STATIC ${SOURCES})

add_subdirectory(test)
