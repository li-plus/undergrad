set(SOURCES ql_manager.cpp ql_interp.cpp ql_error.c)
add_library(ql STATIC ${SOURCES})

add_subdirectory(test)
