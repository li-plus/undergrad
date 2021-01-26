project(sm_cpp)

set(SOURCES sm_manager.cpp sm_error.c)
add_library(sm STATIC ${SOURCES})

add_subdirectory(test)
