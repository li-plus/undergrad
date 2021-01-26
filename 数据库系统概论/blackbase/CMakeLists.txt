cmake_minimum_required(VERSION 3.16)
project(redbase_c)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message("CMAKE_BUILD_TYPE not specified. Default to Release mode.")
    set(CMAKE_BUILD_TYPE Release)
endif ()

message("Using ${CMAKE_BUILD_TYPE} mode.")

set(CMAKE_C_FLAGS "-Wall")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()
add_subdirectory(src)
