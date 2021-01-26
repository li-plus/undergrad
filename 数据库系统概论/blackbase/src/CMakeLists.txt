include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(pf)
add_subdirectory(rm)
add_subdirectory(ix)
add_subdirectory(sm)
add_subdirectory(ql)
add_subdirectory(parser)

add_executable(main main.cpp)
target_link_libraries(main parser ql sm ix rm pf)
