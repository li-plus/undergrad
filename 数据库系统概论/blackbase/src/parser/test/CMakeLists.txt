project(sql_test_cpp)

add_executable(test_parser test_parser.cpp)
target_link_libraries(test_parser parser)
