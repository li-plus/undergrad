project(ql_test_cpp)

add_executable(test_ql test_ql.cpp)
target_link_libraries(test_ql parser ql sm ix rm pf)
add_test(test_ql test_ql)
