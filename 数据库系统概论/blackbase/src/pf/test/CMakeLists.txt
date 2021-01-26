project(pf_test_cpp)

add_executable(test_list test_list.cpp)
add_test(test_list test_list)

add_executable(test_pf_hash_table test_pf_hash_table.cpp)
target_link_libraries(test_pf_hash_table pf)
add_test(test_pf_hash_table test_pf_hash_table)

add_executable(test_pf test_pf.cpp)
target_link_libraries(test_pf pf)
add_test(test_pf test_pf)
