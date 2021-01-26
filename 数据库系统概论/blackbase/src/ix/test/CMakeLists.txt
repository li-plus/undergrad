project(ix_test_cpp)

add_executable(test_ix test_ix.cpp)
target_link_libraries(test_ix ix pf)
add_test(test_ix test_ix)
