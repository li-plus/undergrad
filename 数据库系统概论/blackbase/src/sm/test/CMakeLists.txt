project(sm_test_cpp)

add_executable(test_sm test_sm.cpp)
target_link_libraries(test_sm sm ql ix rm pf)
add_test(test_sm test_sm)
