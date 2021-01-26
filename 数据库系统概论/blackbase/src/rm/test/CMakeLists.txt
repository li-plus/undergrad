project(rm_test_cpp)

add_executable(test_bitmap test_bitmap.cpp)
add_test(test_bitmap test_bitmap)

add_executable(test_rm test_rm.cpp)
target_link_libraries(test_rm rm pf)
add_test(test_rm test_rm)
