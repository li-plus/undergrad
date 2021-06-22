add_executable(
    unit_tests
    main.cpp
    test_spmm.cu
    )

set_target_properties(unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(unit_tests PRIVATE ${CUDA_LIBRARIES} gtest_main expspmm cusparse curand cuda )

add_test(
  NAME
    unit
  COMMAND
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
  )
