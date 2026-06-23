add_executable(drsv_test test.cpp)
target_link_libraries(drsv_test CLI11::CLI11 tatami::eztimer)

add_executable(drsv_test_float test.cpp)
target_link_libraries(drsv_test_float CLI11::CLI11 tatami::eztimer)
target_compile_definitions(drsv_test_float PUBLIC USE_SINGLE)
