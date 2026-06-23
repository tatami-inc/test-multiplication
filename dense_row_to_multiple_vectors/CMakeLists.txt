add_executable(drmv_test test.cpp)
target_link_libraries(drmv_test CLI11::CLI11 tatami::eztimer)

add_executable(drmv_test_float test.cpp)
target_link_libraries(drmv_test_float CLI11::CLI11 tatami::eztimer)
target_compile_definitions(drmv_test_float PUBLIC USE_SINGLE)
