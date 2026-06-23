add_executable(test test.cpp)
target_link_libraries(test CLI11::CLI11 tatami::eztimer)

add_executable(test_float test.cpp)
target_link_libraries(test_float CLI11::CLI11 tatami::eztimer)
target_compile_definitions(test_float PUBLIC USE_SINGLE)
