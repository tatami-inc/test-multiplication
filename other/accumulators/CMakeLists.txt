cmake_minimum_required(VERSION 3.14)

project(
    drsv
    VERSION 1.0.0
    DESCRIPTION "Multiple accumulators"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11
    GIT_TAG v2.6.1
)

FetchContent_Declare(
    eztimer
    GIT_REPOSITORY https://github.com/tatami-inc/eztimer
    GIT_TAG master
)

FetchContent_MakeAvailable(cli11)
FetchContent_MakeAvailable(eztimer)

add_executable(test4 test.cpp implementations.cpp)
target_link_libraries(test4 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(test4 PUBLIC ACCUMULATORS=4)

add_executable(test8 test.cpp implementations.cpp)
target_link_libraries(test8 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(test8 PUBLIC ACCUMULATORS=8)

add_executable(test16 test.cpp implementations.cpp)
target_link_libraries(test16 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(test16 PUBLIC ACCUMULATORS=16)
