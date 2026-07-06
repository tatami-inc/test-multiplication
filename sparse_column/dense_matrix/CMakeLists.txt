cmake_minimum_required(VERSION 3.14)

project(
    scdm 
    VERSION 1.0.0
    DESCRIPTION "Sparse column-major LHS, dense matrix RHS"
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

add_executable(test test.cpp)
target_link_libraries(test CLI11::CLI11 tatami::eztimer)

add_executable(test_float test.cpp)
target_link_libraries(test_float CLI11::CLI11 tatami::eztimer)
target_compile_definitions(test_float PUBLIC USE_SINGLE)
