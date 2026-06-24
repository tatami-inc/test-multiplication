cmake_minimum_required(VERSION 3.14)

project(
    matmult
    VERSION 1.0.0
    DESCRIPTION "Performance tests for matrix multiplication"
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

add_subdirectory(dense_row/single_vector)
add_subdirectory(dense_row/multiple_vectors)
