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

add_executable(single4 single.cpp)
target_link_libraries(single4 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(single4 PUBLIC ACCUMULATORS=4)

add_executable(single8 single.cpp)
target_link_libraries(single8 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(single8 PUBLIC ACCUMULATORS=8)

add_executable(single16 single.cpp)
target_link_libraries(single16 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(single16 PUBLIC ACCUMULATORS=16)

add_executable(blocked4 blocked.cpp)
target_link_libraries(blocked4 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(blocked4 PUBLIC ACCUMULATORS=4)

add_executable(blocked8 blocked.cpp)
target_link_libraries(blocked8 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(blocked8 PUBLIC ACCUMULATORS=8)

add_executable(blocked16 blocked.cpp)
target_link_libraries(blocked16 CLI11::CLI11 tatami::eztimer)
target_compile_definitions(blocked16 PUBLIC ACCUMULATORS=16)
