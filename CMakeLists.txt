cmake_minimum_required(VERSION 3.16)
project(ofdm_link_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(viterbi src/viterbi.cpp)
target_include_directories(viterbi PUBLIC include)
if(MSVC)
  target_compile_options(viterbi PUBLIC /W4)
else()
  target_compile_options(viterbi PUBLIC -Wall -Wextra -Wpedantic)
endif()

# Self-contained executables, so they run without the MinGW runtime on PATH.
if(MINGW)
  add_link_options(-static)
endif()

add_executable(test_viterbi tests/test_viterbi.cpp)
target_link_libraries(test_viterbi PRIVATE viterbi)

add_executable(viterbi_bench tools/viterbi_bench.cpp)
target_link_libraries(viterbi_bench PRIVATE viterbi)

enable_testing()
add_test(NAME viterbi COMMAND test_viterbi ${CMAKE_SOURCE_DIR}/tests/vectors)
