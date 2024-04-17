cmake_minimum_required(VERSION 3.10)

project(SYgraph VERSION 0.1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(sygraph INTERFACE)

target_include_directories(sygraph INTERFACE include)

option(SYGRAPH_BUILD_TESTS "Build tests" ON)
if (SYGRAPH_BUILD_TESTS)
  add_subdirectory(tests)
endif()