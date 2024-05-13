cmake_minimum_required(VERSION 3.10)

project(SYgraph VERSION 0.1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG -fno-omit-frame-pointer" CACHE STRING "Flags used by the compiler during debug builds." FORCE)

add_library(sygraph INTERFACE)

target_include_directories(sygraph INTERFACE include)

option(SYGRAPH_BUILD_TESTS "Build tests" OFF)
option(SYGRAPH_BUILD_EXAMPLES "Build Examples" OFF)
option(SYGRAPH_SAMPLE_DATA "Generate samples data" OFF)
option(SYGRAPH_ENABLE_PROFILING "Enable profiling" OFF)

if (SYGRAPH_BUILD_TESTS)
  add_subdirectory(tests)
  set(SYGRAPH_SAMPLE_DATA ON)
endif()

if (SYGRAPH_SAMPLE_DATA)
  target_compile_definitions(sygraph INTERFACE "GENERATE_SAMPLE_DATA")
endif()

if (SYGRAPH_ENABLE_PROFILING)
  target_compile_definitions(sygraph INTERFACE "ENABLE_PROFILING")
endif()

if (SYGRAPH_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()