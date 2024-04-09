cmake_minimum_required(VERSION 3.10)

project(SYgraph VERSION 0.1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directory
target_include_directories(SYgraph PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
