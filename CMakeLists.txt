cmake_minimum_required(VERSION 3.10)

project(SYgraph VERSION 0.1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(sygraph INTERFACE)

target_include_directories(sygraph INTERFACE include)

link_libraries(sygraph)
set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -fsycl")
add_executable(test src/test.cpp)
