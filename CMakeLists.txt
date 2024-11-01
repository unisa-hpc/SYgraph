cmake_minimum_required(VERSION 3.10)

project(SYgraph VERSION 0.1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags for debug
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG " CACHE STRING "Flags used by the compiler during debug builds." FORCE)

add_library(sygraph INTERFACE)

include(${CMAKE_SOURCE_DIR}/cmake/DoxygenSetup.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/SetupOptions.cmake)

# target_include_directories(sygraph INTERFACE include)
target_include_directories(sygraph INTERFACE  
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  
  $<INSTALL_INTERFACE:include>
)

# Install library headers
install(DIRECTORY include/ DESTINATION include)

# Install the interface library (header-only)
install(TARGETS sygraph EXPORT SYgraphConfig)

# Export the configuration for find_package support
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SYgraphConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(EXPORT SYgraphConfig
        DESTINATION lib/cmake/SYgraph
        FILE SYgraphConfig.cmake)