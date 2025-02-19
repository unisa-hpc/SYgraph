# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

option(SYGRAPH_BUILD_TESTS "Build tests" OFF)
option(SYGRAPH_BUILD_EXAMPLES "Build Examples" OFF)
option(SYGRAPH_SAMPLE_DATA "Generate samples data" OFF)
option(SYGRAPH_ENABLE_PROFILING "Enable profiling" OFF)
option(SYGRAPH_ENABLE_PREFETCH "Enable runtime to prefetch shared memory allocation. Turn it OFF for compatibility" OFF)

set(SYGRAPH_BITMAP_SIZE 32 CACHE STRING "Bitmap size in bits. (32, 64)")
set(SYGRAPH_CU_SIZE 512 CACHE STRING "Number of threads per block in CUDA")

if (SYGRAPH_BUILD_TESTS)
  add_subdirectory(tests)
  set(SYGRAPH_SAMPLE_DATA ON)
endif()

target_compile_definitions(sygraph INTERFACE "BITMAP_SIZE=${SYGRAPH_BITMAP_SIZE}")
target_compile_definitions(sygraph INTERFACE "CU_SIZE=${SYGRAPH_CU_SIZE}")

if (SYGRAPH_SAMPLE_DATA)
  target_compile_definitions(sygraph INTERFACE "GENERATE_SAMPLE_DATA")
endif()

if (SYGRAPH_ENABLE_PROFILING)
  target_compile_definitions(sygraph INTERFACE "ENABLE_PROFILING")
endif()

if (SYGRAPH_ENABLE_PREFETCH)
  target_compile_definitions(sygraph INTERFACE "ENABLE_PREFETCH")
endif()

if (SYGRAPH_BUILD_EXAMPLES)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  add_subdirectory(${CMAKE_SOURCE_DIR}/examples)
endif()