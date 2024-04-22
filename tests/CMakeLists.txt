link_libraries(sygraph)

add_executable(bitmap_frontier frontier/bitmap_frontier.cpp)
add_executable(csr formats/csr.cpp)
add_executable(coo2csr formats/coo2csr.cpp)
add_executable(graph_build graph/graph.cpp)
add_executable(advance operators/advance.cpp)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -fsycl")

enable_testing()

add_test(
  NAME test_bitmap_frontier
  COMMAND bitmap_frontier
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_csr_format
  COMMAND csr
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME coo2csr
  COMMAND coo2csr
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME graph_build
  COMMAND graph_build
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME advance_operator
  COMMAND advance
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)