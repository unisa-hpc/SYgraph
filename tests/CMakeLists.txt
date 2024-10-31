link_libraries(sygraph)

add_executable(bitmap_frontier frontier/bitmap_frontier.cpp)
add_executable(csr formats/csr.cpp)
add_executable(coo2csr_weighted formats/coo2csr.cpp)
add_executable(coo2csr_unweighted formats/coo2csr_unweighted.cpp)
add_executable(graph_build graph/graph.cpp)
add_executable(advance operators/advance.cpp)
add_executable(advance_graph operators/advance_graph.cpp)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -fsycl -fsycl-targets=${ARCH}")

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
  NAME coo2csr_weighted
  COMMAND coo2csr_weighted
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME coo2csr_unweighted
  COMMAND coo2csr_unweighted
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME graph_build
  COMMAND coo2csr_unweighted
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME advance_operator
  COMMAND advance
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME advance_graph_operator
  COMMAND advance_graph
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)