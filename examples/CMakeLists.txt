link_libraries(sygraph)

add_executable(csr_to_binary csr_to_binary.cpp)
add_executable(bfs bfs/bfs.cpp)
add_executable(sssp sssp/sssp.cpp)
add_executable(tc tc/tc.cpp)
add_executable(bc bc/bc.cpp)
add_executable(cc cc/cc.cpp)

set_property(DIRECTORY ${CMAKE_SOURCE_DIR}/ PROPERTY CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

if (DEFINED ARCH)
  set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -fsycl -fsycl-targets=${ARCH}")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
endif()
