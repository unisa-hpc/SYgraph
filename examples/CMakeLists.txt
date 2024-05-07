link_libraries(sygraph)

add_executable(csr_to_binary csr_to_binary.cpp)
add_executable(bfs bfs/bfs.cpp)
add_executable(sssp sssp/sssp.cpp)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

if (DEFINED ARCH)
  set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -fsycl -fsycl-targets=${ARCH}")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
endif()
