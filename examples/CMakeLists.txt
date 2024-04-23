link_libraries(sygraph)

add_executable(bfs bfs/bfs.cpp)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -fsycl")
