link_libraries(sygraph)

add_executable(bitmap_frontier frontier/bitmap_frontier.cpp)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -fsycl")

enable_testing()

add_test(
  NAME test_bitmap_frontier
  COMMAND bitmap_frontier
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)