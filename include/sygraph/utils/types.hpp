#pragma once

#include <cstddef>

namespace sygraph {
inline namespace v0 {
namespace types {

typedef unsigned int index_t;
typedef size_t offset_t;

namespace detail {

constexpr uint byte_size = 8;
constexpr size_t MAX_LOCAL_MEM_SIZE = 512;
constexpr size_t MAX_ACTIVE_ELEMS_SIZE = 1048576; // TODO: [!!] This is a temporary value

} // namespace detail

} // namespace types
} // namespace v0
} // namespace sygraph