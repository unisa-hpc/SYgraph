#pragma once

#include <cstddef>

namespace sygraph {
inline namespace v0 {
namespace types {

typedef unsigned int index_t;
typedef size_t offset_t;
#if BITMAP_SIZE == 32
typedef uint32_t bitmap_type_t;
#elif BITMAP_SIZE == 64
typedef uint64_t bitmap_type_t;
#else
#error "Invalid bitmap size"
#endif

namespace detail {

constexpr size_t MAX_SUBGROUPS = 64; // It can be made dynamic
constexpr uint byte_size = 8;
constexpr size_t MAX_LOCAL_MEM_SIZE = 512;
constexpr size_t MAX_ACTIVE_ELEMS_SIZE = 1048576; // TODO: [!!] This is a temporary value

} // namespace detail

} // namespace types
} // namespace v0
} // namespace sygraph