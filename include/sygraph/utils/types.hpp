#pragma once 

#include <cstddef>

namespace sygraph {
inline namespace v0 {
namespace types {

typedef unsigned int index_t;
typedef size_t offset_t;

namespace detail {

constexpr uint byte_size = 8;

}

} // namespace types
} // namespace v0
} // namespace sygraph