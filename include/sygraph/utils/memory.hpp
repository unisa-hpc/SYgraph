#pragma once

#include <sycl/sycl.hpp>

namespace sygraph {
inline namespace v0 {
namespace memory {

enum class space {
  host,
  device,
  shared
};

namespace detail {

template <typename T, space V>
inline T* memory_alloc(size_t n, sycl::queue& q) {
  if constexpr (V == space::host) {
    return sycl::malloc_host<T>(n, q);
  } else if constexpr (V == space::device) {
    return sycl::malloc_device<T>(n, q);
  } else if constexpr (V == space::shared) {
    return sycl::malloc_shared<T>(n, q);
  }

  throw std::runtime_error("Unknown memory space");
}

} // namespace detail
} // namespace memory
} // namespace v0
} // namespace sygraph