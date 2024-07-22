#pragma once

#include <sycl/sycl.hpp>

namespace sygraph {
inline namespace v0 {
namespace sync {

template<typename T>
SYCL_EXTERNAL inline T atomicFetchAdd(T* ptr, T val) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ptr);
  return ref.fetch_add(val);
}

template<typename T>
SYCL_EXTERNAL inline T load(T* ptr) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ptr);
  return ref.load();
}

template<typename T>
SYCL_EXTERNAL inline T min(T* v1, T* v2) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref1(*v1);
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref2(*v2);
  return ref1.fetch_min(ref2.load());
}

template<typename T>
SYCL_EXTERNAL inline bool cas(T* ptr, T expected, T desired) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ptr);
  auto tmp = ref.load();

  return ref.compare_exchange_strong(expected, desired) ? desired : tmp;
}
} // namespace sync
} // namespace v0
} // namespace sygraph