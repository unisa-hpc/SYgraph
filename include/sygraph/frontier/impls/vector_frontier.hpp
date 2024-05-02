#pragma once

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/vector.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {


template <typename type_t>
class frontier_vector_t;

template <typename type_t>
class device_vector_frontier_t {
  friend class frontier_vector_t<type_t>;
public:
  device_vector_frontier_t(size_t max_size) : max_size(max_size) {}

  SYCL_EXTERNAL inline bool empty() const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    return ref.load() == 0;
  }

  SYCL_EXTERNAL inline size_t size() const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    return ref.load();
  }

  SYCL_EXTERNAL inline bool insert(type_t val) const {
    sycl::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device> ref(tail[0]);
    data[ref++] = val;
    return true;
  }

  SYCL_EXTERNAL inline bool insert(type_t val, size_t idx) const {
    if (idx >= *tail) {
      return false;
    }
    data[idx] = val;
    return true;
  }

  SYCL_EXTERNAL inline bool remove(type_t val) const {
    return false;
  }

  SYCL_EXTERNAL inline void clear() const {
    sycl::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device> ref(tail[0]);
    ref = 0;
  }

  SYCL_EXTERNAL inline size_t prealloc(size_t num_elems) const { // TODO: check for max size
    sycl::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device> ref(tail[0]);
    return ref.fetch_add(num_elems);
  }

protected:
  inline void set_tail(size_t new_tail) {
    *tail = new_tail;
  }
  inline size_t get_tail() const {
    return *tail;
  }

private:
  type_t* data;
  size_t* tail;
  size_t max_size;
};

template <typename type_t>
class frontier_vector_t {
public:
  using frontier_type = type_t;

  static void swap(frontier_vector_t<type_t>& a, frontier_vector_t<type_t>& b) {
    std::swap(a.vector.data, b.vector.data);
    std::swap(a.vector.tail, b.vector.tail);
    // std::swap(a.vector, b.vector);
  }

  frontier_vector_t(sycl::queue& q, size_t num_elems) : q(q), vector(num_elems) {
    vector.data = memory::detail::memory_alloc<type_t, memory::space::shared>(num_elems, q);
    vector.tail = memory::detail::memory_alloc<size_t, memory::space::shared>(1, q);
    vector.set_tail(0);
  }

  ~frontier_vector_t() {
    sycl::free(vector.data, q);
    sycl::free(vector.tail, q);
  }

  size_t get_num_active_elements() const {
    return vector.get_tail();
  }

  inline bool self_allocated() const { return true; }

  void get_active_elements(type_t*& elems) const {
    elems = vector.data;
  }

  inline bool empty() const {
    return vector.empty();
  }

  bool insert(type_t val) {
    vector.data[(*(vector.tail))++] = val;
    return true;
  }

  bool insert(type_t val, size_t idx) {
    if (idx >= vector.get_tail()) {
      return false;
    }
    vector.data[idx] = val;
    return true;
  }

  bool remove(type_t idx) {
    return false;
  }

  frontier_vector_t& operator=(const frontier_vector_t& other) {
    throw std::runtime_error("Not implemented");
  }

  inline void merge(frontier_bitmap_t<type_t>& other) {
    throw std::runtime_error("Not implemented");
  }

  inline void clear() {
    vector.set_tail(0);
  }

  const device_vector_frontier_t<type_t>& get_device_frontier() const {
    return vector;
  }

private:
  sycl::queue& q;          ///< The SYCL queue used for memory allocation.
  device_vector_frontier_t<type_t> vector;
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph