#pragma once

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/types.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {


template<typename T>
class FrontierVector;
template<typename T>
class DeviceVectorFrontier;

template<typename T>
struct LocalVectorFrontier {
  LocalVectorFrontier(size_t max_elems, sycl::handler& cgh) : _max_elems(max_elems), _data(max_elems, cgh), _tail(1, cgh) {} // TODO: [!!] fix this

  SYCL_EXTERNAL void init(sycl::nd_item<1>& item) const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> ref(_tail[0]);
    if (item.get_local_linear_id() == 0) { ref = 0; }
    sycl::group_barrier(item.get_group());
  }

  SYCL_EXTERNAL bool insert(T val) const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> ref(_tail[0]);
    // size_t& ref = tail[0];
    if (ref >= _max_elems) { return 0; }
    size_t loc = ref.fetch_add(1);
    _data[loc] = val;
    return 1;
  }

  SYCL_EXTERNAL void copyToGlobal(sycl::nd_item<1>& item, const DeviceVectorFrontier<T>& out) const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> ref(_tail[0]);
    size_t lid = item.get_local_linear_id();
    sycl::group_barrier(item.get_group());
    size_t address_space = 0;
    if (lid == 0) { address_space = out.prealloc(ref.load()); }
    address_space = sycl::group_broadcast(item.get_group(), address_space, 0);
    for (size_t i = lid; i < ref; i += item.get_local_range(0)) { out.insert(_data[i], address_space + i); }
  }

  sycl::local_accessor<T, 1> _data;
  sycl::local_accessor<size_t, 1> _tail;
  const size_t _max_elems;
};

template<typename T>
class DeviceVectorFrontier {
  friend class FrontierVector<T>;

public:
  DeviceVectorFrontier(size_t max_size) : _max_size(max_size) {}

  SYCL_EXTERNAL bool empty() const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_tail[0]);
    return ref.load() == 0;
  }

  SYCL_EXTERNAL inline size_t size() const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_tail[0]);
    return ref.load();
  }

  SYCL_EXTERNAL inline bool insert(T val) const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_tail[0]);
    if (ref.load() >= _max_size) { return false; }
    _data[ref++] = val;
    return true;
  }

  SYCL_EXTERNAL inline bool insert(T val, size_t idx) const {
    if (idx >= *_tail) { return false; }
    _data[idx] = val;
    return true;
  }

  SYCL_EXTERNAL inline bool remove(T val) const { return false; }

  SYCL_EXTERNAL inline void clear() const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_tail[0]);
    ref = 0;
  }

  SYCL_EXTERNAL inline size_t prealloc(size_t num_elems) const { // TODO: [!] check for max size
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_tail[0]);
    return ref.fetch_add(num_elems);
  }

protected:
  inline void set_tail(size_t new_tail) { *_tail = new_tail; }
  inline size_t get_tail() const { return *_tail; }

private:
  T* _data;
  size_t* _tail;
  size_t _max_size;
};

template<typename T>
class FrontierVector {
public:
  using frontier_type = T;

  static void swap(FrontierVector<T>& a, FrontierVector<T>& b) {
    std::swap(a._vector._data, b._vector._data);
    std::swap(a._vector._tail, b._vector._tail);
  }

  FrontierVector(sycl::queue& q, size_t num_elems) : _queue(q), _vector(num_elems) {
    _vector._data = memory::detail::memoryAlloc<T, memory::space::shared>(num_elems, _queue);
    _vector._tail = memory::detail::memoryAlloc<size_t, memory::space::shared>(1, _queue);
    _vector.set_tail(0);
  }

  ~FrontierVector() {
    sycl::free(_vector._data, _queue);
    sycl::free(_vector._tail, _queue);
  }

  size_t getNumActiveElements() const { return _vector.get_tail(); }

  bool selfAllocated() const { return true; }

  void getActiveElements(T*& elems) const { elems = _vector._data; }

  /**
   * @brief Retrieves the active elements in the bitmap.
   *
   * @param elems The array to store the active elements. It must be pre-allocated with shared-access.
   * @param active If true, it retrieves the active elements, otherwise the inactive elements.
   */
  void getActiveElements(T*& elems, size_t& size) const {
    size = _vector.get_tail();
    elems = _vector._data;
  }

  bool empty() const { return _vector.empty(); }

  bool insert(T val) {
    _vector._data[(*(_vector._tail))++] = val;
    return true;
  }

  bool insert(T val, size_t idx) {
    if (idx >= _vector.get_tail()) { return false; }
    _vector._data[idx] = val;
    return true;
  }

  bool remove(T idx) { return false; }

  FrontierVector& operator=(const FrontierVector& other) { throw std::runtime_error("Not implemented"); }

  void merge(FrontierBitmap<T>& other) { throw std::runtime_error("Not implemented"); }

  void clear() { _vector.set_tail(0); }

  const DeviceVectorFrontier<T>& getDeviceFrontier() const { return _vector; }

  LocalVectorFrontier<T> getLocalFrontier(sycl::handler& cgh) const { return LocalVectorFrontier<T>(types::detail::MAX_LOCAL_MEM_SIZE, cgh); }

private:
  sycl::queue& _queue; ///< The SYCL queue used for memory allocation.
  DeviceVectorFrontier<T> _vector;
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph