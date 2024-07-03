#pragma once

#include <sygraph/frontier/impls/hierarchic_bitmap_frontier.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/types.hpp>
#include <sygraph/utils/vector.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {


template<typename T, size_t Levels>
class FrontierBitvec;

template<typename T, size_t Levels = 2, typename B = uint64_t>     // TODO [!!!] There are too many copies from host to device that degrade the
                                                                   // performance
class BitvecDevice : public HierarchicBitmapDevice<T, Levels, B> { // TODO modify the code in order to select the number of levels
public:
  using bitmap_type = B;

  BitvecDevice(size_t num_elems) : HierarchicBitmapDevice<T, Levels, B>(num_elems) {
    _vector_max_size = 16000; // TODO ! tune on vector size
  }

  SYCL_EXTERNAL bool useVector() const { return static_cast<int>(*_vector_tail < _vector_max_size); }

  SYCL_EXTERNAL T* getVector() const { return _vector; }

  /**
   * @brief Sets the bit at the specified index to 1.
   *
   * @param val The value of the bit to set.
   */
  SYCL_EXTERNAL inline bool insert(T val) const {
    // call super class insert
    HierarchicBitmapDevice<T, Levels, B>::insert(val);
    insertOnlyVector(val);

    return true;
  }

  template<sycl::memory_order MO, sycl::memory_scope MS>
  SYCL_EXTERNAL inline bool insert(T val, const sycl::local_accessor<T, 1>& pad, const sycl::atomic_ref<T, MO, MS>& pad_tail) const {
    HierarchicBitmapDevice<T, 2, B>::insert(val);
    if (pad_tail.load() < _vector_max_size) {
      pad[pad_tail++] = val;
    } else {
      return insertOnlyVector(val);
    }
    return true;
  }

  template<sycl::memory_order MO, sycl::memory_scope MS>
  SYCL_EXTERNAL inline bool
  finalize(sycl::nd_item<1> item, const sycl::local_accessor<T, 1>& pad, const sycl::atomic_ref<T, MO, MS>& pad_tail_ref) const {
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> tail_ref(_vector_tail[0]);

    size_t data_offset = _vector_max_size;
    if (group.leader()) { data_offset = tail_ref.fetch_add(pad_tail_ref.load()); }
    data_offset = sycl::group_broadcast(group, data_offset, 0);
    for (int i = lid; i < pad_tail_ref.load() && i < _vector_max_size; i += item.get_local_range(0)) { _vector[data_offset + i] = pad[i]; }

    return true;
  }

  SYCL_EXTERNAL inline bool insertOnlyVector(T val) const {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> tail_ref(_vector_tail[0]);
    if (tail_ref.load() < _vector_max_size) {
      _vector[tail_ref++] = val;
      return true;
    }
    return false;
  }

  SYCL_EXTERNAL inline void resetVector() const { *_vector_tail = 0; }

  SYCL_EXTERNAL size_t getVectorMaxSize() const { return _vector_max_size; }

  SYCL_EXTERNAL uint32_t getVectorSize() const { return *_vector_tail; }

  SYCL_EXTERNAL uint32_t* getVectorTail() const { return _vector_tail; }

  void setVector(T* vector) { this->_vector = vector; }

  void setVectorTail(uint32_t* vector_tail) { this->_vector_tail = vector_tail; }

protected:
  size_t _vector_max_size;
  uint32_t* _vector_tail;
  T* _vector;
};

/**
 * @class frontier_bitvec_t
 * @brief Represents a bitmap frontier used in SYgraph.
 *
 * The frontier_bitvec_t class provides functionality to manage a bitmap frontier in SYgraph.
 * It allows allocating and freeing memory for the bitmap, as well as retrieving the bitmap range.
 */
/**
 * @class frontier_bitvec_t
 * @brief Represents a bitmap frontier used in SYgraph.
 *
 * The frontier_bitvec_t class provides functionality for managing a bitmap frontier,
 * which is used to track the active elements in a graph traversal algorithm.
 *
 * @tparam bitmap_type The type of the bitmap.
 */
template<typename T, size_t Levels = 2>
class FrontierBitvec : public FrontierHierarchicBitmap<T, Levels, BitvecDevice<T, Levels>> {
public:
  using bitmap_type = typename HierarchicBitmapDevice<T, 2>::bitmap_type;
  using device_frontier_type = BitvecDevice<T, Levels, bitmap_type>;
  /**
   * @brief Constructs a frontier_bitvec_t object.
   *
   * @param q The SYCL queue to use for memory allocation.
   * @param num_elems The number of elements in the bitmap.
   */
  FrontierBitvec(sycl::queue& q, size_t num_elems) : FrontierHierarchicBitmap<T, Levels, BitvecDevice<T, 2>>{q, num_elems} {
    T* vector_ptr = sygraph::memory::detail::memoryAlloc<T, memory::space::device>(this->_bitmap.getVectorMaxSize(), this->_queue);
    uint32_t* vector_tail_ptr = sygraph::memory::detail::memoryAlloc<uint32_t, memory::space::device>(1, this->_queue);

    this->_bitmap.setVector(vector_ptr);
    this->_bitmap.setVectorTail(vector_tail_ptr);
  }

  /**
   * @brief Destroys the frontier_bitvec_t object and frees the allocated memory.
   */
  ~FrontierBitvec() {
    sycl::free(this->_bitmap.getVector(), this->_queue);
    sycl::free(this->_bitmap.getVectorTail(), this->_queue);
  }

  /**
   * @brief Clears the bitmap by setting all bits to 0. It performs only a single memory operation.
   * @note This function should be called only on the host-side.
   */
  void clear() {
    FrontierHierarchicBitmap<T, Levels, BitvecDevice<T, Levels>>::clear();
    this->_queue.fill(this->_bitmap.getVectorTail(), static_cast<bitmap_type>(0), 1).wait();
  }

  static void swap(FrontierBitvec<T, Levels>& a, FrontierBitvec<T, Levels>& b) { std::swap(a._bitmap, b._bitmap); }

  size_t getVectorSize() const {
    uint32_t ret;
    this->_queue.copy(this->_bitmap.getVectorTail(), &ret, 1).wait();
    return ret;
  }

  size_t getVectorMaxSize() const { return this->_bitmap.getVectorMaxSize(); }

  const T* getVector() const { return this->_bitmap.getVector(); }

  bool useVector() const { return this->getVectorSize() < this->getVectorMaxSize(); }
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph