#pragma once

#include <sygraph/frontier/impls/bitmap_frontier.hpp>
#include <sygraph/frontier/impls/vector_frontier.hpp>
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


template<typename T>
class FrontierBitvec;

template<typename T, typename B = uint64_t> // TODO [!!!] There are too many copies from host to device that degrade the performance
class BitvecDevice : public BitmapDevice<T, B> {
public:
  using bitmap_type = B;

  BitvecDevice(size_t num_elems) : BitmapDevice<T, B>(num_elems) {
    _vector_max_size = 32000; // TODO ! tune on vector size
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
    BitmapDevice<T, B>::insert(val);
    insertOnlyVector(val);

    return true;
  }

  template<sycl::memory_order MO, sycl::memory_scope MS>
  SYCL_EXTERNAL inline bool insert(T val, const sycl::local_accessor<T, 1>& pad, const sycl::atomic_ref<T, MO, MS>& pad_tail) const {
    BitmapDevice<T, B>::insert(val);
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

  SYCL_EXTERNAL uint32_t* getVectorSizePtr() const { return _vector_tail; }

  friend class FrontierBitvec<T>;

protected:
  void setPtr(bitmap_type* bitmap_ptr, int* offsets_ptr, uint32_t* offsets_size_ptr, T* vector_ptr, uint32_t* tail_ptr) {
    BitmapDevice<T, B>::setPtr(bitmap_ptr, offsets_ptr, offsets_size_ptr);
    _vector = vector_ptr;
    _vector_tail = tail_ptr;
  }

  size_t _vector_max_size;
  uint32_t* _vector_tail;
  T* _vector;
};

template<typename T>
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
class FrontierBitvec {
public:
  /**
   * @brief Constructs a frontier_bitvec_t object.
   *
   * @param q The SYCL queue to use for memory allocation.
   * @param num_elems The number of elements in the bitmap.
   */
  FrontierBitvec(sycl::queue& q, size_t num_elems) : _queue(q), _bitvec(num_elems) { // TODO: [!] tune on bitmap size
    using bitmap_type = typename BitvecDevice<T>::bitmap_type;
    bitmap_type* bitmap_ptr = sygraph::memory::detail::memoryAlloc<bitmap_type, memory::space::shared>(_bitvec.getBitmapSize(), _queue);
    T* vector_ptr = sygraph::memory::detail::memoryAlloc<T, memory::space::device>(_bitvec.getVectorMaxSize(), _queue);
    uint32_t* vector_tail_ptr = sygraph::memory::detail::memoryAlloc<uint32_t, memory::space::device>(1, _queue);
    int* offsets = sygraph::memory::detail::memoryAlloc<int, memory::space::device>(_bitvec.getBitmapSize(), _queue);
    uint32_t* offsets_size = sygraph::memory::detail::memoryAlloc<uint32_t, memory::space::shared>(1, _queue);
    auto size = _bitvec.getBitmapSize();
    _queue.memset(bitmap_ptr, static_cast<bitmap_type>(0), size).wait();
    _bitvec.setPtr(bitmap_ptr, offsets, offsets_size, vector_ptr, vector_tail_ptr);
  }

  using bitmap_type = typename BitvecDevice<T>::bitmap_type;

  /**
   * @brief Destroys the frontier_bitvec_t object and frees the allocated memory.
   */
  ~FrontierBitvec() {
    sycl::free(_bitvec.getData(), _queue);
    sycl::free(_bitvec._vector, _queue);
    sycl::free(_bitvec._vector_tail, _queue);
  }

  size_t getBitmapSize() const { return _bitvec.getBitmapSize(); }

  size_t getNumElems() const { return _bitvec.getNumElems(); }

  size_t getBitmapRange() const { return _bitvec.getBitmapRange(); }

  bool empty() const { return _bitvec.empty(); }

  bool check(size_t idx) const { return _bitvec.check(idx); }

  bool insert(size_t idx) {
    _queue
        .submit([&](sycl::handler& cgh) {
          auto bitmap = this->getDeviceFrontier();
          cgh.single_task([=]() { bitmap.insert(idx); });
        })
        .wait();
    return true;
  }

  bool remove(size_t idx) {
    _queue
        .submit([&](sycl::handler& cgh) {
          auto bitmap = this->getDeviceFrontier();
          cgh.single_task([=]() { bitmap.remove(idx); });
        })
        .wait();
    return true;
  }

  size_t getNumActiveElements() const { // TODO: [!!!] this kernel is too slow, we need a better way to count the number of active elements
    size_t* count = memory::detail::memoryAlloc<size_t, memory::space::shared>(1, _queue);

    sycl::nd_range<1> nd_range(128, 128); // TODO: [!] tune on these value

    _queue
        .submit([&](sycl::handler& h) {
          auto bitmap = this->getDeviceFrontier();

          h.parallel_for<class get_num_active_elements_kernel>(nd_range, [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            auto lcount = bitmap.getNumActiveElements(item, group);
            if (item.get_global_linear_id() == 0) { *count = lcount; }
          });
        })
        .wait();
    size_t ret = *count;
    sycl::free(count, _queue);
    return ret;
  }

  // operator =
  FrontierBitvec& operator=(const FrontierBitvec& other) {
    if (this == &other) { return *this; }
    _queue.copy(other._bitvec.bitmap, this->_bitvec.bitmap, _bitvec.bitmap_size).wait();
    return *this;
  }

  /**
   * Merges the contents of the current bitmap frontier with the specified frontier.
   *
   * @param other The frontier to merge with.
   * @return The event associated with the operation.
   * @post The current frontier contains the union of the current frontier and the specified frontier. The specified frontier is not modified.
   */
  sygraph::Event merge(FrontierBitvec<T>& other) {
    return _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<class merge_bitmap_frontier_kernel>(sycl::range<1>(bitmap.size),
                                                           [=](sycl::id<1> idx) { bitmap.data[idx] |= other_bitmap.data[idx]; });
    });
  }

  /**
   * @brief Intersects the contents of the current bitmap frontier with the specified frontier.
   *
   * @param other The frontier to intersect with.
   * @return The event associated with the operation.
   * @post The current frontier contains the intersection of the current frontier and the specified frontier. The specified frontier is not modified.
   */
  sygraph::Event intersect(FrontierBitvec<T>& other) {
    return _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<class intersect_bitmap_frontier_kernel>(sycl::range<1>(bitmap.size),
                                                               [=](sycl::id<1> idx) { bitmap.data[idx] &= other_bitmap.data[idx]; });
    });
  }

  /**
   * @brief Clears the bitmap by setting all bits to 0. It performs only a single memory operation.
   * @note This function should be called only on the host-side.
   */
  void clear() {
    auto e1 = _queue.fill(_bitvec.getData(), static_cast<bitmap_type>(0), _bitvec.getBitmapSize());
    auto e2 = _queue.fill(_bitvec._vector_tail, static_cast<size_t>(0), 1);
    e1.wait();
    e2.wait();
  }

  const BitvecDevice<T, bitmap_type>& getDeviceFrontier() const { return _bitvec; }

  static void swap(FrontierBitvec<T>& a, FrontierBitvec<T>& b) { std::swap(a._bitvec, b._bitvec); }

  size_t getVectorSize() const {
    uint32_t ret;
    _queue.copy(_bitvec._vector_tail, &ret, 1).wait();
    return ret;
  }

  size_t getVectorMaxSize() const { return _bitvec.getVectorMaxSize(); }

  const T* getVector() const { return _bitvec.getVector(); }

  bool useVector() const { return this->getVectorSize() < this->getVectorMaxSize(); }

  size_t computeActiveFrontier() const {
    sycl::range<1> local_range{1024}; // TODO: [!] tune on this value
    size_t size = _bitvec.getBitmapSize();
    sycl::range<1> global_range{(size > local_range[0] ? size + local_range[0] - (size % local_range[0]) : local_range[0])};

    auto e = _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();

      sycl::local_accessor<int, 1> local_offsets(local_range[0], cgh);
      sycl::local_accessor<size_t, 1> local_size(1, cgh);
      bitmap.offsets_size[0] = 0;

      cgh.parallel_for(sycl::nd_range<1>{global_range, local_range},
                       [=, offsets_size = bitmap.offsets_size, offsets = bitmap.offsets](sycl::nd_item<1> item) {
                         int gid = item.get_global_linear_id();
                         size_t lid = item.get_local_linear_id();
                         auto group = item.get_group();

                         sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> local_size_ref(local_size[0]);
                         sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> offsets_size_ref{offsets_size[0]};

                         if (group.leader()) { local_size_ref.store(0); }
                         sycl::group_barrier(group);

                         if (bitmap.getData()[gid] != 0) { local_offsets[local_size_ref++] = gid; }
                         sycl::group_barrier(group);

                         uint32_t data_offset = 0;
                         if (group.leader()) { data_offset = offsets_size_ref.fetch_add(local_size_ref.load()); }
                         data_offset = sycl::group_broadcast(group, data_offset, 0);
                         if (lid < local_size_ref.load()) { offsets[data_offset + lid] = local_offsets[lid]; }
                       });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "computeActiveFrontier");
#endif
    return _bitvec.getOffsetsSize()[0];
  }

private:
  sycl::queue& _queue;                  ///< The SYCL queue used for memory allocation.
  BitvecDevice<T, bitmap_type> _bitvec; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph