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


template<typename type_t>
class frontier_bitvec_t;

template<typename type_t, typename bitmap_t = uint64_t> // TODO [!!!] There are too many copies from host to device that degrade the performance
class bitvec_device_t : public bitmap_device_t<type_t, bitmap_t> {
public:
  using bitmap_type = bitmap_t;

  bitvec_device_t(size_t num_elems) : bitmap_device_t<type_t, bitmap_t>(num_elems) {
    vector_max_size = 22000; // TODO ! tune on vector size
  }

  SYCL_EXTERNAL inline bool useVector() const { return *vector_tail < vector_max_size; }

  SYCL_EXTERNAL type_t* getVector() const { return vector; }

  /**
   * @brief Sets the bit at the specified index to 1.
   *
   * @param val The value of the bit to set.
   */
  SYCL_EXTERNAL inline bool insert(type_t val) const {
    // call super class insert
    bitmap_device_t<type_t, bitmap_t>::insert(val);
    insertOnlyVector(val);

    return true;
  }

  template<typename T, sycl::memory_order MO, sycl::memory_scope MS>
  SYCL_EXTERNAL inline bool insert(type_t val, const sycl::local_accessor<type_t, 1>& pad, const sycl::atomic_ref<T, MO, MS>& pad_tail) const {
    bitmap_device_t<type_t, bitmap_t>::insert(val);
    if (pad_tail.load() < vector_max_size) {
      pad[pad_tail++] = val;
      return true;
    }
    return false;
  }

  template<typename T, sycl::memory_order MO, sycl::memory_scope MS>
  SYCL_EXTERNAL inline bool
  finalize(sycl::nd_item<1> item, const sycl::local_accessor<type_t, 1>& pad, const sycl::atomic_ref<T, MO, MS>& pad_tail_ref) const {
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> tail_ref(vector_tail[0]);

    size_t data_offset = vector_max_size;
    if (group.leader()) { data_offset = tail_ref.fetch_add(pad_tail_ref.load()); }
    data_offset = sycl::group_broadcast(group, data_offset, 0);
    for (int i = lid; i < pad_tail_ref.load() && i < vector_max_size; i += item.get_local_range(0)) { vector[data_offset + i] = pad[i]; }

    return true;
  }

  SYCL_EXTERNAL inline bool insertOnlyVector(type_t val) const {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> tail_ref(vector_tail[0]);
    if (tail_ref.load() < vector_max_size) {
      vector[tail_ref++] = val;
      return true;
    }
    return false;
  }

  SYCL_EXTERNAL inline void resetVector() const { *vector_tail = 0; }

  SYCL_EXTERNAL size_t getVectorMaxSize() const { return vector_max_size; }

  SYCL_EXTERNAL uint32_t getVectorSize() const { return *vector_tail; }

  SYCL_EXTERNAL uint32_t* getVectorSizePtr() const { return vector_tail; }

  friend class frontier_bitvec_t<type_t>;

protected:
  void setPtr(bitmap_type* bitmap_ptr, int* offsets_ptr, uint32_t* offsets_size_ptr, type_t* vector_ptr, uint32_t* tail_ptr) {
    bitmap_device_t<type_t, bitmap_t>::setPtr(bitmap_ptr, offsets_ptr, offsets_size_ptr);
    vector = vector_ptr;
    vector_tail = tail_ptr;
  }

  size_t vector_max_size;
  uint32_t* vector_tail;
  type_t* vector;
};

template<typename type_t>
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
class frontier_bitvec_t {
public:
  /**
   * @brief Constructs a frontier_bitvec_t object.
   *
   * @param q The SYCL queue to use for memory allocation.
   * @param num_elems The number of elements in the bitmap.
   */
  frontier_bitvec_t(sycl::queue& q, size_t num_elems) : q(q), bitvec(num_elems) { // TODO: [!] tune on bitmap size
    using bitmap_type = typename bitvec_device_t<type_t>::bitmap_type;
    bitmap_type* bitmap_ptr = sygraph::memory::detail::memoryAlloc<bitmap_type, memory::space::shared>(bitvec.getBitmapSize(), q);
    type_t* vector_ptr = sygraph::memory::detail::memoryAlloc<type_t, memory::space::device>(bitvec.getVectorMaxSize(), q);
    uint32_t* vector_tail_ptr = sygraph::memory::detail::memoryAlloc<uint32_t, memory::space::device>(1, q);
    int* offsets = sygraph::memory::detail::memoryAlloc<int, memory::space::device>(bitvec.getBitmapSize(), q);
    uint32_t* offsets_size = sygraph::memory::detail::memoryAlloc<uint32_t, memory::space::shared>(1, q);
    auto size = bitvec.getBitmapSize();
    q.memset(bitmap_ptr, static_cast<bitmap_type>(0), size).wait();
    bitvec.setPtr(bitmap_ptr, offsets, offsets_size, vector_ptr, vector_tail_ptr);
  }

  using bitmap_type = typename bitvec_device_t<type_t>::bitmap_type;

  /**
   * @brief Destroys the frontier_bitvec_t object and frees the allocated memory.
   */
  ~frontier_bitvec_t() {
    sycl::free(bitvec.getData(), q);
    sycl::free(bitvec.vector, q);
    sycl::free(bitvec.vector_tail, q);
  }

  inline size_t getBitmapSize() const { return bitvec.getBitmapSize(); }

  inline size_t getNumElems() const { return bitvec.getNumElems(); }

  inline size_t getBitmapRange() const { return bitvec.getBitmapRange(); }

  inline bool empty() const { return bitvec.empty(); }

  bool check(size_t idx) const { return bitvec.check(idx); }

  bool insert(size_t idx) {
    q.submit([&](sycl::handler& cgh) {
       auto bitmap = this->getDeviceFrontier();
       cgh.single_task([=]() { bitmap.insert(idx); });
     }).wait();
    return true;
  }

  bool remove(size_t idx) {
    q.submit([&](sycl::handler& cgh) {
       auto bitmap = this->getDeviceFrontier();
       cgh.single_task([=]() { bitmap.remove(idx); });
     }).wait();
    return true;
  }

  size_t getNumActiveElements() const { // TODO: [!!!] this kernel is too slow, we need a better way to count the number of active elements
    size_t* count = memory::detail::memoryAlloc<size_t, memory::space::shared>(1, q);

    sycl::nd_range<1> nd_range(128, 128); // TODO: [!] tune on these value

    q.submit([&](sycl::handler& h) {
       auto bitmap = this->getDeviceFrontier();

       h.parallel_for<class get_num_active_elements_kernel>(nd_range, [=](sycl::nd_item<1> item) {
         auto group = item.get_group();
         auto lcount = bitmap.getNumActiveElements(item, group);
         if (item.get_global_linear_id() == 0) { *count = lcount; }
       });
     }).wait();
    size_t ret = *count;
    sycl::free(count, q);
    return ret;
  }

  // operator =
  frontier_bitvec_t& operator=(const frontier_bitvec_t& other) {
    if (this == &other) { return *this; }
    q.copy(other.bitvec.bitmap, this->bitvec.bitmap, bitvec.bitmap_size).wait();
    return *this;
  }

  /**
   * Merges the contents of the current bitmap frontier with the specified frontier.
   *
   * @param other The frontier to merge with.
   * @return The event associated with the operation.
   * @post The current frontier contains the union of the current frontier and the specified frontier. The specified frontier is not modified.
   */
  sygraph::event merge(frontier_bitvec_t<type_t>& other) {
    return q.submit([&](sycl::handler& cgh) {
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
  sygraph::event intersect(frontier_bitvec_t<type_t>& other) {
    return q.submit([&](sycl::handler& cgh) {
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
  inline void clear() {
    auto e1 = q.fill(bitvec.getData(), static_cast<bitmap_type>(0), bitvec.getBitmapSize());
    auto e2 = q.fill(bitvec.vector_tail, static_cast<size_t>(0), 1);
    e1.wait();
    e2.wait();
  }

  const bitvec_device_t<type_t, bitmap_type>& getDeviceFrontier() const { return bitvec; }

  static void swap(frontier_bitvec_t<type_t>& a, frontier_bitvec_t<type_t>& b) { std::swap(a.bitvec, b.bitvec); }

  const size_t getVectorSize() const {
    uint32_t ret;
    q.copy(bitvec.vector_tail, &ret, 1).wait();
    return ret;
  }

  const size_t getVectorMaxSize() const { return bitvec.getVectorMaxSize(); }

  const type_t* getVector() const { return bitvec.getVector(); }

  const bool useVector() const { return this->getVectorSize() < this->getVectorMaxSize(); }

  const size_t computeActiveFrontier() const {
    sycl::range<1> local_range{1024}; // TODO: [!] tune on this value
    size_t size = bitvec.getBitmapSize();
    sycl::range<1> global_range{(size > local_range[0] ? size + local_range[0] - (size % local_range[0]) : local_range[0])};

    auto e = q.submit([&](sycl::handler& cgh) {
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
    sygraph::profiler::addEvent(e, "computeActiveFrontier");
#endif
    return bitvec.getOffsetsSize()[0];
  }

private:
  sycl::queue& q;                              ///< The SYCL queue used for memory allocation.
  bitvec_device_t<type_t, bitmap_type> bitvec; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph