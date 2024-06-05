#pragma once

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
class frontier_bitmap_t;


template<typename type_t,
         typename bitmap_t = types::bitmap_type_t> // TODO [!!!] There are too many copies from host to device that degrade the performance
class bitmap_device_t {
public:
  using bitmap_type = bitmap_t;

  bitmap_device_t(size_t num_elems) : num_elems(num_elems) {
    range = sizeof(bitmap_type) * sygraph::types::detail::byte_size;
    size = num_elems / range + (num_elems % range != 0);
  }

  /**
   * @brief Retrieves the size of the bitmap.
   *
   * @return The size of the bitmap.
   */
  SYCL_EXTERNAL inline size_t getBitmapSize() const { return size; }

  /**
   * @brief Retrieves the number of elements in the bitmap.
   *
   * @return The number of elements in the bitmap.
   */
  SYCL_EXTERNAL inline size_t getNumElems() const { return num_elems; }

  /**
   * @brief Retrieves the range of the bitmap.
   *
   * @return The range of the bitmap.
   */
  SYCL_EXTERNAL inline const size_t getBitmapRange() const { return range; }

  /**
   * @brief Retrieves a pointer to the bitmap.
   *
   * @return A pointer to the bitmap.
   */
  SYCL_EXTERNAL bitmap_type* getData() const { return data; }

  /**
   * @brief Sets the bit at the specified index to the specified value.
   *
   * @param idx The index of the bit to set.
   * @param val The value to set the bit to.
   */
  SYCL_EXTERNAL inline void set(size_t idx, bool val) const {
    if (val) {
      insert(idx);
    } else {
      remove(idx);
    }
  }

  /**
   * @brief Sets the bit at the specified index to 1.
   *
   * @param idx The index of the bit to set.
   */
  SYCL_EXTERNAL inline bool insert(type_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(data[getBitmapIndex(idx)]);
    ref |= static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % range));
    return true;
  }

  SYCL_EXTERNAL inline size_t prealloc(size_t num_elems) const { return 0; }

  /**
   * @brief Sets the bit at the specified index to 0.
   *
   * @param idx The index of the bit to set.
   */
  SYCL_EXTERNAL inline bool remove(size_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(data[getBitmapIndex(idx)]);
    ref &= ~(static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % range)));
    return true;
  }

  /**
   * @brief Resets the bitmap by setting all bits to 0.
   * @note This function should be executed by a single work-item.
   */
  SYCL_EXTERNAL inline void reset() const {
    for (size_t i = 0; i < size; i++) { data[i] = static_cast<bitmap_type>(0); }
  }

  /**
   * @brief Resets the bit at the specified index to 0.
   * @details All the bits from [bitmap[id], bitmap[id + 1]) are set to 0.
   *
   * @param id The index of the bitmap to reset.
   */
  SYCL_EXTERNAL inline void reset(size_t id) const { data[id] = static_cast<bitmap_type>(0); }

  /**
   * @brief Checks if the bit at the specified index is set.
   *
   * @param idx The index of the bit to check.
   * @return True if the bit is set, false otherwise.
   */
  SYCL_EXTERNAL inline bool check(size_t idx) const { return data[idx / range] & (static_cast<bitmap_type>(1) << (idx % range)); }

  SYCL_EXTERNAL inline bool empty() const { // TODO it might be here the problem of the performance (too many copies from host to device)
    bitmap_type count = static_cast<bitmap_type>(0);
    for (auto i = 0; i < size; i++) { count += data[i]; }
    return count == static_cast<bitmap_type>(0);
  }
  /**
   * @brief Retrieves the bitmap index for the specified index.
   *
   * @param idx The index.
   * @return The bitmap index.
   */
  SYCL_EXTERNAL inline const size_t getBitmapIndex(size_t idx) const { return idx / range; }

  template<int Dim, typename group_t>
  SYCL_EXTERNAL size_t getNumActiveElements(sycl::nd_item<Dim> item, group_t group) const {
    size_t count = 0;
    auto id = item.get_local_linear_id();
    auto local_range = item.get_local_range(0);

    for (size_t i = id; i < size; i += local_range) {
      for (bitmap_type j = 0; j < range; j++) {
        if (data[i] & (static_cast<bitmap_type>(1) << j)) { count++; }
      }
    }

    return sycl::reduce_over_group(group, count, sycl::plus<size_t>());
  }

  SYCL_EXTERNAL inline int* getOffsets() const { return offsets; }

  SYCL_EXTERNAL inline size_t* getOffsetsSize() const { return offsets_size; }

  friend class frontier_bitmap_t<type_t>;

protected:
  void setPtr(bitmap_type* ptr, int* offsets, size_t* offsets_size) {
    data = ptr;
    this->offsets = offsets;
    this->offsets_size = offsets_size;
  }

  size_t range;      ///< The range of the bitmap.
  size_t num_elems;  ///< The number of elements in the bitmap.
  size_t size;       ///< The size of the bitmap.
  bitmap_type* data; ///< Pointer to the bitmap.

  int* offsets;
  size_t* offsets_size;
};

template<typename type_t>
/**
 * @class frontier_bitmap_t
 * @brief Represents a bitmap frontier used in SYgraph.
 *
 * The frontier_bitmap_t class provides functionality to manage a bitmap frontier in SYgraph.
 * It allows allocating and freeing memory for the bitmap, as well as retrieving the bitmap range.
 */
/**
 * @class frontier_bitmap_t
 * @brief Represents a bitmap frontier used in SYgraph.
 *
 * The frontier_bitmap_t class provides functionality for managing a bitmap frontier,
 * which is used to track the active elements in a graph traversal algorithm.
 *
 * @tparam bitmap_type The type of the bitmap.
 */
class frontier_bitmap_t {
public:
  /**
   * @brief Constructs a frontier_bitmap_t object.
   *
   * @param q The SYCL queue to use for memory allocation.
   * @param num_elems The number of elements in the bitmap.
   */
  frontier_bitmap_t(sycl::queue& q, size_t num_elems) : q(q), bitmap(num_elems) { // TODO: [!] tune on bitmap size
    using bitmap_type = typename bitmap_device_t<type_t>::bitmap_type;
    bitmap_type* ptr = sygraph::memory::detail::memoryAlloc<bitmap_type, memory::space::shared>(bitmap.getBitmapSize(), q);
    int* offsets = sygraph::memory::detail::memoryAlloc<int, memory::space::device>(bitmap.getBitmapSize(), q);
    size_t* offsets_size = sygraph::memory::detail::memoryAlloc<size_t, memory::space::shared>(1, q);
    auto size = bitmap.getBitmapSize();
    q.memset(ptr, static_cast<bitmap_type>(0), size).wait();
    q.fill(offsets_size, 0, size).wait();
    bitmap.setPtr(ptr, offsets, offsets_size);
  }

  using bitmap_type = typename bitmap_device_t<type_t>::bitmap_type;

  /**
   * @brief Destroys the frontier_bitmap_t object and frees the allocated memory.
   */
  ~frontier_bitmap_t() {
    sycl::free(bitmap.getData(), q);
    sycl::free(bitmap.getOffsets(), q);
    sycl::free(bitmap.getOffsetsSize(), q);
  }

  inline size_t getBitmapSize() const { return bitmap.size; }

  inline size_t getNumElems() const { return bitmap.num_elems; }

  inline size_t getBitmapRange() const { return bitmap.range; }

  inline bool selfAllocated() const { return false; }

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

  /**
   * @brief Retrieves the active elements in the bitmap.
   *
   * @param elems The array to store the active elements. It must be pre-allocated with shared-access.
   * @param active If true, it retrieves the active elements, otherwise the inactive elements.
   */
  void getActiveElements(type_t*& elems, size_t& size) const {
    constexpr size_t local = 32;
    sycl::range<1> local_size{local}; // TODO: [!] tuning on this value
    sycl::range<1> global_size{(bitmap.size > local ? bitmap.size + local - (bitmap.size % local) : local)};

    sycl::nd_range<1> nd_range(global_size, local_size);

    size_t bitmap_range = this->bitmap.getBitmapRange();

    sycl::buffer<size_t, 1> g_tail_buffer(sycl::range<1>(1));

    auto e = q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();

      sycl::local_accessor<type_t, 1> local_elems(bitmap_range * local_size, cgh);
      sycl::local_accessor<size_t, 1> l_tail(1, cgh);
      sycl::accessor tail_acc(g_tail_buffer, cgh, sycl::read_write);

      cgh.parallel_for<class get_active_elements_kernel>(
          nd_range, [=, bitmap_range = bitmap_range, bitmap_size = bitmap.size, data = bitmap.data](sycl::nd_item<1> item) {
            sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> l_tail_ref(
                l_tail[0]); // TODO: check if acq_rel works
            sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> g_tail_ref(tail_acc[0]);

            auto lid = item.get_local_linear_id();
            auto gid = item.get_global_linear_id();
            auto global_size = item.get_global_range(0);
            auto group = item.get_group();
            auto group_id = item.get_group_linear_id();
            auto group_size = item.get_local_range(0);

            if (lid == 0) {
              l_tail_ref = 0;
              if (gid == 0) { g_tail_ref = 0; }
            }

            sycl::group_barrier(item.get_group());

            if (gid < bitmap_size) {
              auto elem = data[gid];

              for (type_t i = 0; i < bitmap_range; i++) {
                if (elem & (static_cast<bitmap_type>(1) << i)) { local_elems[l_tail_ref++] = i + gid * bitmap_range; }
              }
            }

            sycl::group_barrier(group);

            size_t our_slice = 0;
            if (lid == 0) { our_slice = g_tail_ref.fetch_add(l_tail_ref.load()); }
            our_slice = sycl::group_broadcast(group, our_slice, 0);
            for (size_t i = lid; i < l_tail_ref.load(); i += group_size) { elems[our_slice + i] = local_elems[i]; }
          });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::profiler::addEvent(e, "getActiveElements");
#endif
    size = g_tail_buffer.get_host_access()[0];
  }

  inline bool empty() const { return bitmap.empty(); }

  bool check(size_t idx) const { return bitmap.check(idx); }

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

  // operator =
  frontier_bitmap_t& operator=(const frontier_bitmap_t& other) {
    if (this == &other) { return *this; }
    q.copy(other.bitmap.data, this->bitmap.data, bitmap.size).wait();
    return *this;
  }

  /**
   * Merges the contents of the current bitmap frontier with the specified frontier.
   *
   * @param other The frontier to merge with.
   * @return The event associated with the operation.
   * @post The current frontier contains the union of the current frontier and the specified frontier. The specified frontier is not modified.
   */
  sygraph::event merge(frontier_bitmap_t<type_t>& other) {
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
  sygraph::event intersect(frontier_bitmap_t<type_t>& other) {
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
    q.fill(bitmap.data, static_cast<bitmap_type>(0), bitmap.size).wait();
    q.fill(bitmap.offsets_size, 0, 1).wait();
  }

  const bitmap_device_t<type_t, bitmap_type>& getDeviceFrontier() const { return bitmap; }

  const size_t computeActiveFrontier() const {
    sycl::range<1> local_range{128}; // TODO: [!] tune on this value
    size_t size = bitmap.getBitmapSize();
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
                         sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> offsets_size_ref{offsets_size[0]};

                         if (group.leader()) { local_size_ref.store(0); }
                         sycl::group_barrier(group);

                         if (bitmap.getData()[gid] != 0) { local_offsets[local_size_ref++] = gid; }
                         sycl::group_barrier(group);

                         size_t data_offset = 0;
                         if (group.leader()) { data_offset = offsets_size_ref.fetch_add(local_size_ref.load()); }
                         data_offset = sycl::group_broadcast(group, data_offset, 0);
                         if (lid < local_size_ref.load()) { offsets[data_offset + lid] = local_offsets[lid]; }
                       });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::profiler::addEvent(e, "computeActiveFrontier");
#endif
    return bitmap.offsets_size[0];
  }

  static void swap(frontier_bitmap_t<type_t>& a, frontier_bitmap_t<type_t>& b) { std::swap(a.bitmap, b.bitmap); }

private:
  sycl::queue& q;                              ///< The SYCL queue used for memory allocation.
  bitmap_device_t<type_t, bitmap_type> bitmap; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph