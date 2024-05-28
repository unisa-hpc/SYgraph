#pragma once

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/vector.hpp>
#include <sygraph/utils/types.hpp>
#include <sygraph/sycl/event.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {


template <typename type_t>
class frontier_bitvec_t;

template <typename type_t, typename bitmap_t = uint64_t> //TODO [!!!] There are too many copies from host to device that degrade the performance
class bitvec_device_t {
public:
  using bitmap_type = bitmap_t;

  bitvec_device_t(size_t num_elems) : num_elems(num_elems)  {
    bitmap_range = sizeof(bitmap_type) * sygraph::types::detail::byte_size;
    bitmap_size = num_elems / bitmap_range + (num_elems % bitmap_range != 0);
    vector_max_size = 8192; // TODO ! tune on vector size
  }
  
  /**
   * @brief Retrieves the size of the bitmap.
   * 
   * @return The size of the bitmap.
   */
  SYCL_EXTERNAL inline size_t get_bitmap_size() const {
    return bitmap_size;
  }

  /**
   * @brief Retrieves the number of elements in the bitmap.
   * 
   * @return The number of elements in the bitmap.
   */
  SYCL_EXTERNAL inline size_t get_num_elems() const {
    return num_elems;
  }

  /**
   * @brief Retrieves the size of the vector.
   * 
   * @return The size of the vector.
   */
  SYCL_EXTERNAL inline size_t get_vector_size() const {
    return *vector_tail;
  }

  /**
   * @brief Retrieves the maximum size of the vector.
   * 
   * @return The maximum size of the vector.
   */
  SYCL_EXTERNAL inline size_t get_vector_max_size() const {
    return vector_max_size;
  }

  /**
   * @brief Retrieves the range of the bitmap.
   * 
   * @return The range of the bitmap.
   */
  SYCL_EXTERNAL inline const size_t get_bitmap_range() const {
    return bitmap_range;
  }

  SYCL_EXTERNAL inline bool use_vector() const {
    return *vector_tail < vector_max_size;
  }

  /**
   * @brief Retrieves a pointer to the bitmap.
   * 
   * @return A pointer to the bitmap.
   */
  SYCL_EXTERNAL bitmap_type* get_data() const {
    return bitmap;
  }

  SYCL_EXTERNAL type_t* get_vector() const {
    return vector;
  }

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
   * @param val The value of the bit to set.
   */
  SYCL_EXTERNAL inline bool insert(type_t val) const {
    insert_only_bitmap(val);
    insert_only_vector(val);
    
    return true;
  }

  SYCL_EXTERNAL inline bool insert_only_bitmap(type_t val) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> bitmap_ref(bitmap[get_bitmap_index(val)]);
    bitmap_ref |= static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (val % bitmap_range));
    return true;
  }

  SYCL_EXTERNAL inline bool insert_only_vector(type_t val) const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> tail_ref(vector_tail[0]);
    if (tail_ref < vector_max_size) {
      vector[tail_ref.fetch_add(1)] = val;
      return true;
    }
    return false;
  }

  /**
   * @brief Sets the bit at the specified index to 1.
   * 
   * @param idx The index of the bit to set.
   */
  SYCL_EXTERNAL inline bool insert(type_t val, size_t idx) const {
    return insert(val);
  }

  SYCL_EXTERNAL inline size_t prealloc(size_t num_elems) const {
    return 0;
  }

  /**
   * @brief Sets the bit at the specified index to 0.
   * 
   * @param idx The index of the bit to set.
   */
  SYCL_EXTERNAL inline bool remove(size_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(bitmap[get_bitmap_index(idx)]);
    ref &= ~(static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % bitmap_range)));
    return true;
  }

  /**
   * @brief Resets the bitmap by setting all bits to 0.
   * @note This function should be executed by a single work-item.
   */
  SYCL_EXTERNAL inline void reset_bitmap() const {
    for (size_t i = 0; i < bitmap_size; i++) {
      bitmap[i] = static_cast<bitmap_type>(0);
    }
  }

  /**
   * @brief Resets the bit at the specified index to 0.
   * @details All the bits from [bitmap[id], bitmap[id + 1]) are set to 0.
   * 
   * @param id The index of the bitmap to reset_bitmap.
   */
  SYCL_EXTERNAL inline void reset_bitmap(size_t id) const {
    bitmap[id] = static_cast<bitmap_type>(0);
  }

  SYCL_EXTERNAL inline void reset_vector() const {
    *vector_tail = 0;
  }

  /**
   * @brief Checks if the bit at the specified index is set.
   * 
   * @param idx The index of the bit to check.
   * @return True if the bit is set, false otherwise.
   */
  SYCL_EXTERNAL inline bool check(size_t idx) const {
    return bitmap[idx / bitmap_range] & (static_cast<bitmap_type>(1) << (idx % bitmap_range));
  }

  SYCL_EXTERNAL inline bool empty() const { // TODO it might be here the problem of the performance (too many copies from host to device)
    bitmap_type count = static_cast<bitmap_type>(0);
    for (auto i = 0; i < bitmap_size; i++) {
      count += bitmap[i];
    }
    return count == static_cast<bitmap_type>(0);
  
  }
  /**
   * @brief Retrieves the bitmap index for the specified index.
   * 
   * @param idx The index.
   * @return The bitmap index.
   */
  SYCL_EXTERNAL inline const size_t get_bitmap_index(size_t idx) const {
    return idx / bitmap_range;
  }

  template<int Dim, typename group_t>
  SYCL_EXTERNAL size_t get_num_active_elements(sycl::nd_item<Dim> item, group_t group) const {
    size_t count = 0;
    auto id = item.get_local_linear_id();
    auto local_range = item.get_local_range(0);

    for (size_t i = id; i < bitmap_size; i += local_range) {
      for (bitmap_type j = 0; j < bitmap_range; j++) {
        if (bitmap[i] & (static_cast<bitmap_type>(1) << j)) {
          count++;
        }
      }
    }

    return sycl::reduce_over_group(group, count, sycl::plus<size_t>());
  }

  friend class frontier_bitvec_t<type_t>;
private:
  void set_bitmap_ptr(bitmap_type* ptr) {
    bitmap = ptr;
  }

  void set_vector_ptr(type_t* ptr, size_t* tail_ptr) {
    vector = ptr;
    vector_tail = tail_ptr;
  }

  size_t num_elems;               ///< The number of elements

  size_t bitmap_range;            ///< The range of the bitmap.
  size_t bitmap_size;             ///< The size of the bitmap.
  bitmap_type* bitmap;            ///< Pointer to the bitmap.

  size_t vector_max_size;
  size_t* vector_tail;
  type_t* vector;
};

template <typename type_t>
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
    bitmap_type* bitmap_ptr = sygraph::memory::detail::memory_alloc<bitmap_type, memory::space::shared>(bitvec.get_bitmap_size(), q);
    type_t* vector_ptr = sygraph::memory::detail::memory_alloc<type_t, memory::space::shared>(bitvec.get_vector_max_size(), q);
    size_t* vector_tail_ptr = sygraph::memory::detail::memory_alloc<size_t, memory::space::shared>(1, q);
    auto size = bitvec.get_bitmap_size();
    q.memset(bitmap_ptr, static_cast<bitmap_type>(0), size).wait();
    bitvec.set_bitmap_ptr(bitmap_ptr);
    bitvec.set_vector_ptr(vector_ptr, vector_tail_ptr);
  }

  using bitmap_type = typename bitvec_device_t<type_t>::bitmap_type;

  /**
   * @brief Destroys the frontier_bitvec_t object and frees the allocated memory.
   */
  ~frontier_bitvec_t() {
    sycl::free(bitvec.get_data(), q);
    sycl::free(bitvec.vector, q);
    sycl::free(bitvec.vector_tail, q);
  }

  inline size_t get_bitmap_size() const {
    return bitvec.bitmap_size;
  }

  inline size_t get_num_elems() const {
    return bitvec.num_elems;
  }

  inline size_t get_bitmap_range() const {
    return bitvec.bitmap_range;
  }

  inline bool empty() const {
    return bitvec.empty();
  }

  bool check(size_t idx) const {
    return bitvec.check(idx);
  }

  bool insert(size_t idx) {
    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();
      cgh.single_task([=]() {
        bitmap.insert(idx);
      });
    }).wait();
    return true;
  }

  bool remove(size_t idx) {
    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();
      cgh.single_task([=]() {
        bitmap.remove(idx);
      });
    }).wait();
    return true;
  }

  size_t get_num_active_elements() const { // TODO: [!!!] this kernel is too slow, we need a better way to count the number of active elements
    size_t* count = memory::detail::memory_alloc<size_t, memory::space::shared>(1, q);

    sycl::nd_range<1> nd_range(128, 128); // TODO: [!] tune on these value

    q.submit([&](sycl::handler& h) {
      auto bitmap = this->get_device_frontier();

      h.parallel_for<class get_num_active_elements_kernel>(nd_range, [=](sycl::nd_item<1> item) {
        auto group = item.get_group();
        auto lcount = bitmap.get_num_active_elements(item, group);
        if (item.get_global_linear_id() == 0) {
          *count = lcount;
        }
      });
    }).wait();
    size_t ret = *count;
    sycl::free(count, q);
    return ret;
  }

  //operator =
  frontier_bitvec_t& operator=(const frontier_bitvec_t& other) {
    if (this == &other) {
      return *this;
    }
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
      auto bitmap = this->get_device_frontier();
      auto other_bitmap = other.get_device_frontier();
      cgh.parallel_for<class merge_bitmap_frontier_kernel>(sycl::range<1>(bitmap.size), [=](sycl::id<1> idx) {
        bitmap.data[idx] |= other_bitmap.data[idx];
      });
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
      auto bitmap = this->get_device_frontier();
      auto other_bitmap = other.get_device_frontier();
      cgh.parallel_for<class intersect_bitmap_frontier_kernel>(sycl::range<1>(bitmap.size), [=](sycl::id<1> idx) {
        bitmap.data[idx] &= other_bitmap.data[idx];
      });
    });
  }

  /**
   * @brief Clears the bitmap by setting all bits to 0. It performs only a single memory operation.
   * @note This function should be called only on the host-side.
   */
  inline void clear() {
    q.fill(bitvec.bitmap, static_cast<bitmap_type>(0), bitvec.bitmap_size).wait();
    bitvec.vector_tail[0] = 0;
  }

  const bitvec_device_t<type_t, bitmap_type>& get_device_frontier() const {
    return bitvec;
  }

  static void swap(frontier_bitvec_t<type_t>& a, frontier_bitvec_t<type_t>& b) {
    std::swap(a.bitvec, b.bitvec);
  }

private:
  sycl::queue& q;                               ///< The SYCL queue used for memory allocation.
  bitvec_device_t<type_t, bitmap_type> bitvec;  ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph