#pragma once

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/vector.hpp>
#include <sygraph/utils/types.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/frontier/impls/bitmap_frontier.hpp>
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
class bitvec_device_t : public bitmap_device_t<bitmap_t> {
public:
  using bitmap_type = bitmap_t;

  bitvec_device_t(size_t num_elems) : bitmap_device_t<bitmap_t>(num_elems) {    
    vector_max_size = 8192; // TODO ! tune on vector size
  }

  SYCL_EXTERNAL inline bool use_vector() const {
    return *vector_tail < vector_max_size;
  }

  SYCL_EXTERNAL type_t* get_vector() const {
    return vector;
  }

  /**
   * @brief Sets the bit at the specified index to 1.
   * 
   * @param val The value of the bit to set.
   */
  SYCL_EXTERNAL inline bool insert(type_t val) const {
    // call super class insert
    bitmap_device_t<bitmap_t>::insert(val);
    insert_only_vector(val);
  
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

  SYCL_EXTERNAL inline void reset_vector() const {
    *vector_tail = 0;
  }

  SYCL_EXTERNAL size_t get_vector_max_size() const {
    return vector_max_size;
  }

  SYCL_EXTERNAL size_t get_vector_size() const {
    return *vector_tail;
  }

  friend class frontier_bitvec_t<type_t>;
protected:
  void set_ptr(bitmap_type* bitmap_ptr, type_t* vector_ptr, size_t* tail_ptr) {
    bitmap_device_t<bitmap_t>::set_ptr(bitmap_ptr);
    vector = vector_ptr;
    vector_tail = tail_ptr;
  }

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
    bitvec.set_ptr(bitmap_ptr, vector_ptr, vector_tail_ptr);
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
    return bitvec.get_bitmap_size();
  }

  inline size_t get_num_elems() const {
    return bitvec.get_num_elems();
  }

  inline size_t get_bitmap_range() const {
    return bitvec.get_bitmap_range();
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
    q.fill(bitvec.get_data(), static_cast<bitmap_type>(0), bitvec.get_bitmap_size()).wait();
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