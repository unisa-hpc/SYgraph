#pragma once

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/vector.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {


template <typename type_t>
class frontier_bitmap_t;


template <typename type_t, typename bitmap_t = uint64_t>
class bitmap_device_t {
public:
  using bitmap_type = bitmap_t;

  bitmap_device_t(size_t num_elems) : num_elems(num_elems)  {
    range = sizeof(bitmap_type) * sygraph::types::detail::byte_size;
    size = num_elems / range + (num_elems % range != 0);
  }
  
  /**
   * @brief Retrieves the size of the bitmap.
   * 
   * @return The size of the bitmap.
   */
  SYCL_EXTERNAL inline size_t get_bitmap_size() const {
    return size;
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
   * @brief Retrieves the range of the bitmap.
   * 
   * @return The range of the bitmap.
   */
  SYCL_EXTERNAL inline const size_t get_bitmap_range() const {
    return range;
  }

  /**
   * @brief Retrieves a pointer to the bitmap.
   * 
   * @return A pointer to the bitmap.
   */
  SYCL_EXTERNAL bitmap_type* get_data() const {
    return data;
  }

  /**
   * @brief Sets the bit at the specified index to the specified value.
   * 
   * @param idx The index of the bit to set.
   * @param val The value to set the bit to.
   */
  SYCL_EXTERNAL inline void set(type_t idx, bool val) const {
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
  SYCL_EXTERNAL inline void insert(type_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::work_group> ref(data[get_bitmap_index(idx)]);
    ref |= 1 << (idx % range);
  }

  /**
   * @brief Sets the bit at the specified index to 0.
   * 
   * @param idx The index of the bit to set.
   */
  SYCL_EXTERNAL inline void remove(type_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::work_group> ref(data[get_bitmap_index(idx)]);
    ref &= ~(1 << (idx % range));
  }

  /**
   * @brief Resets the bitmap by setting all bits to 0.
   * @note This function should be executed by a single work-item.
   */
  SYCL_EXTERNAL inline void reset() const {
    for (size_t i = 0; i < size; i++) {
      data[i] = 0;
    }
  }

  /**
   * @brief Resets the bit at the specified index to 0.
   * @details All the bits from [bitmap[id], bitmap[id + 1]) are set to 0.
   * 
   * @param id The index of the bitmap to reset.
   */
  SYCL_EXTERNAL inline void reset(type_t id) const {
    data[id] = 0;
  }

  /**
   * @brief Checks if the bit at the specified index is set.
   * 
   * @param idx The index of the bit to check.
   * @return True if the bit is set, false otherwise.
   */
  SYCL_EXTERNAL inline bool check(type_t idx) const {
    return data[get_bitmap_index(idx)] & (1 << (idx % range));
  }

  /**
   * @brief Retrieves the bitmap index for the specified index.
   * 
   * @param idx The index.
   * @return The bitmap index.
   */
  SYCL_EXTERNAL inline const type_t get_bitmap_index(type_t idx) const {
    return idx / range;
  }

  /**
   * @brief Retrieves the number of active elements in the bitmap.
   * @note This function should be called only on the host-side.
   * @todo Implement a more efficient version of this function.
   * @return The number of active elements in the bitmap.
   */
  SYCL_EXTERNAL size_t get_num_active_elements() const {

    size_t count = 0;

    for (size_t i = 0; i < size; i ++) {
      for (size_t j = 0; j < range; j++) {
        if (data[i] & (static_cast<bitmap_type>(1) << j)) {
          count++;
        }
      }
    }

    return count;
  }


  template<int Dim, typename group_t>
  SYCL_EXTERNAL size_t get_num_active_elements(sycl::nd_item<Dim> item, group_t group) const {
    size_t count = 0;
    auto id = item.get_local_linear_id();
    auto range = item.get_local_range(0);

    for (size_t i = id; i < size; i += range) {
      for (size_t j = 0; j < range; j++) {
        if (data[i] & (static_cast<bitmap_type>(1) << j)) {
          count++;
        }
      }
    }

    return sycl::reduce_over_group(group, count, sycl::plus<size_t>());
  }

  friend class frontier_bitmap_t<type_t>;
private:
  void set_ptr(bitmap_type* ptr) {
    data = ptr;
  }

  size_t range;            ///< The range of the bitmap.
  size_t num_elems;        ///< The number of elements in the bitmap.
  size_t size;             ///< The size of the bitmap.
  bitmap_type* data;     ///< Pointer to the bitmap.
};

template <typename type_t>
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
   * @todo tune on bitmap size
   * @param num_elems The number of elements in the bitmap.
   */
  frontier_bitmap_t(sycl::queue& q, size_t num_elems) : q(q), num_elems(num_elems), bitmap(num_elems) {
    using bitmap_type = typename bitmap_device_t<type_t>::bitmap_type;
    bitmap_type* ptr = sygraph::memory::detail::memory_alloc<bitmap_type, memory::space::device>(bitmap.get_bitmap_size(), q);
    range = bitmap.get_bitmap_range();
    size = bitmap.get_bitmap_size();
    q.memset(ptr, static_cast<bitmap_type>(0), size).wait();
    bitmap.set_ptr(ptr);
  }

  using bitmap_type = typename bitmap_device_t<type_t>::bitmap_type;

  /**
   * @brief Destroys the frontier_bitmap_t object and frees the allocated memory.
   */
  ~frontier_bitmap_t() {
    sycl::free(bitmap.get_data(), q);
  }

  inline size_t get_bitmap_size() const {
    return this->size;
  }

  inline size_t get_num_elems() const {
    return this->num_elems;
  }

  inline size_t get_bitmap_range() const {
    return this->range;
  }

  size_t get_num_active_elements() {
    bitmap_type* count = memory::detail::memory_alloc<bitmap_type, memory::space::shared>(1, q);

    q.submit([&](sycl::handler& h) {
      auto bitmap = this->get_device_frontier();
      h.single_task([=]() {
        *count = bitmap.get_num_active_elements();
      });
    }).wait();

    size_t ret = *count;
    sycl::free(count, q);
    return ret;
  }

  /**
   * @todo try to implement this operation with a gather in SYCL
   * @note now it is inefficient since it happens on the host
  */
  std::vector<type_t> get_active_elements() {
    std::vector<type_t> ret;
    for (type_t i = 0; i < num_elems; i++) {
      if (check(i)) {
        ret.push_back(i);
      }
    }
    return ret;
  }

  bool check(type_t idx) {
    sycl::buffer<bool, 1> ret(1);
    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();
      sycl::accessor ret_acc(ret, cgh, sycl::write_only);
      cgh.single_task([=]() {
        ret_acc[0] = bitmap.check(idx);
      });
    }).wait();
    sycl::host_accessor ret_acc(ret, sycl::read_only);
    return ret_acc[0];
  }

  void insert(type_t idx) {
    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();
      cgh.single_task([=]() {
        bitmap.insert(idx);
      });
    }).wait();
  }

  void remove(type_t idx) {
    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();
      cgh.single_task([=]() {
        bitmap.remove(idx);
      });
    }).wait();
  }

  /**
   * @brief Clears the bitmap by setting all bits to 0. It performs only a single memory operation.
   * @note This function should be called only on the host-side.
   */
  inline void clear() {
    q.memset(bitmap.get_data(), static_cast<bitmap_type>(0), bitmap.get_bitmap_size()).wait();
  }

  bitmap_device_t<type_t, bitmap_type>& get_device_frontier() {
    return bitmap;
  }

private:
  struct bitmap_host_t {
    sycl::queue& q;
    bitmap_type* data;

    bitmap_host_t (sycl::queue& q, bitmap_type* ptr) : q(q), data(ptr) {}

    bitmap_type operator[](size_t idx) const {
      return data[idx];
    }

    ~bitmap_host_t() {
      sycl::free(data, q);
    }
  };

  bitmap_host_t alloc_host() {
    auto ptr = bitmap.get_data();
    auto data = memory::detail::memory_alloc<bitmap_type, memory::space::host>(size, q);

    q.copy<bitmap_type>(ptr, data, size).wait();

    return bitmap_host_t{q, data};
  }

  size_t size, range, num_elems;
  sycl::queue& q;          ///< The SYCL queue used for memory allocation.
  bitmap_device_t<type_t, bitmap_type> bitmap; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph