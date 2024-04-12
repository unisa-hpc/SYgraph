#pragma once

#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {




template <typename type_t, typename bitmap_t = uint64_t>
class bitmap_device_t {
public:
  using bitmap_type = bitmap_t;

  bitmap_device_t(size_t num_elems) : num_elems(num_elems) {
    range = sizeof(bitmap_type) * 8;
    size = num_elems / range + 1;
  }

  void setPtr(bitmap_type* ptr) {
    data = ptr;
  }

  /**
   * @brief Retrieves the size of the bitmap.
   * 
   * @return The size of the bitmap.
   */
  SYCL_EXTERNAL inline size_t getSize() const {
    return size;
  }

  /**
   * @brief Retrieves the number of elements in the bitmap.
   * 
   * @return The number of elements in the bitmap.
   */
  SYCL_EXTERNAL inline size_t getNumElems() const {
    return num_elems;
  }

  /**
   * @brief Retrieves the range of the bitmap.
   * 
   * @return The range of the bitmap.
   */
  SYCL_EXTERNAL inline const size_t getBitmapRange() const {
    return range;
  }

  /**
   * @brief Retrieves a pointer to the bitmap.
   * 
   * @return A pointer to the bitmap.
   */
  SYCL_EXTERNAL bitmap_type* getData() {
    return data;
  }

  /**
   * @brief Prepares the bitmap for use.
   * 
   * @tparam t The types of the elements to prepare.
   */
  // void prepare(type_t... t) {

  // }

  /**
   * @brief Sets the bit at the specified index to 1.
   * 
   * @tparam idx_t The type of the index.
   * @param idx The index of the bit to set.
   */
  template<typename idx_t>
  SYCL_EXTERNAL inline void setOn(idx_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(data[getBitmapIndex(idx)]);
    ref.fetch_or(1 << (idx % range));
  }

  /**
   * @brief Sets the bit at the specified index to 0.
   * 
   * @tparam idx_t The type of the index.
   * @param idx The index of the bit to set.
   */
  template<typename idx_t>
  SYCL_EXTERNAL inline void setOff(idx_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(data[getBitmapIndex(idx)]);
    ref.fetch_and(~(1 << (idx % range)));
  }

  /**
   * @brief Resets the bitmap by setting all bits to 0.
   */
  SYCL_EXTERNAL inline void reset() const {
    for (size_t i = 0; i < size; i++) {
      data[i] = 0;
    }
  }

  /**
   * @brief Resets the bit at the specified index to 0.
   * 
   * @tparam idx_t The type of the index.
   * @param id The index of the bit to reset.
   */
  template<typename idx_t>
  SYCL_EXTERNAL inline void reset(idx_t id) const {
    data[id] = 0;
  }

  /**
   * @brief Checks if the bit at the specified index is set.
   * 
   * @tparam idx_t The type of the index.
   * @param idx The index of the bit to check.
   * @return True if the bit is set, false otherwise.
   */
  template<typename idx_t>
  SYCL_EXTERNAL inline bool check(idx_t idx) const {
    return data[getBitmapIndex(idx)] & (1 << (idx % range));
  }

  /**
   * @brief Retrieves the bitmap index for the specified index.
   * 
   * @tparam idx_t The type of the index.
   * @param idx The index.
   * @return The bitmap index.
   */
  template <typename idx_t>
  SYCL_EXTERNAL inline const idx_t getBitmapIndex(idx_t idx) const {
    return idx / range;
  }

  /**
   * @brief Retrieves the number of active elements in the bitmap.
   * @note This function should be called only on the host-side.
   * @return The number of active elements in the bitmap.
   */
  SYCL_EXTERNAL size_t getNumActiveElements() const {

    size_t count = 0;

    for (size_t i = 0; i < size; i++) {
      for (int j = 0; j < range; j++) {
        if (data[i] & (1 << j)) {
          count++;
        }
      }
    }

    return count;
  }

private:
  SYCL_EXTERNAL size_t countSetBits(bitmap_type N) const {
    uint count = 0;
    uint k=findk(N);
    uint val, x;
  
    for (int i = k; i >= 0; i--) {
      val=sycl::pow<float>(2,i);
      x=val & N; 
      
      if(x > 0){ count++;}
    }
    return count;
  }

  SYCL_EXTERNAL size_t findk(bitmap_type n) const { 
    int k; 
    int i=0; 
    int val=sycl::pow<float>(2,i);
    while(val<=n) { 
      k=i; i++;
      val=sycl::pow<float>(2,i);
    }
    return k;
  }

  size_t range;            ///< The range of the bitmap.
  size_t num_elems;        ///< The number of elements in the bitmap.
  size_t size;             ///< The size of the bitmap.
  bitmap_type* data;     ///< Pointer to the bitmap.
};

template <typename type_t, typename bitmap_t = uint64_t>
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
  using bitmap_type = bitmap_t;

  /**
   * @brief Constructs a frontier_bitmap_t object.
   * 
   * @param q The SYCL queue to use for memory allocation.
   * @param num_elems The number of elements in the bitmap.
   */
  frontier_bitmap_t(sycl::queue& q, size_t num_elems) : q(q), bitmap(num_elems) {
    bitmap_type* ptr = sygraph::memory::detail::memory_alloc<bitmap_type, memory::space::device>(bitmap.getSize(), q);
    bitmap.setPtr(ptr);
  }

  /**
   * @brief Destroys the frontier_bitmap_t object and frees the allocated memory.
   */
  ~frontier_bitmap_t() {
    sycl::free(bitmap.getData(), q);
  }

  inline size_t getSize() const {
    return bitmap.getSize();
  }

  inline size_t getNumElems() const {
    return bitmap.getNumElems();
  }

  inline size_t getBitmapRange() const {
    return bitmap.getBitmapRange();
  }

  size_t getNumActiveElements() {
    auto ptr = bitmap.getData();
    size_t size = bitmap.getSize();
    size_t range = bitmap.getBitmapRange();
    auto data = memory::detail::memory_alloc<bitmap_type, memory::space::host>(size, q);

    q.copy<bitmap_type>(ptr, data, size).wait();
    
    size_t count = 0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < range; j++) {
        if (data[i] & (1 << j)) {
          count++;
        }
      }
    }

    sycl::free(data, q);
    return count;
  }

  void insert(type_t idx) {
    throw std::runtime_error("Not implemented");
  }

  void remove(type_t idx) {
    throw std::runtime_error("Not implemented");
  }

  /**
   * @brief Clears the bitmap by setting all bits to 0. It performs only a single memory operation.
   * @note This function should be called only on the host-side.
   */
  inline void clear() {
    q.memset(bitmap.getData(), static_cast<bitmap_type>(0), bitmap.getSize()).wait();
  }

  bitmap_device_t<type_t, bitmap_type>& getDeviceBitmap() {
    return bitmap;
  }

private:

  sycl::queue& q;          ///< The SYCL queue used for memory allocation.
  bitmap_device_t<type_t, bitmap_type> bitmap; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph