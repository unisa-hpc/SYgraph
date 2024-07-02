#pragma once

#include <sygraph/frontier/impls/kernels.hpp>
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
class FrontierHierarchicBitmap;


template<typename T, size_t Levels, typename B = uint32_t>
class HierarchicBitmapDevice {
public:
  using bitmap_type = B;

  template<typename... Args>
  HierarchicBitmapDevice(size_t num_elems) : _num_elems(num_elems) {
    _range = sizeof(bitmap_type) * sygraph::types::detail::byte_size;
    _size[0] = num_elems / _range + (num_elems % _range != 0);

    for (size_t i = 1; i < Levels; i++) { _size[i] = _size[i - 1] / _range + (_size[i - 1] % _range != 0); }
  }

  SYCL_EXTERNAL inline size_t getBitmapSize() const { return _size[0]; }

  SYCL_EXTERNAL inline size_t getNumElems() const { return _num_elems; }

  SYCL_EXTERNAL inline const size_t getBitmapRange() const { return _range; }

  SYCL_EXTERNAL inline bitmap_type* getData() const { return _data[0]; }

  SYCL_EXTERNAL inline void set(size_t idx, bool val) const {
    if (val) {
      insert(idx);
    } else {
      remove(idx);
    }
  }

  SYCL_EXTERNAL inline bool insert(T idx) const {
#pragma unroll
    for (size_t i = 0; i < Levels; i++) {
      T lidx = idx;
      for (size_t _ = 0; _ < i; _++) { lidx /= _range; } // the index must be divided by the range^level
      if (!(_data[i][getBitmapIndex(lidx)] & (static_cast<bitmap_type>(1) << (lidx % _range)))) {
        sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_data[i][getBitmapIndex(lidx)]);
        ref |= static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (lidx % _range));
      }
    }
    return true;
  }

  SYCL_EXTERNAL inline size_t prealloc(size_t num_elems) const { return 0; }

  SYCL_EXTERNAL inline bool remove(size_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_data[0][getBitmapIndex(idx)]);
    ref &= ~(static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % _range)));
    return true;
  }

  SYCL_EXTERNAL inline void reset() const {
    for (size_t i = 0; i < _size; i++) { _data[i] = static_cast<bitmap_type>(0); }
  }

  SYCL_EXTERNAL inline void reset(size_t id) const { _data[id] = static_cast<bitmap_type>(0); }

  SYCL_EXTERNAL inline bool check(size_t idx) const { return _data[0][idx / _range] & (static_cast<bitmap_type>(1) << (idx % _range)); }

  SYCL_EXTERNAL inline bool empty() const { // TODO it might be here the problem of the performance (too many copies from host to device)
    bitmap_type count = static_cast<bitmap_type>(0);
    for (auto i = 0; i < _size[Levels - 1]; i++) { count += _data[Levels - 1][i]; }
    return count == static_cast<bitmap_type>(0);
  }

  SYCL_EXTERNAL inline const size_t getBitmapIndex(size_t idx) const { return idx / _range; }

  SYCL_EXTERNAL inline int* getOffsets() const { return _offsets; }

  SYCL_EXTERNAL inline uint32_t* getOffsetsSize() const { return _offsets_size; }

  SYCL_EXTERNAL inline size_t getBitmapSize(const uint level) const { return _size[level]; }

  SYCL_EXTERNAL inline bitmap_type* getData(const uint level) const { return _data[level]; }

  SYCL_EXTERNAL inline bool check(const uint level, size_t idx) const {
    return _data[level][idx / _range] & (static_cast<bitmap_type>(1) << (idx % _range));
  }

  friend class FrontierHierarchicBitmap<T, Levels>;

protected:
  void setPtr(bitmap_type* ptr[Levels], int* offsets, uint32_t* offsets_size) {
    for (size_t i = 0; i < Levels; i++) { this->_data[i] = ptr[i]; }
    this->_offsets = offsets;
    this->_offsets_size = offsets_size;
  }


  uint _range;                ///< The range of the bitmap.
  size_t _num_elems;          ///< The number of elements in the bitmap.
  size_t _size[Levels];       ///< The size of the bitmap.
  bitmap_type* _data[Levels]; ///< Pointer to the bitmap.

  int* _offsets;
  uint32_t* _offsets_size;
};

template<typename T, size_t Levels = 2>
class FrontierHierarchicBitmap {
public:
  using bitmap_type = typename HierarchicBitmapDevice<T, Levels>::bitmap_type;
  using device_frontier_type = HierarchicBitmapDevice<T, Levels>;

  FrontierHierarchicBitmap(sycl::queue& q, size_t num_elems) : _queue(q), _bitmap(num_elems) { // TODO: [!] tune on bitmap size

    bitmap_type* ptr[Levels];
#pragma unroll
    for (size_t i = 0; i < Levels; i++) {
      size_t size = _bitmap.getBitmapSize(i);
      ptr[i] = sygraph::memory::detail::memoryAlloc<bitmap_type, memory::space::shared>(size, _queue);
      _queue.fill(ptr[i], static_cast<bitmap_type>(0), size);
    }
    _queue.wait();
    int* offsets = sygraph::memory::detail::memoryAlloc<int, memory::space::device>(_bitmap.getBitmapSize(), _queue);
    uint32_t* offsets_size = sygraph::memory::detail::memoryAlloc<uint32_t, memory::space::shared>(1, _queue);
    auto size = _bitmap.getBitmapSize();
    _queue.fill(offsets_size, 0, size).wait();
    _bitmap.setPtr(ptr, offsets, offsets_size);
  }

  ~FrontierHierarchicBitmap() {
    for (size_t i = 0; i < Levels; i++) { sycl::free(_bitmap.getData(i), _queue); }
    sycl::free(_bitmap.getOffsets(), _queue);
    sycl::free(_bitmap.getOffsetsSize(), _queue);
  }

  size_t getBitmapSize() const { return _bitmap.getBitmapSize(); }

  size_t getNumElems() const { return _bitmap.getNumElems(); }

  size_t getBitmapRange() const { return _bitmap.getBitmapRange(); }

  bool selfAllocated() const { return false; }

  bool empty() const { return _bitmap.empty(); }

  bool check(size_t idx) const {
    return _queue
        .submit([&](sycl::handler& cgh) {
          auto bitmap = this->getDeviceFrontier();
          cgh.single_task([=]() { bitmap.check(idx); });
        })
        .wait();
  }

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

  // operator =
  FrontierHierarchicBitmap& operator=(const FrontierHierarchicBitmap& other) {
    if (this == &other) { return *this; }
    _queue.copy(other._bitmap.getData(), this->_bitmap.getData(), _bitmap.getBitmapSize()).wait();
    return *this;
  }

  sygraph::Event merge(FrontierHierarchicBitmap<T>& other) {
    return _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<class merge_bitmap_frontier_kernel>(sycl::range<1>(bitmap.getBitmapSize()),
                                                           [=](sycl::id<1> idx) { bitmap.getData()[idx] |= other_bitmap.getData()[idx]; });
    });
  }

  sygraph::Event intersect(FrontierHierarchicBitmap<T>& other) {
    return _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<class intersect_bitmap_frontier_kernel>(sycl::range<1>(bitmap.getBitmapSize()),
                                                               [=](sycl::id<1> idx) { bitmap.getData()[idx] &= other_bitmap.getData()[idx]; });
    });
  }


  void clear() {
    for (size_t i = 0; i < Levels; i++) { _queue.fill(_bitmap.getData(i), static_cast<bitmap_type>(0), _bitmap.getBitmapSize(i)).wait(); }
  }

  const HierarchicBitmapDevice<T, Levels, bitmap_type>& getDeviceFrontier() const { return _bitmap; }

  size_t computeActiveFrontier() const { // TODO: Only works with 2 levels now
    auto e = kernels::computeActiveFrontier(*this, _queue);

    e.wait();

#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "computeActiveFrontier");
#endif
    return _bitmap._offsets_size[0];
  }

  static void swap(FrontierHierarchicBitmap<T>& a, FrontierHierarchicBitmap<T>& b) { std::swap(a._bitmap, b._bitmap); }

private:
  sycl::queue& _queue;                                    ///< The SYCL queue used for memory allocation.
  HierarchicBitmapDevice<T, Levels, bitmap_type> _bitmap; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph