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
template<size_t Levels, typename B = types::bitmap_type_t>
class BitmapState {
public:
  std::array<size_t, Levels> size;
  std::array<std::vector<B>, Levels> data;
};

namespace detail {


template<typename DeviceFrontier>
concept DeviceFrontierConcept = requires(DeviceFrontier f) {
  { f.getBitmapSize() } -> std::convertible_to<size_t>;
  { f.getNumElems() } -> std::convertible_to<size_t>;
  { f.getBitmapRange() } -> std::convertible_to<size_t>;
  { f.getData() } -> std::convertible_to<typename DeviceFrontier::bitmap_type*>;
  { f.set(0, true) } -> std::convertible_to<void>;
  { f.insert(0) } -> std::convertible_to<bool>;
  { f.remove(0) } -> std::convertible_to<bool>;
  { f.reset() } -> std::convertible_to<void>;
  { f.reset(0) } -> std::convertible_to<void>;
  { f.check(0) } -> std::convertible_to<bool>;
  { f.empty() } -> std::convertible_to<bool>;
  { f.getBitmapIndex(0) } -> std::convertible_to<size_t>;
  { f.getOffsets() } -> std::convertible_to<int*>;
  { f.getOffsetsSize() } -> std::convertible_to<uint32_t*>;
};

template<typename T, size_t Levels, DeviceFrontierConcept DeviceFrontier>
class FrontierMLB;

template<typename T, size_t Levels, typename B = types::bitmap_type_t>
class MLBDevice {
public:
  using bitmap_type = B;

  MLBDevice(size_t num_elems) : _num_elems(num_elems) {
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

  SYCL_EXTERNAL inline bool empty() const {
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

  void setData(bitmap_type* data[Levels]) {
    for (size_t i = 0; i < Levels; i++) { this->_data[i] = data[i]; }
  }

  void setOffsets(int* offsets) { this->_offsets = offsets; }

  void setOffsetsSize(uint32_t* offsets_size) { this->_offsets_size = offsets_size; }

protected:
  uint _range;                ///< The range of the bitmap.
  size_t _num_elems;          ///< The number of elements in the bitmap.
  size_t _size[Levels];       ///< The size of the bitmap.
  bitmap_type* _data[Levels]; ///< Pointer to the bitmap.

  int* _offsets;
  uint32_t* _offsets_size;
};

template<typename T, size_t Levels = 2, DeviceFrontierConcept DeviceFrontier = MLBDevice<T, Levels>>
class FrontierMLB {
public:
  using bitmap_type = typename DeviceFrontier::bitmap_type;
  using device_frontier_type = DeviceFrontier;
  using frontier_state_type = BitmapState<Levels, bitmap_type>;

  FrontierMLB(sycl::queue& q, size_t num_elems) : _queue(q), _bitmap(num_elems) {
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

    _bitmap.setData(ptr);
    _bitmap.setOffsets(offsets);
    _bitmap.setOffsetsSize(offsets_size);
  }

  ~FrontierMLB() {
    for (size_t i = 0; i < Levels; i++) { sycl::free(_bitmap.getData(i), _queue); }
    sycl::free(_bitmap.getOffsets(), _queue);
    sycl::free(_bitmap.getOffsetsSize(), _queue);
  }

  size_t getBitmapSize() const { return _bitmap.getBitmapSize(); }

  size_t getNumElems() const { return _bitmap.getNumElems(); }

  size_t getBitmapRange() const { return _bitmap.getBitmapRange(); }

  bool selfAllocated() const { return false; }

  bool empty() const { return _bitmap.empty(); }

  bool check(size_t idx) const { return _bitmap.check(idx); }

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

  inline const size_t size() const {
    sycl::buffer<size_t, 1> size_buf(sycl::range<1>(1));
    size_t frontier_size = this->getBitmapSize();
    size_t bitmap_range = this->getBitmapRange();
    _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto size_acc = size_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>{frontier_size}, [=](sycl::id<1> idx) {
        if (idx == 0) { size_acc[0] = 0; }
        sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(size_acc[0]);
        size_t num_active_nodes = 0;
        bitmap_type t = bitmap.getData()[idx];
        for (size_t i = 0; i < bitmap_range; i++) {
          if (t & (static_cast<bitmap_type>(1) << i)) { num_active_nodes++; }
        }
        ref += num_active_nodes;
      });
    });
    _queue.wait();
    sycl::host_accessor<size_t, 1> size_acc(size_buf);
    return size_acc[0];
  }

  // operator =
  FrontierMLB& operator=(const FrontierMLB& other) {
    if (this == &other) { return *this; }
    _queue.copy(other._bitmap.getData(), this->_bitmap.getData(), _bitmap.getBitmapSize()).wait();
    return *this;
  }

  sygraph::Event merge(FrontierMLB<T>& other) {
    return _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<class merge_bitmap_frontier_kernel>(sycl::range<1>(bitmap.getBitmapSize()),
                                                           [=](sycl::id<1> idx) { bitmap.getData()[idx] |= other_bitmap.getData()[idx]; });
    });
  }

  sygraph::Event intersect(FrontierMLB<T>& other) {
    return _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<class intersect_bitmap_frontier_kernel>(sycl::range<1>(bitmap.getBitmapSize()),
                                                               [=](sycl::id<1> idx) { bitmap.getData()[idx] &= other_bitmap.getData()[idx]; });
    });
  }

  sygraph::frontier::BitmapState<Levels, bitmap_type> saveState() {
    sygraph::frontier::BitmapState<Levels, bitmap_type> state;
    for (size_t i = 0; i < Levels; i++) {
      state.size[i] = _bitmap.getBitmapSize(i);
      state.data[i].resize(state.size[i]);
      _queue.copy(_bitmap.getData(i), state.data[i].data(), state.size[i]);
    }
    return state;
  }

  void loadState(const sygraph::frontier::BitmapState<Levels, bitmap_type>& state) {
    for (size_t i = 0; i < Levels; i++) {
      assert(state.size[i] == _bitmap.getBitmapSize(i));
      _queue.copy(state.data[i].data(), _bitmap.getData(i), state.size[i]);
    }
  }

  void clear() {
    for (size_t i = 0; i < Levels; i++) { _queue.fill(_bitmap.getData(i), static_cast<bitmap_type>(0), _bitmap.getBitmapSize(i)); }
    _queue.fill(_bitmap.getOffsetsSize(), static_cast<uint32_t>(0), 1);
    _queue.wait();
  }

  const DeviceFrontier& getDeviceFrontier() const { return _bitmap; }

  size_t computeActiveFrontier() const {
    if constexpr (Levels != 2) { throw std::runtime_error("Only 2 levels are supported"); }

    sycl::range<1> local_range{128};
    auto bitmap = this->getDeviceFrontier();
    size_t size = bitmap.getBitmapSize(1);
    uint32_t range = bitmap.getBitmapRange();
    sycl::range<1> global_range{(size > local_range[0] ? size + local_range[0] - (size % local_range[0]) : local_range[0])};

    size_t size_offsets = bitmap.getOffsetsSize()[0];
    if (size_offsets > 0) { return size_offsets; }

    auto e = this->_queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<int, 1> local_offsets(local_range[0] * range, cgh);
      sycl::local_accessor<uint32_t, 1> local_size(1, cgh);
      bitmap.getOffsetsSize()[0] = 0;


      cgh.parallel_for(sycl::nd_range<1>{global_range, local_range},
                       [=, offsets_size = bitmap.getOffsetsSize(), offsets = bitmap.getOffsets()](sycl::nd_item<1> item) {
                         int gid = item.get_global_linear_id();
                         size_t lid = item.get_local_linear_id();
                         auto group = item.get_group();

                         sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> local_size_ref(local_size[0]);
                         sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> offsets_size_ref{offsets_size[0]};

                         if (group.leader()) { local_size_ref.store(0); }
                         sycl::group_barrier(group);

                         if (gid < size) {
                           bitmap_type data = bitmap.getData(1)[gid];
                           for (size_t i = 0; i < range; i++) {
                             if (data & (static_cast<bitmap_type>(1) << i)) { local_offsets[local_size_ref++] = i + gid * range; }
                           }
                         }

                         sycl::group_barrier(group);

                         size_t data_offset = 0;
                         if (group.leader()) { data_offset = offsets_size_ref.fetch_add(local_size_ref.load()); }
                         data_offset = sycl::group_broadcast(group, data_offset, 0);
                         for (size_t i = lid; i < local_size_ref.load(); i += item.get_local_range(0)) {
                           offsets[data_offset + i] = local_offsets[i];
                         }
                       });
    });

    e.wait();

#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "computeActiveFrontier");
#endif
    return _bitmap.getOffsetsSize()[0];
  }

  static void swap(FrontierMLB<T>& a, FrontierMLB<T>& b) { std::swap(a._bitmap, b._bitmap); }

protected:
  sycl::queue& _queue;    ///< The SYCL queue used for memory allocation.
  DeviceFrontier _bitmap; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph