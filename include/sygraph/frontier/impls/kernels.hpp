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
namespace kernels {


template<typename Frontier>
concept FrontierConcept = requires(Frontier frontier) {
  { frontier.getDeviceFrontier() } -> std::same_as<const typename Frontier::device_frontier_type&>;
};

template<FrontierConcept Frontier>
Event computeActiveFrontier(const Frontier& frontier, sycl::queue& queue) {
  sycl::range<1> local_range{128};
  auto bitmap = frontier.getDeviceFrontier();
  using bitmap_type = typename decltype(bitmap)::bitmap_type;
  size_t size = bitmap.getBitmapSize(1);
  uint32_t range = bitmap.getBitmapRange();
  sycl::range<1> global_range{(size > local_range[0] ? size + local_range[0] - (size % local_range[0]) : local_range[0])};

  return queue.submit([&](sycl::handler& cgh) {
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
                       for (size_t i = lid; i < local_size_ref.load(); i += item.get_local_range(0)) { offsets[data_offset + i] = local_offsets[i]; }
                     });
  });
}

} // namespace kernels
} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph