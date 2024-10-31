#pragma once

#include <sycl/sycl.hpp>
#include <sygraph/sycl/event.hpp>
#include <unordered_map>
#include <vector>

namespace sygraph {
inline namespace v0 {
namespace detail {
namespace device {

uint32_t getMaxNumSubgroups(sycl::queue& queue) {
  auto device = queue.get_device();
  return device.get_info<sycl::info::device::max_num_sub_groups>();
}

uint32_t getSubgroupSize(sycl::queue& queue) {
  auto device = queue.get_device();
  return device.get_info<sycl::info::device::sub_group_sizes>()[0];
}

uint32_t getWorkgroupSize(sycl::queue& queue) {
  auto device = queue.get_device();
  return device.get_info<sycl::info::device::max_work_group_size>();
}

} // namespace device
} // namespace detail
} // namespace v0
} // namespace sygraph