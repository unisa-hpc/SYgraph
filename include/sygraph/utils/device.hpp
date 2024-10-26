#pragma once

#include <sycl/sycl.hpp>
#include <sygraph/sycl/event.hpp>
#include <unordered_map>
#include <vector>

namespace sygraph {
inline namespace v0 {
namespace details {
namespace device {

uint32_t getMaxNumSubgroups(sycl::queue& queue) {
  auto device = queue.get_device();
  return device.get_info<sycl::info::device::max_num_sub_groups>();
};


} // namespace device
} // namespace details
} // namespace v0
} // namespace sygraph