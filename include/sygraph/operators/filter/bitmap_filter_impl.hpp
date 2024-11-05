#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/workitem_mapped.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
namespace operators {

namespace filter {
namespace detail {

class inplace_filter_kernel;
class external_filter_kernel;

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FT, typename LambdaT>
sygraph::Event
launchBitmapKernelExternal(GraphT& graph, const sygraph::frontier::Frontier<T, FT>& in, sygraph::frontier::Frontier<T, FT>& out, LambdaT&& functor) {
  if constexpr (FT != sygraph::frontier::frontier_type::bitmap && FT != sygraph::frontier::frontier_type::mlb) {
    throw std::runtime_error("Invalid frontier type");
  }

  auto q = graph.getQueue();

  size_t num_nodes = graph.getVertexCount();

  size_t bitmap_range = in.getBitmapRange();
  size_t offsets_size = in.computeActiveFrontier();

  out.clear();

  using type_t = T;

  auto out_dev = out.getDeviceFrontier();
  auto in_dev = in.getDeviceFrontier();

  sygraph::Event e = q.submit([&](sycl::handler& cgh) {
    sycl::range<1> local_range{bitmap_range};
    size_t global_size = offsets_size * local_range[0];
    sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

    cgh.parallel_for<external_filter_kernel>(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = in_dev.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && in_dev.check(actual_id) && functor(actual_id)) { out_dev.insert(actual_id); }
    });
  });

  return e;
}

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FT, typename LambdaT>
sygraph::Event launchBitmapKernelInplace(GraphT& graph, const sygraph::frontier::Frontier<T, FT>& frontier, LambdaT&& functor) {
  if constexpr (FT != sygraph::frontier::frontier_type::bitmap && FT != sygraph::frontier::frontier_type::mlb) {
    throw std::runtime_error("Invalid frontier type");
  }

  auto q = graph.getQueue();
  auto dev_frontier = frontier.getDeviceFrontier();

  size_t num_nodes = graph.getVertexCount();

  size_t bitmap_range = frontier.getBitmapRange();
  size_t offsets_size = frontier.computeActiveFrontier();

  using type_t = T;

  sygraph::Event e = q.submit([&](sycl::handler& cgh) {
    sycl::range<1> local_range{bitmap_range};
    size_t global_size = offsets_size * local_range[0];
    sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

    cgh.parallel_for<inplace_filter_kernel>(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = dev_frontier.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && dev_frontier.check(actual_id) && functor(actual_id)) { dev_frontier.remove(actual_id); }
    });
  });

  return e;
}

} // namespace detail
} // namespace filter
} // namespace operators
} // namespace sygraph