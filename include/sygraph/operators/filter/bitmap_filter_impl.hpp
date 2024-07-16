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
inline namespace v0 {
namespace operators {

namespace filter {
namespace detail {

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FT, typename LambdaT>
sygraph::Event
launchBitmapKernelExternal(GraphT& graph, const sygraph::frontier::Frontier<T, FT>& in, sygraph::frontier::Frontier<T, FT>& out, LambdaT&& functor) {
  if constexpr (FT != sygraph::frontier::frontier_type::bitmap && FT != sygraph::frontier::frontier_type::bitvec
                && FT != sygraph::frontier::frontier_type::hierachic_bitmap) {
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

    cgh.parallel_for(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = in_dev.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && in_dev.check(actual_id) && functor(actual_id)) { out_dev.insert(actual_id); }
    });

    // cgh.parallel_for<class external_filter_kernel>(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) { // TODO: check if it works
    //   type_t element = idx[0];
    //   if (in_dev.check(element) && functor(element)) { out_dev.insert(element); }
    // });
  });

  return e;
}

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FT, typename LambdaT>
sygraph::Event launchBitmapKernelInplace(GraphT& graph, const sygraph::frontier::Frontier<T, FT>& frontier, LambdaT&& functor) {
  if constexpr (FT != sygraph::frontier::frontier_type::bitmap && FT != sygraph::frontier::frontier_type::bitvec
                && FT != sygraph::frontier::frontier_type::hierachic_bitmap) {
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

    cgh.parallel_for(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = dev_frontier.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && dev_frontier.check(actual_id) && functor(actual_id)) { dev_frontier.remove(actual_id); }
    });

    // cgh.parallel_for<class inplace_filter_kernel>(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) { // TODO: check if it works
    //   type_t element = idx[0];
    //   if (out_dev.check(element) && !functor(element)) { out_dev.remove(element); }
    // });
  });

  return e;
}


template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event inplace(GraphT& graph, const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& frontier, LambdaT&& functor) {
  return launchBitmapKernelInplace(graph, frontier, functor);
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event external(GraphT& graph,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& in,
                        sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& out,
                        LambdaT&& functor) {
  return launchBitmapKernelExternal(graph, in, out, functor);
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event inplace(GraphT& graph, const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitvec>& frontier, LambdaT&& functor) {
  return launchBitmapKernelInplace(graph, frontier, functor);
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event external(GraphT& graph,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitvec>& in,
                        sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitvec>& out,
                        LambdaT&& functor) {
  return launchBitmapKernelExternal(graph, in, out, functor);
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event
inplace(GraphT& graph, const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::hierachic_bitmap>& frontier, LambdaT&& functor) {
  return launchBitmapKernelInplace(graph, frontier, functor);
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event external(GraphT& graph,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::hierachic_bitmap>& in,
                        sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::hierachic_bitmap>& out,
                        LambdaT&& functor) {
  return launchBitmapKernelExternal(graph, in, out, functor);
}

} // namespace detail
} // namespace filter
} // namespace operators
} // namespace v0
} // namespace sygraph