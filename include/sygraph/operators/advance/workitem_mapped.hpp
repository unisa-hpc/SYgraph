#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

namespace detail {

namespace workitem_mapped {


template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event frontier(GraphT& graph,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& in,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& out,
                        LambdaT&& functor) {
  sycl::queue& q = graph.getQueue();

  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE; // TODO change this
  T* active_elements;
  if (!in.selfAllocated()) { active_elements = memory::detail::memoryAlloc<T, memory::space::shared>(active_elements_size, q); }
  in.getActiveElements(active_elements, active_elements_size);

  sygraph::Event ret{q.submit([&](sycl::handler& cgh) {
    auto in_dev_frontier = in.getDeviceFrontier();
    auto out_dev_frontier = out.getDeviceFrontier();
    auto graph_dev = graph.getDeviceGraph();
    sycl::stream os(1024, 256, cgh);

    cgh.parallel_for<class vertex_workitem_advance_kernel>(sycl::range<1>(active_elements_size), [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      auto start = graph_dev.begin(element);
      auto end = graph_dev.end(element);

      // each work item takes care of all the neighbors of the vertex he is responsible for
      for (auto i = start; i != end; ++i) {
        auto edge = i.getIndex();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *i;
        if (functor(element, neighbor, edge, weight)) {
          bool val = out_dev_frontier.insert(neighbor);
          if (!val) { os << "Error inserting " << neighbor << sycl::endl; }
        }
      }
    });
  })};

  if (!in.selfAllocated()) { sycl::free(active_elements, q); }
  return ret;
}

// TODO implement also the graph version

} // namespace workitem_mapped
} // namespace detail
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph