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
sygraph::event vertex(GraphT& graph,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& in,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& out,
                      LambdaT&& functor) {
  sycl::queue& q = graph.getQueue();

  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  T* active_elements;
  if (!in.selfAllocated()) { active_elements = memory::detail::memoryAlloc<T, memory::space::shared>(active_elements_size, q); }
  in.getActiveElements(active_elements, active_elements_size);

  sygraph::event ret{q.submit([&](sycl::handler& cgh) {
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
        auto edge = i.get_index();
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

template<typename GraphT, typename T, typename LambdaT>
sygraph::event edge(GraphT& graph,
                    const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::edge, sygraph::frontier::FrontierType::bitmap>& in,
                    const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::edge, sygraph::frontier::FrontierType::bitmap>& out,
                    LambdaT&& functor) {
  sycl::queue& q = graph.getQueue();

  size_t active_elements_size = in.getNumActiveElements();
  T* active_elements = memory::detail::memoryAlloc<T, memory::space::shared>(active_elements_size, q);
  in.getActiveElements(active_elements);

  sygraph::event ret{q.submit([&](sycl::handler& cgh) {
    auto in_dev_frontier = in.getDeviceFrontier();
    auto out_dev_frontier = out.getDeviceFrontier();
    auto graph_dev = graph.getDeviceGraph();

    cgh.parallel_for<class edge_advance_kernel>(sycl::range<1>(active_elements_size), [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      auto start = graph_dev.begin(element);
      auto end = graph_dev.end(element);

      // each work item takes care of all the neighbors of the vertex he is responsible for
      for (auto i = start; i != end; ++i) {
        auto edge = i.get_index();
        auto weight = graph_dev.getEdgeWeight(element, edge);
        auto neighbor = *i;
        if (functor(element, neighbor, edge, weight)) { out_dev_frontier.insert(edge); }
      }
    });
  })};

  sycl::free(active_elements, q);
  return ret;
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::event vertex(GraphT& graph,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::vector>& in,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::vector>& out,
                      LambdaT&& functor) {
  sycl::queue& q = graph.getQueue();

  size_t active_elements_size = in.getNumActiveElements();
  T* active_elements;
  in.getActiveElements(active_elements, active_elements_size);

  sygraph::event ret{q.submit([&](sycl::handler& cgh) {
    auto in_dev_frontier = in.getDeviceFrontier();
    auto out_dev_frontier = out.getDeviceFrontier();
    auto graph_dev = graph.getDeviceGraph();
    constexpr size_t LOCAL_MEM_SIZE = types::detail::MAX_LOCAL_MEM_SIZE;

    sycl::range<1> local_size(128); // TODO: tune this value
    sycl::range<1> global_size(active_elements_size <= local_size[0] ? local_size[0]
                                                                     : (active_elements_size + local_size[0] - active_elements_size % local_size[0]));

    sycl::local_accessor<T, 1> l_frontier(LOCAL_MEM_SIZE, cgh);
    sycl::local_accessor<size_t, 1> l_frontier_tail(1, cgh);

    cgh.parallel_for<class vertex_local_mem_advance_kernel>(sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
      size_t gid = item.get_global_linear_id();
      size_t lid = item.get_local_linear_id();
      sycl::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::work_group> l_frontier_tail_ref(l_frontier_tail[0]);

      if (lid == 0) { l_frontier_tail_ref = 0; }
      sycl::group_barrier(item.get_group());

      if (gid < active_elements_size) {
        auto element = active_elements[gid];
        auto start = graph_dev.begin(element);
        auto end = graph_dev.end(element);

        // each work item takes care of all the neighbors of the vertex he is responsible for
        for (auto i = start; i != end; ++i) {
          auto edge = i.get_index();
          auto weight = graph_dev.getEdgeWeight(edge);
          auto neighbor = *i;
          if (functor(element, neighbor, edge, weight)) {
            if (l_frontier_tail_ref < LOCAL_MEM_SIZE) { // if the local memory is not full, we can use it
              l_frontier[l_frontier_tail_ref++] = neighbor;
            } else { // if the local memory is full, we need to use the global mem
              out_dev_frontier.insert(neighbor);
            }
          }
        }
      }
      sycl::group_barrier(item.get_group());
      size_t address_space = 0;
      if (lid == 0) { address_space = out_dev_frontier.prealloc(l_frontier_tail_ref.load()); }
      address_space = sycl::group_broadcast(item.get_group(), address_space, 0);
      for (size_t i = lid; i < l_frontier_tail_ref; i += item.get_local_range(0)) { out_dev_frontier.insert(l_frontier[i], address_space + i); }
    });
  })};

  return ret;
}


} // namespace workitem_mapped
} // namespace detail
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph