#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

namespace detail {

namespace workitem_mapped {


template <typename graph_t,
          typename T,
          typename lambda_t>
sygraph::event vertex(graph_t& graph,   
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& in, 
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& out, 
                      lambda_t&& functor) {
  
  sycl::queue& q = graph.get_queue();
  
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  T* active_elements;
  if (!in.self_allocated()) {
    active_elements = memory::detail::memory_alloc<T, memory::space::shared>(active_elements_size, q);
  }
  in.get_active_elements(active_elements, active_elements_size);

  sygraph::event ret {q.submit([&](sycl::handler& cgh) {
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();
    sycl::stream os(1024, 256, cgh);

    cgh.parallel_for<class vertex_workitem_advance_kernel>(sycl::range<1>(active_elements_size), [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      auto start = graphDev.begin(element);
      auto end = graphDev.end(element);

      // each work item takes care of all the neighbors of the vertex he is responsible for
      for (auto i = start; i != end; ++i) {
        auto edge = i.get_index();
        auto weight = graphDev.get_edge_weight(edge);
        auto neighbor = *i;
        if (functor(element, neighbor, edge, weight)) {
          bool val = outDevFrontier.insert(neighbor);
          if (!val) {
            os << "Error inserting " << neighbor << sycl::endl;
          }
        }
      }
    });
  })};

  if (!in.self_allocated()) {
    sycl::free(active_elements, q);
  }
  return ret;
}

template <typename graph_t,
          typename T,
          typename lambda_t>
sygraph::event edge(graph_t& graph, 
                    const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::edge, sygraph::frontier::FrontierType::bitmap>& in, 
                    const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::edge, sygraph::frontier::FrontierType::bitmap>& out, 
                    lambda_t&& functor) {
  
  sycl::queue& q = graph.get_queue();

  size_t active_elements_size = in.get_num_active_elements();
  T* active_elements = memory::detail::memory_alloc<T, memory::space::shared>(active_elements_size, q);
  in.get_active_elements(active_elements);

  sygraph::event ret {q.submit([&](sycl::handler& cgh) {
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    cgh.parallel_for<class edge_advance_kernel>(sycl::range<1>(active_elements_size), [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      auto start = graphDev.begin(element);
      auto end = graphDev.end(element);

      // each work item takes care of all the neighbors of the vertex he is responsible for
      for (auto i = start; i != end; ++i) {
        auto edge = i.get_index();
        auto weight = graphDev.get_edge_weight(element, edge);
        auto neighbor = *i;
        if (functor(element, neighbor, edge, weight)) {
          outDevFrontier.insert(edge);
        }
      }
    });
  })};

  sycl::free(active_elements, q);
  return ret;
}

template <typename graph_t,
          typename T,
          typename lambda_t>
sygraph::event vertex(graph_t& graph,   
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::vector>& in, 
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::vector>& out, 
                      lambda_t&& functor) {  
  sycl::queue& q = graph.get_queue();

  size_t active_elements_size = in.get_num_active_elements();
  T* active_elements;
  in.get_active_elements(active_elements, active_elements_size);

  sygraph::event ret {q.submit([&](sycl::handler& cgh) {
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();
    constexpr size_t LOCAL_MEM_SIZE = types::detail::MAX_LOCAL_MEM_SIZE;

    sycl::range<1> local_size(128); // TODO: tune this value
    sycl::range<1> global_size(active_elements_size <= local_size[0] ? local_size[0] : (active_elements_size + local_size[0] - active_elements_size % local_size[0]));

    sycl::local_accessor<T, 1> l_frontier(LOCAL_MEM_SIZE, cgh);
    sycl::local_accessor<size_t, 1> l_frontier_tail(1, cgh);

    cgh.parallel_for<class vertex_local_mem_advance_kernel>(sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
      size_t gid = item.get_global_linear_id();
      size_t lid = item.get_local_linear_id();
      sycl::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::work_group> l_frontier_tail_ref(l_frontier_tail[0]);

      if (lid == 0) {
        l_frontier_tail_ref = 0;
      }
      sycl::group_barrier(item.get_group());

      if (gid < active_elements_size) {
        auto element = active_elements[gid];
        auto start = graphDev.begin(element);
        auto end = graphDev.end(element);

        // each work item takes care of all the neighbors of the vertex he is responsible for
        for (auto i = start; i != end; ++i) {
          auto edge = i.get_index();
          auto weight = graphDev.get_edge_weight(edge);
          auto neighbor = *i;
          if (functor(element, neighbor, edge, weight)) {
            if (l_frontier_tail_ref < LOCAL_MEM_SIZE) { // if the local memory is not full, we can use it
              l_frontier[l_frontier_tail_ref++] = neighbor;
            } else { // if the local memory is full, we need to use the global mem
              outDevFrontier.insert(neighbor);
            }
          }
        }
      }
      sycl::group_barrier(item.get_group());
      size_t address_space = 0;
      if (lid == 0) {
        address_space = outDevFrontier.prealloc(l_frontier_tail_ref.load());
      }
      address_space = sycl::group_broadcast(item.get_group(), address_space, 0);
      for (size_t i = lid; i < l_frontier_tail_ref; i += item.get_local_range(0)) {
        outDevFrontier.insert(l_frontier[i], address_space + i);
      }
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