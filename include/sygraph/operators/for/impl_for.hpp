#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/operators/advance/workitem_mapped.hpp>
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace compute {

namespace detail {

template <typename graph_t,
          typename T,
          typename sygraph::frontier::FrontierView FrontierView,
          typename lambda_t>
sygraph::event execute(graph_t& graph, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitmap>& frontier, 
                              lambda_t&& functor) {
  auto q = graph.get_queue();

  size_t num_nodes = graph.get_vertex_count();
  auto devFrontier = frontier.get_device_frontier();

  size_t local_size = frontier.get_bitmap_range();
  size_t global_size = frontier.compute_offsets();

  size_t bitmap_range = frontier.get_bitmap_range();
  size_t offsets_size = frontier.compute_offsets();

  sygraph::event e = q.submit([&] (sycl::handler& cgh) {
    sycl::range<1> local_range{bitmap_range};
    size_t global_size = offsets_size * local_range[0];
    sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

    cgh.parallel_for<class for_kernel>(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = devFrontier.get_offsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;
      
      if (actual_id < num_nodes && devFrontier.check(actual_id)) {
        functor(actual_id);
      }
    });
  });
  // sygraph::event e = q.submit([&](sycl::handler& cgh) {
  //   cgh.parallel_for<class for_kernel>(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
  //     if (devFrontier.check(idx[0])) {
  //       functor(idx[0]);
  //     }
  //   });
  // });

  return e;
}

template <typename graph_t,
          typename T,
          typename sygraph::frontier::FrontierView FrontierView,
          typename lambda_t>
sygraph::event execute(graph_t& graph, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::vector>& frontier, 
                              lambda_t&& functor) {
  auto q = graph.get_queue();

  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  T* active_elements;
  if (!frontier.self_allocated()) {
    active_elements = sycl::malloc_shared<T>(active_elements_size, q);
  }
  frontier.get_active_elements(active_elements, active_elements_size);

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class for_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      functor(element);
    });
  });

  if (!frontier.self_allocated()) {
    sycl::free(active_elements, q);
  }

  return e;
}

template <typename graph_t,
          typename T,
          typename sygraph::frontier::FrontierView FrontierView,
          typename lambda_t>
sygraph::event execute(graph_t& graph, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitvec>& frontier, 
                              lambda_t&& functor) {
  auto q = graph.get_queue();
  auto devFrontier = frontier.get_device_frontier();

  sygraph::event e;
  if (devFrontier.use_vector()) {
    T* active_elements = devFrontier.get_vector();
    size_t size = devFrontier.get_vector_size();

    e = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::id<1> idx) {
        auto element = active_elements[idx];
        functor(element);
      });
    });
  } else {
    size_t num_nodes = graph.get_vertex_count();

    e = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
        if (devFrontier.check(idx[0])) {
          functor(idx[0]);
        }
      });
    });
  }

  return e;
}

} // namespace detail
} // namespace compute
} // namespace operators
} // namespace v0
} // namespace sygraph