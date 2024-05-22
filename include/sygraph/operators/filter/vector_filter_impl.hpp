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

namespace filter {
namespace detail {

template <typename graph_t,
          typename T,
          typename sygraph::frontier::FrontierView FrontierView,
          typename lambda_t>
sygraph::event vector_inplace(graph_t& graph, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::vector>& frontier, 
                              lambda_t&& functor) {
  auto q = graph.get_queue();

  using type_t = T;
  // size_t active_elements_size = frontier.get_num_active_elements();
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  type_t* active_elements;
  if (!frontier.self_allocated()) {
    active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  }
  frontier.get_active_elements(active_elements, active_elements_size);
  auto outDev = frontier.get_device_frontier();

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class inplace_filter_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      if (!functor(element)) {
        outDev.remove(element);
      }
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
sygraph::event vector_external(graph_t& graph, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::vector>& in, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::vector>& out, 
                              lambda_t&& functor) {
  auto q = graph.get_queue();
  out.clear();

  using type_t = T;
  // size_t active_elements_size = in.get_num_active_elements();
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  type_t* active_elements;
  if (!in.self_allocated()) {
    active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  }
  in.get_active_elements(active_elements, active_elements_size);

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class external_filter_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      if (functor(element)) {
        out.insert(element);
      }
    });
  });

  if (!in.self_allocated()) {
    sycl::free(active_elements, q);
  }

  return e;
}

} // namespace detail
} // namespace filter
} // namespace operators
} // namespace v0
} // namespace sygraph