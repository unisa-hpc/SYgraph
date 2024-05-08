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

namespace parallel_for {

template <typename graph_t,
          typename frontier_t,
          typename lambda_t>
sygraph::event execute(graph_t& graph, frontier_t& frontier, lambda_t&& functor) {
  auto q = graph.get_queue();

  using type_t = typename frontier_t::type_t;
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  // size_t active_elements_size = frontier.get_num_active_elements();
  type_t* active_elements;
  if (!frontier.self_allocated()) {
    active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
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

} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph