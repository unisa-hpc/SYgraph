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

#include <sygraph/operators/for/impl_for.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace compute {


inline namespace v1 {
template<typename graph_t,
         typename T,
         typename sygraph::frontier::FrontierView FrontierView,
         typename sygraph::frontier::FrontierType FrontierType,
         typename lambda_t>
sygraph::event execute(graph_t& graph, const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& frontier, lambda_t&& functor) {
  return sygraph::operators::compute::detail::execute(graph, frontier, std::forward<lambda_t>(functor));
}
} // namespace v1

namespace v0 {
template<typename graph_t, typename frontier_t, typename lambda_t>
sygraph::event execute(graph_t& graph, frontier_t& frontier, lambda_t&& functor) {
  auto q = graph.getQueue();

  using type_t = typename frontier_t::type_t;
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  type_t* active_elements;
  if (!frontier.selfAllocated()) { active_elements = memory::detail::memoryAlloc<type_t, memory::space::shared>(active_elements_size, q); }
  frontier.getActiveElements(active_elements, active_elements_size);

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class for_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      functor(element);
    });
  });

  if (!frontier.selfAllocated()) { sycl::free(active_elements, q); }

  return e;
}
} // namespace v0

} // namespace compute
} // namespace operators
} // namespace v0
} // namespace sygraph