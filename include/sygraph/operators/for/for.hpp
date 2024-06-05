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
template<graph::detail::GraphConcept GraphT,
         typename T,
         typename sygraph::frontier::FrontierView FrontierView,
         typename sygraph::frontier::FrontierType FrontierType,
         typename LambdaT>
sygraph::event execute(GraphT& graph, const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& frontier, LambdaT&& functor) {
  return sygraph::operators::compute::detail::execute(graph, frontier, std::forward<LambdaT>(functor));
}
} // namespace v1

namespace v0 {
template<typename GraphT, typename frontier_t, typename LambdaT>
sygraph::event execute(GraphT& graph, frontier_t& frontier, LambdaT&& functor) {
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