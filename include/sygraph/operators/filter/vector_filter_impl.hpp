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

template<graph::detail::GraphConcept GraphT, typename T, typename sygraph::frontier::frontier_view FrontierView, typename LambdaT>
sygraph::Event
inplace(GraphT& graph, const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::frontier_type::vector>& frontier, LambdaT&& functor) {
  auto q = graph.getQueue();

  using type_t = T;
  // size_t active_elements_size = frontier.getNumActiveElements();
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  type_t* active_elements;
  if (!frontier.selfAllocated()) { active_elements = memory::detail::memoryAlloc<type_t, memory::space::shared>(active_elements_size, q); }
  frontier.getActiveElements(active_elements, active_elements_size);
  auto out_dev = frontier.getDeviceFrontier();

  sygraph::Event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class inplace_filter_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      if (!functor(element)) { out_dev.remove(element); }
    });
  });

  if (!frontier.selfAllocated()) { sycl::free(active_elements, q); }

  return e;
}

template<typename GraphT, typename T, typename sygraph::frontier::frontier_view FrontierView, typename LambdaT>
sygraph::Event external(GraphT& graph,
                        const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::frontier_type::vector>& in,
                        const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::frontier_type::vector>& out,
                        LambdaT&& functor) {
  auto q = graph.getQueue();
  out.clear();

  using type_t = T;
  // size_t active_elements_size = in.getNumActiveElements();
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  type_t* active_elements;
  if (!in.selfAllocated()) { active_elements = memory::detail::memoryAlloc<type_t, memory::space::shared>(active_elements_size, q); }
  in.getActiveElements(active_elements, active_elements_size);

  sygraph::Event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class external_filter_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      if (functor(element)) { out.insert(element); }
    });
  });

  if (!in.selfAllocated()) { sycl::free(active_elements, q); }

  return e;
}

} // namespace detail
} // namespace filter
} // namespace operators
} // namespace v0
} // namespace sygraph