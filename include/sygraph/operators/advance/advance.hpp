#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/workgroup_mapped.hpp>
#include <sygraph/operators/advance/workitem_mapped.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

template<sygraph::operators::LoadBalancer lb, typename GraphT, typename LambdaT, typename T, typename frontier::FrontierType FrontierType>
sygraph::event vertex(GraphT& graph,
                      sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, FrontierType>& in,
                      sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, FrontierType>& out,
                      LambdaT&& functor) {
  if constexpr (lb == sygraph::operators::LoadBalancer::workitem_mapped) {
    return sygraph::operators::advance::detail::workitem_mapped::vertex(graph, in, out, std::forward<LambdaT>(functor));
  } else if constexpr (lb == sygraph::operators::LoadBalancer::workgroup_mapped) {
    return sygraph::operators::advance::detail::workgroup_mapped::vertex(graph, in, out, std::forward<LambdaT>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

template<sygraph::operators::LoadBalancer lb, typename GraphT, typename LambdaT, typename T, typename frontier::FrontierType FrontierType>
sygraph::event edge(GraphT& graph, sygraph::frontier::Frontier<T>& in, sygraph::frontier::Frontier<T>& out, LambdaT&& functor) {
  if constexpr (lb == sygraph::operators::LoadBalancer::workitem_mapped) {
    return sygraph::operators::advance::detail::workitem_mapped::edge(graph, in, out, std::forward<LambdaT>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph