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

template<sygraph::operators::load_balancer Lb, typename GraphT, typename LambdaT, typename T, typename frontier::frontier_type FrontierType>
sygraph::Event vertex(GraphT& graph,
                      sygraph::frontier::Frontier<T, sygraph::frontier::frontier_view::vertex, FrontierType>& in,
                      sygraph::frontier::Frontier<T, sygraph::frontier::frontier_view::vertex, FrontierType>& out,
                      LambdaT&& functor) {
  if constexpr (Lb == sygraph::operators::load_balancer::workitem_mapped) {
    return sygraph::operators::advance::detail::workitem_mapped::vertex(graph, in, out, std::forward<LambdaT>(functor));
  } else if constexpr (Lb == sygraph::operators::load_balancer::workgroup_mapped) {
    return sygraph::operators::advance::detail::workgroup_mapped::vertex(graph, in, out, std::forward<LambdaT>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

template<sygraph::operators::load_balancer Lb, typename GraphT, typename LambdaT, typename T, typename frontier::frontier_type FrontierType>
sygraph::Event edge(GraphT& graph, sygraph::frontier::Frontier<T>& in, sygraph::frontier::Frontier<T>& out, LambdaT&& functor) {
  if constexpr (Lb == sygraph::operators::load_balancer::workitem_mapped) {
    return sygraph::operators::advance::detail::workitem_mapped::edge(graph, in, out, std::forward<LambdaT>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph