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

template<sygraph::operators::load_balancer Lb,
         frontier::frontier_view FW,
         typename GraphT,
         typename LambdaT,
         typename T,
         frontier::frontier_type FrontierType>
sygraph::Event vertices(GraphT& graph, sygraph::frontier::Frontier<T, FrontierType>& out, LambdaT&& functor) {
  auto in = sygraph::frontier::Frontier<bool, sygraph::frontier::frontier_type::none>{};
  if constexpr (Lb == sygraph::operators::load_balancer::workgroup_mapped) {
    return sygraph::operators::advance::detail::workgroup_mapped::launchBitmapKernel<sygraph::frontier::frontier_view::graph, FW, T>(
        graph, in, out, std::forward<LambdaT>(functor));
    // return sygraph::operators::advance::detail::workgroup_mapped::vertices<FW>(graph, out, std::forward<LambdaT>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

template<sygraph::operators::load_balancer Lb, typename GraphT, typename LambdaT>
sygraph::Event vertices(GraphT& graph, LambdaT&& functor) {
  auto out = sygraph::frontier::Frontier<bool, sygraph::frontier::frontier_type::none>{};
  return vertices<Lb, sygraph::frontier::frontier_view::none>(graph, out, std::forward<LambdaT>(functor));
}

template<sygraph::operators::load_balancer Lb,
         frontier::frontier_view InView,
         frontier::frontier_view OutView,
         typename GraphT,
         typename LambdaT,
         typename T,
         frontier::frontier_type FrontierType>
sygraph::Event
frontier(GraphT& graph, sygraph::frontier::Frontier<T, FrontierType>& in, sygraph::frontier::Frontier<T, FrontierType>& out, LambdaT&& functor) {
  if constexpr (Lb == sygraph::operators::load_balancer::workitem_mapped) {
    return sygraph::operators::advance::detail::workitem_mapped::frontier<InView, OutView>(graph, in, out, std::forward<LambdaT>(functor));
  } else if constexpr (Lb == sygraph::operators::load_balancer::workgroup_mapped) {
    return sygraph::operators::advance::detail::workgroup_mapped::launchBitmapKernel<InView, OutView, T>(
        graph, in, out, std::forward<LambdaT>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

template<sygraph::operators::load_balancer Lb,
         frontier::frontier_view InView,
         typename GraphT,
         typename LambdaT,
         typename T,
         frontier::frontier_type FrontierType>
sygraph::Event frontier(GraphT& graph, sygraph::frontier::Frontier<T, FrontierType>& in, LambdaT&& functor) {
  auto out = sygraph::frontier::Frontier<bool, sygraph::frontier::frontier_type::none>{};
  return frontier<Lb, InView, sygraph::frontier::frontier_view::none>(graph, in, out, std::forward<LambdaT>(functor));
}

} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph