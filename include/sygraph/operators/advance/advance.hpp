#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/operators/advance/workitem_mapped.hpp>
#include <sygraph/operators/advance/workgroup_mapped.hpp>
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

template <sygraph::operators::LoadBalancer lb,
          typename graph_t,
          typename lambda_t,
          typename T,
          typename frontier::FrontierType FrontierType>
sygraph::event vertex(graph_t& graph,
                      sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, FrontierType>& in,
                      sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, FrontierType>& out,
                      lambda_t&& functor) {
  if constexpr (FrontierType == sygraph::frontier::FrontierType::bitmap) {
    if constexpr (lb == sygraph::operators::LoadBalancer::workitem_mapped) {
      return sygraph::operators::advance::detail::workitem_mapped::bitmap::vertex(graph, in, out, std::forward<lambda_t>(functor));
    } else if constexpr (lb == sygraph::operators::LoadBalancer::workgroup_mapped) {
      return sygraph::operators::advance::detail::workgroup_mapped::bitmap::vertex(graph, in, out, std::forward<lambda_t>(functor));
    } else {
      throw std::runtime_error("Load balancer not implemented");
    }
  } else {
    throw std::runtime_error("Frontier type not implemented");
  }
}

template <sygraph::operators::LoadBalancer lb,
          typename graph_t,
          typename lambda_t,
          typename T,
          typename frontier::FrontierType FrontierType>
sygraph::event edge(graph_t& graph,
                    sygraph::frontier::Frontier<T>& in,
                    sygraph::frontier::Frontier<T>& out,
                    lambda_t&& functor) {
  if constexpr (FrontierType == sygraph::frontier::FrontierType::bitmap) {
    if constexpr (lb == sygraph::operators::LoadBalancer::workitem_mapped) {
      return sygraph::operators::advance::detail::workitem_mapped::bitmap::edge(graph, in, out, std::forward<lambda_t>(functor));
    } else {
      throw std::runtime_error("Load balancer not implemented");
    }
  } else {
    throw std::runtime_error("Frontier type not implemented");
  }
}

} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph