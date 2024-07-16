#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

#include <sygraph/operators/intersection/bitmap_intersect_impl.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {
namespace intersection {

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FrontierType, typename LambdaT>
sygraph::Event execute(GraphT& graph,
                       const sygraph::frontier::Frontier<T, FrontierType>& in1,
                       const sygraph::frontier::Frontier<T, FrontierType>& in2,
                       const sygraph::frontier::Frontier<T, FrontierType>& out,
                       LambdaT&& functor) {
  if constexpr (FrontierType == sygraph::frontier::frontier_type::bitmap) {
    return sygraph::operators::intersection::detail::bitmapExecute(graph, in1, in2, out, std::forward<LambdaT>(functor));
  } else if constexpr (FrontierType == sygraph::frontier::frontier_type::vector) {
    throw std::runtime_error("Frontier type not implemented");
  } else {
    throw std::runtime_error("Frontier type not implemented");
  }
}


} // namespace intersection
} // namespace operators
} // namespace v0
} // namespace sygraph