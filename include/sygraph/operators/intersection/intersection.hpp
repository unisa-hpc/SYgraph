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

template<graph::detail::GraphConcept GraphT,
         typename T,
         typename sygraph::frontier::FrontierView FrontierView,
         typename sygraph::frontier::FrontierType FrontierType,
         typename LambdaT>
sygraph::event execute(GraphT& graph,
                       const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& in1,
                       const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& in2,
                       const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& out,
                       LambdaT&& functor) {
  if constexpr (FrontierType == sygraph::frontier::FrontierType::bitmap) {
    return sygraph::operators::intersection::detail::bitmapExecute(graph, in1, in2, out, std::forward<LambdaT>(functor));
  } else if constexpr (FrontierType == sygraph::frontier::FrontierType::vector) {
    throw std::runtime_error("Frontier type not implemented");
  } else {
    throw std::runtime_error("Frontier type not implemented");
  }
}


} // namespace intersection
} // namespace operators
} // namespace v0
} // namespace sygraph