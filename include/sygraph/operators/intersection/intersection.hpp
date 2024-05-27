#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

#include <sygraph/operators/intersection/bitmap_intersect_impl.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {
namespace intersection {

template <typename graph_t,
          typename T,
          typename sygraph::frontier::FrontierView FrontierView,
          typename sygraph::frontier::FrontierType FrontierType,
          typename lambda_t>
sygraph::event execute(graph_t& graph, 
                       const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& in1, 
                       const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& in2, 
                       const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& out, 
                       lambda_t&& functor) {
  if constexpr (FrontierType == sygraph::frontier::FrontierType::bitmap) {
    return sygraph::operators::intersection::detail::bitmap_execute(graph, in1, in2, out, std::forward<lambda_t>(functor));
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