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

#include <sygraph/operators/filter/bitmap_filter_impl.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace filter {

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FrontierType, typename LambdaT>
sygraph::Event inplace(GraphT& graph, const sygraph::frontier::Frontier<T, FrontierType>& frontier, LambdaT&& functor) {
  return sygraph::operators::filter::detail::inplace(graph, frontier, std::forward<LambdaT>(functor));
}

template<typename GraphT, typename T, sygraph::frontier::frontier_type FrontierType, typename LambdaT>
sygraph::Event external(GraphT& graph,
                        const sygraph::frontier::Frontier<T, FrontierType>& in,
                        sygraph::frontier::Frontier<T, FrontierType>& out,
                        LambdaT&& functor) {
  return sygraph::operators::filter::detail::external(graph, in, out, std::forward<LambdaT>(functor));
}


} // namespace filter
} // namespace operators
} // namespace v0
} // namespace sygraph