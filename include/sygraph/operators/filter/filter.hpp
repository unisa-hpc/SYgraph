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
#include <sygraph/operators/filter/vector_filter_impl.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace filter {

template<typename graph_t,
         typename T,
         typename sygraph::frontier::FrontierView FrontierView,
         typename sygraph::frontier::FrontierType FrontierType,
         typename lambda_t>
sygraph::event inplace(graph_t& graph, const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& frontier, lambda_t&& functor) {
  return sygraph::operators::filter::detail::inplace(graph, frontier, std::forward<lambda_t>(functor));
}

template<typename graph_t,
         typename T,
         typename sygraph::frontier::FrontierView FrontierView,
         typename sygraph::frontier::FrontierType FrontierType,
         typename lambda_t>
sygraph::event external(graph_t& graph,
                        const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& in,
                        const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& out,
                        lambda_t&& functor) {
  return sygraph::operators::filter::detail::external(graph, in, out, std::forward<lambda_t>(functor));
}


} // namespace filter
} // namespace operators
} // namespace v0
} // namespace sygraph