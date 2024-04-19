#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

namespace detail {

template <typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
void push(graph_t& graph, const in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  // TODO: Implement push advance
}

} // namespace detail  
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph