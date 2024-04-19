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
  /**
   * TODO: Implement the logic for advancing the frontier.
   * The logic for implementing this class shouldn't take into account any modification of any data structure.
   * That can be done in the user code.
   * This function should only implmement the advance logic.
   * Also, the lambda should be called with the correct arguments (so maybe specifying a signature for the lambda would be a good idea).
   * The lambda also, should return a boolean value indicating if the node/edge should be added to the out frontier (maybe an auxiliar data structure might be needed).
   * How can I do that in order to not lose neither performance nor flexibility? 
  */
}

} // namespace detail  
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph