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

#include <sygraph/operators/for/impl_for.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace compute {


/**
 * @brief Executes a given functor over a graph and its frontier.
 *
 * This function launches a bitmap kernel to perform computations on the graph
 * using the provided frontier and functor.
 *
 * @tparam GraphT The type of the graph, which must satisfy the GraphConcept.
 * @tparam T The type of the elements in the frontier.
 * @tparam FrontierType The type of the frontier.
 * @tparam LambdaT The type of the functor to be executed.
 *
 * @param graph The graph on which the computation is to be performed.
 * @param frontier The frontier containing the elements to be processed.
 * @param functor The functor to be executed on the graph and frontier.
 *
 * @return An Event object representing the execution of the functor.
 */
template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FrontierType, typename LambdaT>
sygraph::Event execute(GraphT& graph, const sygraph::frontier::Frontier<T, FrontierType>& frontier, LambdaT&& functor) {
  return sygraph::operators::compute::detail::launchBitmapKernel(graph, frontier, functor);
}

} // namespace compute
} // namespace operators
} // namespace v0
} // namespace sygraph