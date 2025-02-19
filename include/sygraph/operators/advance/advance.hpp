/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
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
namespace operators {

namespace advance {

/**
 * @brief Processes the vertices of a graph using a specified functor.
 *
 * This function processes the vertices of the given graph using the provided functor.
 * It utilizes a load balancer to determine the appropriate processing strategy.
 *
 * @tparam GraphT The type of the graph.
 * @tparam T The type of the elements in the frontier.
 * @tparam FrontierType The type of the frontier.
 * @tparam LambdaT The type of the functor.
 * @tparam Lb The load balancer type.
 * @tparam FW The frontier view type.
 *
 * @param graph The graph to be processed.
 * @param out The output frontier where the results will be stored.
 * @param functor The functor to be applied to each vertex.
 *
 * @return A sygraph::Event representing the processing event.
 *
 * @throws std::runtime_error If the load balancer is not implemented.
 */
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

/**
 * @brief Applies a functor to all vertices in the graph.
 *
 * This function iterates over all vertices in the provided graph and applies
 * the given functor to each vertex. It uses a specified load balancer to
 * distribute the work. The output is not stored in any frontier.
 *
 * @tparam Lb The load balancer type to be used.
 * @tparam GraphT The type of the graph.
 * @tparam LambdaT The type of the functor to be applied to each vertex.
 *
 * @param graph The graph whose vertices will be processed.
 * @param functor The functor to be applied to each vertex.
 *
 * @return An event representing the completion of the operation.
 */
template<sygraph::operators::load_balancer Lb, typename GraphT, typename LambdaT>
sygraph::Event vertices(GraphT& graph, LambdaT&& functor) {
  auto out = sygraph::frontier::Frontier<void, sygraph::frontier::frontier_type::none>{};
  return vertices<Lb, sygraph::frontier::frontier_view::none>(graph, out, std::forward<LambdaT>(functor));
}

/**
 * @brief Processes the frontier of a graph using a specified load balancer and functor.
 *
 * This function processes the frontier of a graph using the specified load balancer
 * and functor. It supports different types of load balancers and invokes the appropriate
 * implementation based on the load balancer type.
 *
 * @tparam Lb The type of load balancer to use. Must be one of the values from
 *            `sygraph::operators::load_balancer`.
 * @tparam InView The type of the input frontier view.
 * @tparam OutView The type of the output frontier view.
 * @tparam GraphT The type of the graph.
 * @tparam LambdaT The type of the functor to apply to each element in the frontier.
 * @tparam T The type of the elements in the frontier.
 * @tparam FrontierType The type of the frontier.
 *
 * @param graph The graph to process.
 * @param in The input frontier.
 * @param out The output frontier.
 * @param functor The functor to apply to each element in the frontier.
 *
 * @return A `sygraph::Event` representing the completion of the frontier processing.
 *
 * @throws std::runtime_error If the specified load balancer is not implemented.
 */
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

/**
 * @brief Applies a functor to a graph's frontier using a specified load balancer and input view.
 *
 * This function processes the given frontier of a graph using the provided functor. It utilizes
 * a specified load balancer and input view type to manage the processing. The function returns
 * an event representing the completion of the operation. The output is not stored in any frontier.
 *
 * @tparam Lb The load balancer type to be used.
 * @tparam InView The input view type of the frontier.
 * @tparam GraphT The type of the graph.
 * @tparam LambdaT The type of the functor to be applied.
 * @tparam T The type of elements in the input frontier.
 * @tparam FrontierType The type of the input frontier.
 *
 * @param graph The graph to be processed.
 * @param in The input frontier to be processed.
 * @param functor The functor to be applied to the frontier.
 *
 * @return An event representing the completion of the frontier processing.
 */
template<sygraph::operators::load_balancer Lb,
         frontier::frontier_view InView,
         typename GraphT,
         typename LambdaT,
         typename T,
         frontier::frontier_type FrontierType>
sygraph::Event frontier(GraphT& graph, sygraph::frontier::Frontier<T, FrontierType>& in, LambdaT&& functor) {
  auto out = sygraph::frontier::Frontier<void, sygraph::frontier::frontier_type::none>{};
  return frontier<Lb, InView, sygraph::frontier::frontier_view::none>(graph, in, out, std::forward<LambdaT>(functor));
}

} // namespace advance
} // namespace operators
} // namespace sygraph