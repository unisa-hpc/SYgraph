/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
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
namespace operators {
namespace intersection {

/**
 * @brief Executes the intersection operation on two input frontiers and stores the result in the output frontier.
 *
 * This function performs the intersection of two input frontiers (`in1` and `in2`) and stores the result in the output
 * frontier (`out`). The operation is executed using a provided functor (`functor`). The function supports different
 * types of frontiers, and currently, it only supports the bitmap frontier type.
 *
 * @tparam GraphT The type of the graph.
 * @tparam T The type of the elements in the frontiers.
 * @tparam FrontierType The type of the frontier (e.g., bitmap).
 * @tparam LambdaT The type of the functor used for the intersection operation.
 *
 * @param graph The graph on which the intersection operation is performed.
 * @param in1 The first input frontier.
 * @param in2 The second input frontier.
 * @param out The output frontier where the result of the intersection is stored.
 * @param functor The functor used to perform the intersection operation.
 *
 * @return An event representing the execution of the intersection operation.
 *
 * @throws std::runtime_error If the frontier type is not implemented.
 */
template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FrontierType, typename LambdaT>
sygraph::Event execute(GraphT& graph,
                       const sygraph::frontier::Frontier<T, FrontierType>& in1,
                       const sygraph::frontier::Frontier<T, FrontierType>& in2,
                       const sygraph::frontier::Frontier<T, FrontierType>& out,
                       LambdaT&& functor) {
  if constexpr (FrontierType == sygraph::frontier::frontier_type::bitmap) {
    return sygraph::operators::intersection::detail::bitmapExecute(graph, in1, in2, out, std::forward<LambdaT>(functor));
  } else {
    throw std::runtime_error("Frontier type not implemented");
  }
}


} // namespace intersection
} // namespace operators
} // namespace sygraph