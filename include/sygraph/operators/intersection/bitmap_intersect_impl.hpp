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

namespace sygraph {
namespace operators {
namespace intersection {
namespace detail {

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event bitmapExecute(GraphT& graph,
                             const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& in1,
                             const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& in2,
                             const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& out,
                             LambdaT&& functor) {
  out.clear();
  out.merge(in1).wait();
  return out.intersect(in2);
}

} // namespace detail
} // namespace intersection
} // namespace operators
} // namespace sygraph