/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace sygraph {


/**
 * @namespace graph
 * The graph namespace contains classes and functions related to graph structures and algorithms.
 */
namespace graph {


/**
 * @struct Properties
 * @brief A structure to hold properties of a graph.
 *
 * This structure contains properties that define the characteristics of a graph.
 *
 * @var Properties::directed
 * Indicates whether the graph is directed. If true, the graph is directed; otherwise, it is undirected.
 *
 * @var Properties::weighted
 * Indicates whether the graph is weighted. If true, the graph has weights associated with its edges; otherwise, it is unweighted.
 */
struct Properties {
  bool directed{false};
  bool weighted{false};
};

} // namespace graph
} // namespace sygraph