#pragma once

namespace sygraph {
inline namespace v0 {


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
} // namespace v0
} // namespace sygraph