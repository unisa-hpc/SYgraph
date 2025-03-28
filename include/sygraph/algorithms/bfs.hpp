/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <memory>

/**
 * @namespace sygraph
 * @brief Namespace for the SYgraph library.
 *
 * The sygraph namespace contains classes and functions for graph algorithms and data structures.
 */
namespace sygraph {
namespace algorithms {
namespace detail {
/**
 * @brief Represents an instance of the Breadth-First Search (BFS) algorithm on a graph.
 *
 * The BFSInstance struct encapsulates the necessary data and operations for performing the BFS algorithm on a graph.
 * It stores the graph, the source vertex, and arrays for distances and parents.
 */
template<typename GraphType>
struct BFSInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

  GraphType& G;      /**< The graph on which the BFS algorithm will be performed. */
  vertex_t source;   /**< The source vertex for the BFS algorithm. */
  edge_t* distances; /**< Array to store the distances from the source vertex to each vertex in the graph. */
  vertex_t* parents; /**< Array to store the parent vertex of each vertex in the graph during the BFS traversal. */

  /**
   * @brief Constructs a BFSInstance object.
   *
   * @param G The graph on which the BFS algorithm will be performed.
   * @param source The source vertex for the BFS algorithm.
   */
  BFSInstance(GraphType& G, vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    // Initialize distances
    distances = memory::detail::memoryAlloc<edge_t, memory::space::shared>(size, queue);
    queue.fill(distances, static_cast<edge_t>(size + 1), size).wait();
    distances[source] = static_cast<edge_t>(0); // Distance from source to itself is 0

    // Initialize parents
    parents = memory::detail::memoryAlloc<vertex_t, memory::space::shared>(size, queue);
    queue.fill(parents, static_cast<vertex_t>(-1), size).wait();
  }

  size_t getVisitedVertices() const {
    size_t vertex_count = G.getVertexCount();
    size_t visited_nodes = 0;
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      if (distances[i] != static_cast<edge_t>(vertex_count + 1)) { visited_nodes++; }
    }
    return visited_nodes;
  }

  size_t getVisitedEdges() const {
    size_t vertex_count = G.getVertexCount();
    size_t visited_edges = 0;
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      if (distances[i] != static_cast<edge_t>(vertex_count + 1)) { visited_edges += G.getDegree(i); }
    }
    return visited_edges;
  }

  /**
   * @brief Destroys the BFSInstance object and frees the allocated memory.
   */
  ~BFSInstance() {
    sycl::free(distances, G.getQueue());
    sycl::free(parents, G.getQueue());
  }
};
} // namespace detail

/**
 * @brief Represents the Breadth-First Search (BFS) algorithm.
 *
 * The BFS algorithm is used to traverse or search a graph in a breadthward motion,
 * starting from a given source vertex. It visits all the vertices at the same level
 * before moving to the next level.
 *
 * @tparam GraphType The type of the graph on which the BFS algorithm will be performed.
 */
template<typename GraphType> // TODO: Implement the getParents method.
class BFS {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

public:
  /**
   * @brief Constructs a BFS object.
   */
  BFS(GraphType& g) : _g(g) {};

  /**
   * @brief Initializes the BFS algorithm with the given graph and source vertex.
   *
   * @param G The graph on which the BFS algorithm will be performed.
   * @param source The source vertex for the BFS algorithm.
   */
  void init(vertex_t& source) { _instance = std::make_unique<detail::BFSInstance<GraphType>>(_g, source); }

  /**
   * @brief Resets the BFS algorithm.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Runs the BFS algorithm.
   *
   * @tparam EnableProfiling A boolean flag to enable profiling.
   * @throws std::runtime_error if the BFS instance is not initialized.
   */
  template<bool EnableProfiling = false>
  void run() {
    if (!_instance) { throw std::runtime_error("BFS instance not initialized"); }

    auto& G = _instance->G;
    auto& source = _instance->source;
    auto& distances = _instance->distances;
    auto& parents = _instance->parents;

    sycl::queue& queue = G.getQueue();

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);
    auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);

    in_frontier.insert(source);

    size_t size = G.getVertexCount();
    int iter = 0;

    // TODO: Add automatic load_balancing for the type of graph.
    while (!in_frontier.empty()) {
      auto e1 = sygraph::operators::advance::frontier<load_balance_t::workgroup_mapped, frontier_view_t::vertex, frontier_view_t::vertex>(
          G, in_frontier, out_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
            if (distances[dst] == size + 1) {
              distances[dst] = iter + 1;
              parents[dst] = src;
              return true;
            }
            return false;
          });
      e1.waitAndThrow();

#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e1, "advance");
#endif

      sygraph::frontier::swap(in_frontier, out_frontier);
      out_frontier.clear();
      iter++;
    }

#ifdef ENABLE_PROFILING
    sygraph::Profiler::addVisitedEdges(_instance->getVisitedEdges());
#endif
  }

  /**
   * @brief Returns the distances from the source vertex to a vertex in the graph.
   *
   * @param vertex The vertex for which to get the distance.
   * @return A pointer to the array of distances.
   */
  edge_t getDistance(size_t vertex) const { return _instance->distances[vertex]; }

  /**
   * @brief Returns the distances from the source vertex to all vertices in the graph.
   *
   * @return A vector of distances.
   */
  std::vector<edge_t> getDistances() const {
    std::vector<edge_t> distances(_instance->G.getVertexCount());
    sycl::queue& queue = _instance->G.getQueue();
    queue.memcpy(distances.data(), _instance->distances, distances.size() * sizeof(edge_t)).wait();
    return distances;
  }

  /**
   * @brief Returns the parent vertices for a vertex in the graph.
   *
   * @param vertex The vertex for which to get the parent vertices.
   * @return A pointer to the array of parent vertices.
   */
  vertex_t getParent(size_t vertex) const { return _instance->parents[vertex]; }

  /**
   * @brief Returns the parent vertices for all vertices in the graph.
   *
   * @return A vector of parent vertices.
   */
  std::vector<vertex_t> getParents() const {
    std::vector<vertex_t> parents(_instance->G.getVertexCount());
    sycl::queue& queue = _instance->G.getQueue();
    queue.memcpy(parents.data(), _instance->parents, parents.size() * sizeof(vertex_t)).wait();
    return parents;
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::BFSInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace sygraph
