#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/graph/properties.hpp>


namespace sygraph {
inline namespace v0 {
namespace graph {


template<typename vertex_t, typename edge_t, typename weight_t>
class Graph {
public:
  Graph(Properties properties) : properties(properties) {}
  Graph() = default;

  ~Graph() = default;

  const Properties& getProperties() const { return properties; }

  virtual inline size_t getVertexCount() const = 0;

  virtual inline size_t getEdgeCount() const = 0;

  virtual inline size_t getDegree(vertex_t vertex) const = 0;

  virtual inline vertex_t getFirstNeighbor(vertex_t vertex) const = 0;

  virtual inline vertex_t getSourceVertex(edge_t edge) const = 0;

  virtual inline vertex_t getDestinationVertex(edge_t edge) const = 0;

  virtual inline weight_t getEdgeWeight(edge_t edge) const = 0;

private:
  graph::Properties properties;
};

namespace detail {

template<typename DeviceGraphT>
concept DeviceGraphConcept = requires(DeviceGraphT g) {
  { g.getVertexCount() } -> std::convertible_to<size_t>;
  { g.getEdgeCount() } -> std::convertible_to<size_t>;
  { g.getDegree(std::declval<typename DeviceGraphT::vertex_t>()) } -> std::convertible_to<size_t>;
  { g.getFirstNeighbor(std::declval<typename DeviceGraphT::vertex_t>()) } -> std::convertible_to<typename DeviceGraphT::vertex_t>;
  { g.getSourceVertex(std::declval<typename DeviceGraphT::edge_t>()) } -> std::convertible_to<typename DeviceGraphT::vertex_t>;
  { g.getDestinationVertex(std::declval<typename DeviceGraphT::edge_t>()) } -> std::convertible_to<typename DeviceGraphT::vertex_t>;
  { g.getEdgeWeight(std::declval<typename DeviceGraphT::edge_t>()) } -> std::convertible_to<typename DeviceGraphT::weight_t>;
};

template<typename GraphT>
concept GraphConcept = requires(GraphT g) {
  { g.getQueue() } -> std::convertible_to<sycl::queue>;
  { g.getDeviceGraph() };
} && detail::DeviceGraphConcept<GraphT>;

} // namespace detail

} // namespace graph
} // namespace v0
} // namespace sygraph