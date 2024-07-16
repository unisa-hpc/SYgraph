#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/graph/properties.hpp>


namespace sygraph {
inline namespace v0 {
namespace graph {


template<typename VertexT, typename EdgeT, typename WeightT>
class Graph {
public:
  Graph(Properties properties) : _properties(properties) {}
  Graph() = default;

  ~Graph() = default;

  const Properties& getProperties() const { return _properties; }

  virtual inline size_t getVertexCount() const = 0;

  virtual inline size_t getEdgeCount() const = 0;

  virtual inline size_t getDegree(VertexT vertex) const = 0;

  virtual inline VertexT getFirstNeighbor(VertexT vertex) const = 0;

  virtual inline VertexT getSourceVertex(EdgeT edge) const = 0;

  virtual inline VertexT getDestinationVertex(EdgeT edge) const = 0;

  virtual inline WeightT getEdgeWeight(EdgeT edge) const = 0;

private:
  graph::Properties _properties;
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
  {
    g.getIntersectionCount(std::declval<typename DeviceGraphT::vertex_t>(),
                           std::declval<typename DeviceGraphT::vertex_t>(),
                           std::function<void(typename DeviceGraphT::vertex_t)>())
  } -> std::convertible_to<size_t>;
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