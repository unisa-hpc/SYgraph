#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/graph/properties.hpp>


namespace sygraph {
inline namespace v0 {
namespace graph {


template<typename vertex_t,
         typename edge_t,
         typename weight_t>
class Graph {
public:

  Graph(Properties properties) : properties(properties) {}
  Graph() = default;
  
  ~Graph() = default;

  const Properties& getProperties() const {
    return properties;
  }

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

} // namespace graph
} // namespace v0
} // namespace sygraph