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

  const Properties& get_properties() const {
    return properties;
  }

  virtual inline size_t get_vertex_count() const = 0;

  virtual inline size_t get_edge_count() const = 0;

private:

  graph::Properties properties;
};

} // namespace graph
} // namespace v0
} // namespace sygraph