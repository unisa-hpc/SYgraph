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

private:

  graph::Properties properties;
};

} // namespace graph
} // namespace v0
} // namespace sygraph