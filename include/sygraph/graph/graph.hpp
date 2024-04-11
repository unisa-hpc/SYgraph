#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/graph/graph_impl.hpp>
#include <sygraph/graph/properties.hpp>


namespace sygraph {
inline namespace v0 {
namespace graph {


template<typename vertex_t,
         typename edge_t,
         typename weight_t,
         class... graph_view_t>
class Graph : public graph_view_t... {
public:
  using graph_view_t...::graph_view_t;
  Graph(Properties properties) : properties(properties) {}
  
  ~Graph() = default;

private:

  graph::Properties properties;
};

} // namespace graph
} // namespace v0
} // namespace sygraph